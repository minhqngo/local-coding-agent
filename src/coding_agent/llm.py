"""LLM clients.

Two implementations behind one tiny protocol (``chat(messages) -> str``):

* :class:`OpenAICompatClient` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (llama.cpp ``llama-server``, Ollama, LM Studio,
  vLLM). No SDK; just ``requests``.
* :class:`MockLLMClient` — a deterministic offline stand-in driven by a
  ``responder`` callable, so the whole agent runs (and is tested) without a
  single model weight on disk.

Two kinds of failure, deliberately kept apart
---------------------------------------------
* :class:`ChatError` — the model *answered*, but uselessly: no JSON, empty, cut
  off. The agent loop recovers by telling the model what went wrong and asking
  again. This is normal with 1–3B models.
* :class:`ModelUnavailable` — we never got an answer: server down, timeout,
  HTTP error, prompt larger than the context window. Asking the model again
  cannot fix any of these, so the loop stops and tells the human.

Confusing the two is a classic agent bug: a dead server looks like a "malformed
response", and the loop politely retries it dozens of times.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import RoleConfig

Message = dict[str, str]
Responder = Callable[[str, list[Message]], str]


class ChatError(RuntimeError):
    """The model replied, but not usefully. Recoverable: tell it and retry."""


class ModelUnavailable(RuntimeError):
    """No usable reply could be obtained at all. Not recoverable by the model."""

    def __init__(self, role: str, message: str, hint: str = ""):
        self.role = role
        self.hint = hint
        super().__init__(f"[{role}] {message}" + (f"\n  hint: {hint}" if hint else ""))


class LLMClient(Protocol):
    role: str

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        ...


class TraceSink(Protocol):
    def model_call(self, **fields: Any) -> None:
        ...


# --------------------------------------------------------------------------- #
# Parsing model output
# --------------------------------------------------------------------------- #

def extract_json(text: str) -> Any:
    """Pull the first well-formed JSON value out of a model response.

    Small models wrap JSON in prose, ```json fences, or emit a stray ``<think>``
    block first. This is deliberately forgiving: fenced block first, then the
    first balanced ``{...}``/``[...]`` span that parses.
    """

    if text is None:
        raise ChatError("empty response")

    # Strip closed Qwen-style thinking blocks, then any UNTERMINATED one (a
    # thinking model truncated by max_tokens leaves an open <think> whose stray
    # braces must not be mistaken for the decision JSON).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()

    # 1) fenced ```json ... ``` (or bare ``` ... ```)
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try every balanced span in order — the first may be invalid while a
        # later one is the real payload (e.g. "pseudo {a:1} then {\"ok\":true}").
        # Prefer an object: in "Step [1]: {...}" the [1] is prose, not the answer.
        first_other = None
        for span in _balanced_spans(candidate):
            try:
                value = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
            if first_other is None:
                first_other = (value,)
        if first_other is not None:
            return first_other[0]
    raise ChatError(f"no JSON object found in response: {text[:200]!r}")


def _balanced_spans(text: str):
    """Yield each balanced {...} / [...] substring in order, respecting strings."""

    starts = {"{": "}", "[": "]"}
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in starts:
            close = starts[ch]
            depth = 0
            in_str = False
            escape = False
            for j in range(i, n):
                c = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
            else:
                # No matching close from this opener (e.g. prose "Note [unclosed."):
                # skip just this character and keep looking for a real span.
                i += 1
                continue
            i = j + 1
        else:
            i += 1


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #

def _server_error_text(resp: Any) -> str:
    try:
        body = resp.json()
        err = body.get("error", body)
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    except Exception:  # noqa: BLE001 - best effort
        return getattr(resp, "text", "")[:300]


@dataclass
class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client (no SDK dependency)."""

    role: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: int = 120
    thinking: bool | None = None
    api_key: str = "sk-local"  # llama-server ignores this; kept for compatibility
    tracer: TraceSink | None = None
    connect_retries: int = 1   # retry once if the server is momentarily unreachable

    def _post(self, url: str, payload: dict[str, Any]) -> Any:
        import requests  # local import so --mock needs no network stack

        for attempt in range(self.connect_retries + 1):
            try:
                return requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout_s,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            # Order matters: ConnectTimeout is both a ConnectionError and a
            # Timeout, and means "nobody is listening", not "model too slow".
            except requests.exceptions.ConnectionError as exc:
                if attempt < self.connect_retries:
                    time.sleep(1.0)
                    continue
                raise ModelUnavailable(
                    self.role,
                    f"cannot reach {url}: {exc.__class__.__name__}",
                    "is the server running? try `coding-agent serve status`",
                ) from exc
            except requests.exceptions.Timeout as exc:
                raise ModelUnavailable(
                    self.role,
                    f"no reply from {url} within {self.timeout_s}s",
                    "the model is too slow for this timeout: raise timeout_s in the "
                    "profile, lower max_tokens, or turn thinking off",
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise ModelUnavailable(self.role, f"request to {url} failed: {exc}") from exc

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        budget = self.max_tokens if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": budget,
            "stream": False,
        }
        if self.thinking is not None:
            # llama-server / vLLM forward this to the chat template; Qwen3-family
            # templates switch thinking on/off with it, others ignore it.
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking}

        started = time.monotonic()
        trace: dict[str, Any] = {"role": self.role, "model": self.model, "messages": messages}
        try:
            resp = self._post(url, payload)
            status = getattr(resp, "status_code", 200)
            if status >= 400:
                detail = _server_error_text(resp)
                hint = ""
                if "context" in detail.lower() or "too long" in detail.lower():
                    hint = (f"the prompt is larger than the served context window; "
                            f"raise ctx for '{self.role}' in the profile")
                raise ModelUnavailable(self.role, f"HTTP {status} from {url}: {detail}", hint)
            data = resp.json()
            try:
                choice = data["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelUnavailable(
                    self.role, f"malformed response from {url}: {str(data)[:300]}"
                ) from exc

            content = message.get("content") or ""
            # llama-server returns thinking separately when it parses it out.
            reasoning = message.get("reasoning_content") or ""
            finish = choice.get("finish_reason")
            trace.update(response=content, reasoning=reasoning,
                         finish_reason=finish, usage=data.get("usage"))
            if not content.strip():
                if finish == "length":
                    raise ChatError(
                        f"empty answer: the model used its whole max_tokens={budget} "
                        "budget before answering (usually thinking). Raise max_tokens "
                        "or set thinking: false for this role."
                    )
                raise ChatError("empty answer from the model")
            return content
        except (ChatError, ModelUnavailable) as exc:
            trace["error"] = str(exc)
            raise
        finally:
            trace["latency_s"] = round(time.monotonic() - started, 3)
            if self.tracer is not None:
                self.tracer.model_call(**trace)


@dataclass
class MockLLMClient:
    """Offline client. Delegates every call to ``responder(role, messages)``."""

    role: str
    responder: Responder
    model: str = "mock"
    tracer: TraceSink | None = None

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        reply = self.responder(self.role, messages)
        if self.tracer is not None:
            self.tracer.model_call(role=self.role, model=self.model, messages=messages,
                                   response=reply, latency_s=0.0)
        return reply


def make_client(
    role_cfg: RoleConfig,
    *,
    mock: bool,
    responder: Responder | None = None,
    tracer: TraceSink | None = None,
) -> LLMClient:
    if mock:
        if responder is None:
            raise ValueError("mock client requires a responder")
        return MockLLMClient(role=role_cfg.name, responder=responder, model="mock", tracer=tracer)
    return OpenAICompatClient(
        role=role_cfg.name,
        base_url=role_cfg.base_url,
        model=role_cfg.model,
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        timeout_s=role_cfg.timeout_s,
        thinking=role_cfg.thinking,
        tracer=tracer,
    )
