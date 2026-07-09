"""LLM clients.

Two implementations behind one tiny protocol:

* :class:`OpenAICompatClient` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint (llama.cpp ``llama-server`` or Ollama).
* :class:`MockLLMClient` — a deterministic offline stand-in driven by a
  ``responder`` callable, so the entire agent loop runs (and is tested) without
  a single model weight on disk.

Roles never construct clients directly; the :mod:`coding_agent.registry` builds
them from config and hands them out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import RoleConfig

Message = dict[str, str]
Responder = Callable[[str, list[Message]], str]


class ChatError(RuntimeError):
    """Raised when a chat completion cannot be obtained."""


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


def extract_json(text: str) -> Any:
    """Pull the first well-formed JSON value out of a model response.

    Small models wrap JSON in prose, ```json fences, or emit a stray ``<think>``
    block first. This is deliberately forgiving: fenced block first, then the
    first balanced ``{...}``/``[...]`` span.
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
        for span in _balanced_spans(candidate):
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue
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
                # No matching close from this opener; stop scanning.
                return
            i = j + 1
        else:
            i += 1


@dataclass
class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client (no SDK dependency)."""

    role: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: int = 120
    thinking: bool = False
    api_key: str = "sk-local"  # llama-server ignores this; kept for compatibility

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import requests  # local import so --mock needs no network stack

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if self.thinking:
            # llama-server / vLLM forward this to the Qwen3-family chat template.
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.timeout_s,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface any transport failure uniformly
            raise ChatError(
                f"[{self.role}] chat request to {url} failed: {exc}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(
                f"[{self.role}] malformed response from {url}: {data!r}"
            ) from exc


@dataclass
class MockLLMClient:
    """Offline client. Delegates every call to ``responder(role, messages)``."""

    role: str
    responder: Responder
    model: str = "mock"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.responder(self.role, messages)


def make_client(role_cfg: RoleConfig, *, mock: bool, responder: Responder | None = None) -> LLMClient:
    if mock:
        if responder is None:
            raise ValueError("mock client requires a responder")
        return MockLLMClient(role=role_cfg.name, responder=responder, model="mock")
    return OpenAICompatClient(
        role=role_cfg.name,
        base_url=role_cfg.base_url,
        model=role_cfg.model,
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        timeout_s=role_cfg.timeout_s,
        thinking=role_cfg.thinking,
    )
