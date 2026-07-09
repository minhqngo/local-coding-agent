"""Role wrappers — one thin class per model, each with a single job.

The division of labour (from the design plan):

* **Planner** (Qwen3.5-2B, thinking) owns the conversation and decides the next
  move. It *never* writes code.
* **Coder** (Qwen2.5-Coder-3B) turns one concrete instruction into a code
  action. It never decides strategy.
* **Dispatcher** (LFM2-1.2B-Tool) formats a loose intended-action into an exact
  tool call. Deterministic, temperature 0.
* **Extractor** (LFM2-350M-Extract) compresses raw tool output into small
  structured JSON before it re-enters the planner's context.
* **RAG** (LFM2-1.2B-RAG) answers questions about the codebase over retrieved
  chunks.

Each class formats a prompt, calls its client, and parses the result. Prompts
live here so they are easy to tune per model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import ChatError, LLMClient, Message, extract_json

# Max chars of raw output the extractor sees. Kept head+tail because pytest puts
# its verdict ("N failed, M passed") at the TAIL of long output.
_EXTRACT_INPUT_CHARS = 8000


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[truncated {len(text) - limit} chars]...\n{text[-half:]}"


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = """You are the PLANNER of a small coding agent. You orchestrate \
specialists; you never write code yourself.

On each turn, read the task and the compressed observations so far, then output \
exactly ONE JSON object describing the next move:

{
  "thought": "<one sentence of reasoning>",
  "next": "coder" | "tool" | "rag" | "finish",
  "instruction": "<concrete step for the coder>",   // when next == "coder"
  "tool": {"name": "run_shell", "arguments": {"command": "..."}},  // when next == "tool"
  "question": "<repo question>",                     // when next == "rag"
  "summary": "<what was accomplished>"               // when next == "finish"
}

Rules:
- Delegate all code changes to the coder via "coder".
- Run shell/tests yourself via "tool" (e.g. the test command).
- You may ONLY choose "finish" after the tests have been run and passed in this run.
- Keep instructions to the coder small and concrete (one file / one change).
Output JSON only. No prose outside the JSON object.
"""


@dataclass
class PlannerDecision:
    thought: str
    next: str
    instruction: str = ""
    tool: dict[str, Any] | None = None
    question: str = ""
    summary: str = ""

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "PlannerDecision":
        return cls(
            thought=str(obj.get("thought", "")),
            next=str(obj.get("next", "")).strip().lower(),
            instruction=str(obj.get("instruction", "")),
            tool=obj.get("tool"),
            question=str(obj.get("question", "")),
            summary=str(obj.get("summary", "")),
        )


class Planner:
    def __init__(self, client: LLMClient):
        self.client = client

    def decide(
        self,
        task: str,
        observations: list[str],
        tests_passed: bool,
        workspace: str = "",
    ) -> PlannerDecision:
        obs_block = "\n\n".join(observations[-8:]) or "(no observations yet)"
        # Static sections (task, workspace snapshot) first, volatile ones
        # (observations) last — cache-friendly prompt ordering.
        parts = [f"TASK:\n{task}"]
        if workspace:
            parts.append(f"WORKSPACE:\n{workspace}")
        parts.append(f"TESTS_PASSED_THIS_RUN: {tests_passed}")
        parts.append(f"OBSERVATIONS (compressed, newest last):\n{obs_block}")
        parts.append("Respond with the next-move JSON object.")
        messages: list[Message] = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw = self.client.chat(messages)
        return PlannerDecision.from_obj(extract_json(raw))


# --------------------------------------------------------------------------- #
# Coder
# --------------------------------------------------------------------------- #

# Small models reliably fail to escape newlines/quotes inside JSON strings, so
# multi-line bodies (file content, patch text) travel in raw XML-style tags
# instead. A small JSON header carries the action + path; the tags carry the
# text verbatim. Pure JSON is still accepted (the mock brain and tests use it).
_CODER_FORMAT = """Respond with a small JSON header, then put any multi-line text \
(file contents, patch text) in raw tags so you never have to escape it:

{"rationale": "<one sentence>", "action": "write_file", "path": "calc.py"}
<content>
def safe_divide(a, b):
    ...
</content>

Actions:
- read_file:  {"action": "read_file", "path": "f.py"}
- write_file: header {"action":"write_file","path":"f.py"} + <content>FULL FILE</content>
- patch_file: header {"action":"patch_file","path":"f.py"} + <old_text>EXACT SNIPPET</old_text><new_text>REPLACEMENT</new_text>
- run_shell:  {"action": "run_shell", "command": "python -m pytest -q"}

Prefer patch_file for a small edit to an existing file (old_text must appear \
EXACTLY ONCE); use write_file for a brand-new file (emit the COMPLETE contents). \
A single small action may be pure JSON with no tags."""

CODER_SYSTEM = (
    "You are the CODER of a small coding agent. You receive ONE concrete "
    "instruction and any relevant context, and you perform a single code action.\n\n"
    + _CODER_FORMAT
    + "\n\nYou do not decide overall strategy — just execute this one instruction well."
)


@dataclass
class CoderAction:
    rationale: str
    action: str
    path: str = ""
    content: str = ""
    command: str = ""
    old_text: str = ""
    new_text: str = ""

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "CoderAction":
        return cls(
            rationale=str(obj.get("rationale", "")),
            action=str(obj.get("action", "")).strip().lower(),
            path=str(obj.get("path", "")),
            content=obj.get("content", "") or "",
            command=str(obj.get("command", "")),
            old_text=obj.get("old_text", "") or "",
            new_text=obj.get("new_text", "") or "",
        )


def _extract_tag(raw: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(?P<body>.*?)</{tag}>", raw, re.S)
    if not m:
        return None
    body = m.group("body")
    # Drop the single framing newline the block format usually adds, top & tail.
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n"):
        body = body[:-1]
    return body


def parse_coder_response(raw: str) -> CoderAction:
    """Parse the hybrid header+tags format, or plain JSON (backward compatible)."""

    tags = {t: _extract_tag(raw, t) for t in ("content", "old_text", "new_text")}
    # Strip tag blocks before JSON parsing so braces inside file content can't
    # confuse the balanced-span scanner.
    header = raw
    for t in ("content", "old_text", "new_text"):
        header = re.sub(rf"<{t}>.*?</{t}>", "", header, flags=re.S)
    obj = extract_json(header)
    if not isinstance(obj, dict):
        raise ChatError(f"coder response was not a JSON object: {raw[:120]!r}")
    for key, value in tags.items():
        if value is not None:
            obj[key] = value
    return CoderAction.from_obj(obj)


# Phase-2 baseline: the coder ALONE, owning its own loop. This is the
# experimental control — no planner, no extractor, no dispatcher.
CODER_BASELINE_SYSTEM = (
    "You are a coding agent working ALONE. You are given a task and the raw "
    "output of your previous actions, and you fix the task yourself.\n\n"
    + _CODER_FORMAT
    + '\n\nAlso available: {"action": "finish"} — use it ONLY once the tests pass. '
    "Read files before editing them; run the test command to verify your fix."
)


class Coder:
    def __init__(self, client: LLMClient):
        self.client = client

    def act(self, instruction: str, context: str = "") -> CoderAction:
        user = f"INSTRUCTION:\n{instruction}\n"
        if context:
            user += f"\nCONTEXT:\n{context}\n"
        user += "\nRespond with a single code action."
        messages: list[Message] = [
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = self.client.chat(messages)
        return parse_coder_response(raw)

    def act_alone(
        self, task: str, context: str, test_command: str, workspace: str = ""
    ) -> CoderAction:
        """Baseline control: the coder drives itself with no glue models."""

        parts = [f"TASK:\n{task}", f"TEST COMMAND: {test_command}"]
        if workspace:
            parts.append(f"WORKSPACE:\n{workspace}")
        parts.append(f"RECENT OUTPUT:\n{context or '(nothing yet)'}")
        parts.append("Respond with your next single code action.")
        user = "\n\n".join(parts)
        messages: list[Message] = [
            {"role": "system", "content": CODER_BASELINE_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = self.client.chat(messages)
        return parse_coder_response(raw)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

DISPATCHER_SYSTEM = """You convert a described intent into ONE exact tool call. \
Output JSON only: {"name": "<tool>", "arguments": {...}}.

Available tools:
- read_file(path)
- write_file(path, content)
- patch_file(path, old_text, new_text)
- run_shell(command)
- search_codebase(query)

Choose the single most appropriate tool and fill its arguments precisely. Do not \
invent arguments the tool does not accept. Output JSON only.
"""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "ToolCall":
        name = str(obj.get("name", "")).strip()
        args = obj.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        return cls(name=name, arguments=args)


class Dispatcher:
    """Formats a coder action (or free-form intent) into a strict tool call."""

    def __init__(self, client: LLMClient):
        self.client = client

    def format_call(self, intent: str) -> ToolCall:
        messages: list[Message] = [
            {"role": "system", "content": DISPATCHER_SYSTEM},
            {"role": "user", "content": f"INTENT:\n{intent}\n\nEmit the tool-call JSON."},
        ]
        raw = self.client.chat(messages, temperature=0.0)
        return ToolCall.from_obj(extract_json(raw))

    @staticmethod
    def from_coder_action(action: CoderAction) -> ToolCall:
        """Deterministic mapping used when the coder already produced a clean action.

        The LLM dispatcher exists for messy/free-form intents; when the coder's
        JSON is already well-formed we normalise it directly and skip a hop.
        """

        if action.action == "write_file":
            return ToolCall("write_file", {"path": action.path, "content": action.content})
        if action.action == "patch_file":
            return ToolCall(
                "patch_file",
                {"path": action.path, "old_text": action.old_text, "new_text": action.new_text},
            )
        if action.action == "read_file":
            return ToolCall("read_file", {"path": action.path})
        if action.action == "run_shell":
            return ToolCall("run_shell", {"command": action.command})
        raise ValueError(f"cannot map coder action {action.action!r} to a tool call")


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

EXTRACTOR_SYSTEM = """You compress raw command/tool output into a small JSON \
object matching the requested schema. Discard noise; keep only what the planner \
needs to decide the next step. Output JSON only, matching the schema keys.
"""

# Default schema for test/command output.
DEFAULT_EXTRACT_SCHEMA = {
    "tests_passed": "boolean — true only if the test run clearly passed",
    "failing_test": "name of the first failing test, or null",
    "file": "file path implicated, or null",
    "line": "line number implicated, or null",
    "error_type": "exception/error class, or null",
    "message": "one-line human summary of the outcome",
}


class Extractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def compress(self, raw_output: str, schema: dict[str, str] | None = None) -> dict[str, Any]:
        schema = schema or DEFAULT_EXTRACT_SCHEMA
        user = (
            "SCHEMA (key: meaning):\n"
            + json.dumps(schema, indent=2)
            + "\n\nRAW OUTPUT:\n"
            + _head_tail(raw_output, _EXTRACT_INPUT_CHARS)
            + "\n\nReturn a JSON object with exactly the schema keys."
        )
        messages: list[Message] = [
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            raw = self.client.chat(messages, temperature=0.0)
            obj = extract_json(raw)
        except ChatError:
            # Degrade gracefully — a garbled extractor must never abort the run.
            # Fall back to a truncated-raw observation the planner can still read.
            return {"message": _head_tail(raw_output.strip(), 400), "tests_passed": False}
        return obj if isinstance(obj, dict) else {"message": str(obj)}


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #

RAG_SYSTEM = """You answer questions about a codebase using ONLY the provided \
context chunks. Be concise and cite file paths. If the answer is not in the \
context, say so plainly.
"""


class RepoQA:
    def __init__(self, client: LLMClient):
        self.client = client

    def answer(self, question: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks) if chunks else "(no context retrieved)"
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        messages: list[Message] = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user},
        ]
        return self.client.chat(messages)
