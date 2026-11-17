"""Role wrappers — one thin class per model job: prompt in, parsed object out.

The division of labour in the multi-model agent:

* **Planner** (Qwen3.5-2B) owns the conversation and decides the next move.
  It *never* writes code.
* **Coder** (Qwen2.5-Coder-3B) turns one concrete instruction into one code
  action. It never decides strategy. (In single-model mode it does both.)
* **Dispatcher** (LFM2-1.2B-Tool) formats a loose intended action into an exact
  tool call. Only used when the deterministic mapping can't.
* **Extractor** (LFM2-350M-Extract) compresses *long* tool output into small
  JSON before it enters the planner's context. It summarises; it never decides
  pass/fail — the harness owns facts like exit codes.
* **RAG** (MiniCPM5-1B, or the planner's model on small GPUs) answers questions
  about the codebase over retrieved chunks.

Each class formats a prompt, calls its client, and parses the result. All
prompts live in this file so they are easy to read side by side and tune.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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

On each turn, read the task, the workspace and the observations so far, then \
output exactly ONE JSON object describing the next move:

{
  "thought": "<one sentence of reasoning>",
  "next": "coder" | "tool" | "rag" | "finish",
  "instruction": "<concrete step for the coder>",   // when next == "coder"
  "tool": {"name": "run_shell", "arguments": {"command": "..."}},  // when next == "tool"
  "question": "<repo question>",                     // when next == "rag"
  "summary": "<what was accomplished>"               // when next == "finish"
}

Tools you can call directly with "tool":
- run_shell(command)      e.g. the test command from the workspace section
- read_file(path)         a file, or "." to list the repo
- search_codebase(query)  only if retrieval is enabled

Rules:
- Delegate all code changes to the coder via "coder" (one file, one change).
- Each observation starts with a line of FACTS from the harness (exit code, \
PASSED/FAILED). Trust that line over any summary below it.
- You may ONLY choose "finish" after the test command PASSED in this run.
- If the same result keeps coming back, change approach instead of repeating.
Output JSON only. No prose outside the JSON object.
"""


@dataclass
class PlannerDecision:
    thought: str
    next: str
    instruction: str = ""
    tool: Any = None
    question: str = ""
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "PlannerDecision":
        if not isinstance(obj, dict):
            raise ChatError(f"planner reply was not a JSON object: {str(obj)[:120]!r}")
        decision = cls(
            thought=str(obj.get("thought", "")),
            next=str(obj.get("next", "")).strip().lower(),
            instruction=str(obj.get("instruction", "")),
            tool=obj.get("tool"),
            question=str(obj.get("question", "")),
            summary=str(obj.get("summary", "")),
            raw=obj,
        )
        # Small planners often name the tool in "next" ({"next": "read_file",
        # "arguments": {...}}). Same intent, different shape: fix it in code.
        decision.next = _NEXT_ALIASES.get(decision.next, decision.next)
        if decision.next in _PLANNER_TOOL_NAMES:
            decision.tool = decision.tool if isinstance(decision.tool, dict) else decision.next
            decision.next = "tool"
        return decision

    def tool_call(self) -> "ToolCall":
        """The requested tool call, whichever shape the model used."""

        if isinstance(self.tool, dict):
            return ToolCall.from_obj(self.tool)
        name = self.tool if isinstance(self.tool, str) else ""
        args = self.raw.get("arguments")
        if not isinstance(args, dict):
            args = {k: self.raw[k] for k in ("path", "command", "query") if k in self.raw}
        return ToolCall(name=name.strip(), arguments=args)


_PLANNER_TOOL_NAMES = {"read_file", "run_shell", "search_codebase", "write_file", "patch_file"}
_NEXT_ALIASES = {"read": "read_file", "shell": "run_shell", "run": "run_shell",
                 "search": "search_codebase", "code": "coder", "done": "finish"}


class Planner:
    def __init__(self, client: LLMClient):
        self.client = client

    # How many recent observations the planner sees. Older ones fall off: a
    # small model's attention degrades long before its context window fills.
    MAX_OBSERVATIONS = 8

    def decide(
        self,
        task: str,
        observations: list[str],
        tests_passed: bool,
        workspace: str = "",
        turn: int | None = None,
        max_turns: int | None = None,
        budget_chars: int | None = None,
    ) -> PlannerDecision:
        recent = observations[-self.MAX_OBSERVATIONS:]
        if budget_chars:
            # Keep the whole prompt inside the served context: drop the oldest
            # observations first (an overflow is a hard HTTP 400, not a hint).
            room = budget_chars - len(PLANNER_SYSTEM) - len(task) - len(workspace) - 300
            room = max(room, 1500)
            while len(recent) > 1 and sum(len(o) + 2 for o in recent) > room:
                recent = recent[1:]
            if recent and len(recent[0]) > room:
                recent = [recent[0][: room - 20] + "\n...[clipped]"]
        obs_block = "\n\n".join(recent) or "(no observations yet)"
        # Static sections (task, workspace snapshot) first, volatile ones
        # (observations) last: llama-server reuses the KV cache for a prompt
        # prefix it has already seen, so stable-first ordering is faster.
        parts = [f"TASK:\n{task}"]
        if workspace:
            parts.append(f"WORKSPACE:\n{workspace}")
        parts.append(f"OBSERVATIONS (newest last):\n{obs_block}")
        status = f"TESTS_PASSED_THIS_RUN: {tests_passed}"
        if turn is not None and max_turns is not None:
            status += f"\nTURN: {turn} of {max_turns}"
        parts.append(status)
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
#
# Lesson learned from traces: the examples below are copied LITERALLY by a 3B
# model. An earlier one-line summary ("header {...} + <old_text>...</old_text>")
# made the model write whole patches on one line, "+" included, which broke
# every file it touched. Show the exact multi-line shape you want.
#
# Second lesson: keep examples UNRELATED to real tasks. When the example was a
# `safe_divide` fix in calc.py, the model asked to fix calc.py's safe_divide
# pasted the example's old_text instead of the file's — and an eval task about
# that same function was quietly given its answer.
_CODER_FORMAT = """Reply with exactly ONE action: a one-line JSON header, then any \
code in raw tags on their own lines (never escape code inside JSON). The examples \
below are about an imaginary file; always use the real file's text.

Read a file (do this before editing a file you have not seen):
{"rationale": "see the current code", "action": "read_file", "path": "shop/cart.py"}

Edit part of an existing file. Copy the lines to replace EXACTLY from the file, \
with the same indentation; they must appear exactly once:
{"rationale": "apply the discount", "action": "patch_file", "path": "shop/cart.py"}
<old_text>
    def total(self):
        return sum(item.price for item in self.items)
</old_text>
<new_text>
    def total(self, discount=0.0):
        subtotal = sum(item.price for item in self.items)
        return subtotal * (1 - discount)
</new_text>

Create a new file, or rewrite a small file completely:
{"rationale": "new helper module", "action": "write_file", "path": "shop/money.py"}
<content>
def to_cents(amount):
    return round(amount * 100)
</content>

Run a command:
{"rationale": "check the change", "action": "run_shell", "command": "python -m pytest -q"}"""

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


# Single-model agent: the coder ALONE, owning its own loop. Also the
# experimental control for the multi-model agent (same harness, no team).
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

    def act(self, instruction: str, context: str = "", task: str = "") -> CoderAction:
        # The overall task rides along: a planner's paraphrase can drop the one
        # detail that matters ("return 0.0", not "return an error message").
        user = f"OVERALL TASK (for context):\n{task}\n\n" if task else ""
        user += f"YOUR INSTRUCTION:\n{instruction}\n"
        if context:
            user += f"\nCONTEXT (recent files and tool output, newest last):\n{context}\n"
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
        parts.append(f"RECENT OUTPUT (newest last):\n{context or '(nothing yet)'}")
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
    def from_obj(cls, obj: Any) -> "ToolCall":
        """Normalise the many shapes small models use for "a tool call".

        Accepted: ``{"name": n, "arguments": {...}}`` (canonical), the same with
        ``args``/``parameters``/``input``, flat ``{"name": n, "command": ...}``,
        and keyed ``{"run_shell": {"command": ...}}``. Fixing these with a few
        lines of code is cheaper and more reliable than another model call.
        """

        if not isinstance(obj, dict):
            return cls(name="", arguments={})
        name = str(obj.get("name") or obj.get("tool") or "").strip()
        args: Any = None
        for key in ("arguments", "args", "parameters", "input"):
            if isinstance(obj.get(key), dict):
                args = obj[key]
                break
        if not name and len(obj) == 1:
            (only_key, only_val), = obj.items()
            if isinstance(only_val, dict):
                name, args = str(only_key), only_val
        if args is None:
            skip = {"name", "tool", "arguments", "args", "parameters", "input"}
            args = {k: v for k, v in obj.items() if k not in skip}
        if isinstance(args, str):  # e.g. arguments as a JSON string (OpenAI style)
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return cls(name=name, arguments=args if isinstance(args, dict) else {})


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

# Default schema for long command output. Note what is NOT here: "did it pass?".
# The harness knows the exit code; asking a 350M model to re-derive it only adds
# a way to be wrong.
DEFAULT_EXTRACT_SCHEMA = {
    "failing_test": "name of the first failing test, or null",
    "file": "file path implicated, or null",
    "line": "line number implicated, or null",
    "error_type": "exception/error class, or null",
    "message": "one-line summary of the most important error or result",
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
            return {"message": _head_tail(raw_output.strip(), 400)}
        if not isinstance(obj, dict):
            return {"message": str(obj)}
        # Even if the model invents a verdict key, it doesn't get a vote.
        for key in ("tests_passed", "passed", "success", "ok"):
            obj.pop(key, None)
        return obj


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
