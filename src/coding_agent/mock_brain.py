"""Deterministic offline brain for ``--mock`` mode.

One :class:`MockBrain` instance is shared across every mock role client, so the
whole planner→coder→dispatcher→sandbox→extractor loop runs end-to-end with no
network and no weights. It is used by the test suite and by the trivial-bug
example.

The brain is *not* trying to be smart — it is a small, predictable state machine:

1. planner: delegate the next scripted coder action; once actions run out,
   run the test command; finish when observations show tests passed.
2. coder: pop the next scripted :class:`CoderAction` (supplied per task).
3. dispatcher: mirror the strict coder→tool mapping.
4. extractor: heuristically parse command output into the schema.
5. rag: echo a canned answer over the provided context.

Because it is deterministic, tests assert exact loop behaviour; because it is
data-driven (``coder_actions``), examples inject the real fix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import Message, extract_json


@dataclass
class MockBrain:
    """A scriptable stand-in for the whole model fleet."""

    # Ordered code actions the mock "coder" will emit, one per delegation.
    coder_actions: list[dict[str, Any]] = field(default_factory=list)
    test_command: str = "python -m pytest -q"
    # Canned RAG answer (tests can override).
    rag_answer: str = "Based on the retrieved chunks, the relevant logic lives in the cited files."
    # When True, the coder drives the whole loop itself (Phase-2 baseline):
    # scripted edits → run the test command → finish.
    single_model: bool = False

    _coder_idx: int = 0
    _baseline_tested: bool = False

    # -- public: the Responder signature the clients call -------------------- #
    def __call__(self, role: str, messages: list[Message]) -> str:
        handler = {
            "planner": self._planner,
            "coder": self._coder,
            "dispatcher": self._dispatcher,
            "extractor": self._extractor,
            "rag": self._rag,
        }.get(role)
        if handler is None:
            return json.dumps({"message": f"mock: no handler for role {role}"})
        return handler(messages)

    # -- planner ------------------------------------------------------------- #
    def _planner(self, messages: list[Message]) -> str:
        user = _last_user(messages)
        tests_passed = "TESTS_PASSED_THIS_RUN: True" in user

        if tests_passed:
            return json.dumps(
                {
                    "thought": "Tests pass; nothing more to do.",
                    "next": "finish",
                    "summary": "Applied the scripted fix and verified the test command passes.",
                }
            )

        if self._coder_idx < len(self.coder_actions):
            action = self.coder_actions[self._coder_idx]
            instruction = action.get(
                "instruction",
                f"Apply change #{self._coder_idx + 1} to fix the failing tests.",
            )
            return json.dumps(
                {
                    "thought": "Delegate the next concrete edit to the coder.",
                    "next": "coder",
                    "instruction": instruction,
                }
            )

        # All scripted edits applied — run the tests.
        return json.dumps(
            {
                "thought": "All edits applied; run the test command to verify.",
                "next": "tool",
                "tool": {"name": "run_shell", "arguments": {"command": self.test_command}},
            }
        )

    # -- coder --------------------------------------------------------------- #
    def _coder(self, messages: list[Message]) -> str:
        if self.single_model:
            return self._baseline_coder()
        if self._coder_idx < len(self.coder_actions):
            action = dict(self.coder_actions[self._coder_idx])
            self._coder_idx += 1
            action.setdefault("rationale", "Apply the scripted fix.")
            action.pop("instruction", None)
            return json.dumps(action)
        # Nothing scripted: a harmless read so the loop stays well-formed.
        return json.dumps(
            {"rationale": "No scripted action; inspect the repo.", "action": "read_file", "path": "."}
        )

    def _baseline_coder(self) -> str:
        """Coder-only baseline: scripted edits → run tests → finish."""

        if self._coder_idx < len(self.coder_actions):
            action = dict(self.coder_actions[self._coder_idx])
            self._coder_idx += 1
            action.setdefault("rationale", "Apply the scripted fix.")
            action.pop("instruction", None)
            return json.dumps(action)
        if not self._baseline_tested:
            self._baseline_tested = True
            return json.dumps(
                {"rationale": "Verify the fix.", "action": "run_shell", "command": self.test_command}
            )
        return json.dumps({"rationale": "Tests pass; done.", "action": "finish"})

    # -- dispatcher ---------------------------------------------------------- #
    def _dispatcher(self, messages: list[Message]) -> str:
        """Recover the intent from the prompt and emit a strict tool call."""

        user = _last_user(messages)
        try:
            obj = extract_json(user)
        except Exception:  # noqa: BLE001
            obj = {}
        # Under dispatch_all the intent is already a formed tool call
        # {"name", "arguments"} — pass it straight through.
        if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            return json.dumps({"name": obj["name"], "arguments": obj["arguments"]})
        # Otherwise it's a coder action {action, path, content, command}.
        action = str(obj.get("action", "")).lower()
        if action == "write_file":
            return json.dumps({"name": "write_file", "arguments": {"path": obj.get("path", ""), "content": obj.get("content", "")}})
        if action == "patch_file":
            return json.dumps({"name": "patch_file", "arguments": {
                "path": obj.get("path", ""),
                "old_text": obj.get("old_text", ""),
                "new_text": obj.get("new_text", ""),
            }})
        if action == "read_file":
            return json.dumps({"name": "read_file", "arguments": {"path": obj.get("path", "")}})
        if obj.get("command"):
            return json.dumps({"name": "run_shell", "arguments": {"command": obj["command"]}})
        # Fall back to searching the codebase for whatever text we saw.
        return json.dumps({"name": "search_codebase", "arguments": {"query": user[:120]}})

    # -- extractor ----------------------------------------------------------- #
    def _extractor(self, messages: list[Message]) -> str:
        user = _last_user(messages)
        raw = _section(user, "RAW OUTPUT:")
        return json.dumps(_heuristic_extract(raw))

    # -- rag ----------------------------------------------------------------- #
    def _rag(self, messages: list[Message]) -> str:
        return self.rag_answer


def _last_user(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _section(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text


def _heuristic_extract(raw: str) -> dict[str, Any]:
    """Best-effort parse of pytest/command output into the default schema."""

    lower = raw.lower()
    passed = bool(re.search(r"\bpassed\b", lower)) and not re.search(r"\bfailed\b|\berror\b", lower)
    # pytest summary like "1 failed, 2 passed"
    if re.search(r"\d+\s+failed", lower):
        passed = False

    failing = None
    m = re.search(r"(?:FAILED|failed)\s+([\w/\.]+::[\w\[\]-]+)", raw)
    if m:
        failing = m.group(1)

    file_ = None
    line = None
    fm = re.search(r'File "([^"]+)", line (\d+)', raw) or re.search(r"([\w/\.]+):(\d+):", raw)
    if fm:
        file_, line = fm.group(1), int(fm.group(2))

    err = None
    em = re.search(r"\b([A-Z][A-Za-z]*Error|Exception)\b", raw)
    if em:
        err = em.group(1)

    if passed:
        message = "Test command passed."
    elif failing:
        message = f"Test {failing} failed."
    elif err:
        message = f"{err} raised."
    else:
        message = (raw.strip().splitlines() or ["(no output)"])[-1][:200]

    return {
        "tests_passed": passed,
        "failing_test": failing,
        "file": file_,
        "line": line,
        "error_type": err,
        "message": message,
    }
