"""The harness: everything between "the model asked for X" and "here is what happened".

A coding agent is two things:

* a **policy** that decides the next action (one model, or a team of models), and
* a **harness** that executes actions and enforces the rules.

The model proposes; the harness disposes. This module is the harness, and it is
shared by both agents in this repo (:mod:`coding_agent.single_model` and
:mod:`coding_agent.loop`), so the only difference between them is *how they
decide* — which is exactly what the eval compares.

Each guardrail below exists because small models fail in a specific way:

1. **Ground truth comes from the harness, not a model.** Exit codes, "was this
   the test command", "did it pass" — computed here, deterministically, and
   shown to the model as a one-line :meth:`ToolResult.headline`. (A 350M
   summariser once reported ``tests_passed: true`` for "No module named pip".)
2. **Test gate.** A run may only finish after the *exact* configured test
   command exited 0, and any later edit invalidates that pass.
3. **Repeated-call breaker.** The third identical call in a row is refused.
4. **Stall detector.** Models dodge (3) by varying a command slightly while
   hitting the same wall. So we also fingerprint *outcomes*: when the same
   result keeps coming back we warn the model, and eventually stop the run.
5. **Tests after every edit.** Small models make a correct fix and then never
   check it; the harness runs the test command after each successful edit and
   reports the result in the same step (``loop.test_after_edit``).
6. **Validation before confirmation.** With ``--confirm``, the human is only
   asked about calls that are well-formed.
7. **Everything is traced** (see :mod:`coding_agent.trace`).
"""

from __future__ import annotations

import hashlib
import posixpath
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import Config
from .roles import ToolCall
from .tools import (
    MUTATING_TOOLS, TOOLS, ToolContext, ToolError, as_text, execute_tool, missing_args,
)

Logger = Callable[[str], None]
ConfirmFn = Callable[[ToolCall], bool]

# Calls that need a human yes under --confirm.
CONFIRM_TOOLS = {"write_file", "patch_file", "run_shell"}
# Stall detection: look at this many recent results...
STALL_WINDOW = 8
# ...and warn the model once the same outcome has been seen this many times.
STALL_WARN_AT = 3

_NOISE = [
    (re.compile(r"\d+(\.\d+)?s\b"), "<t>"),          # "in 0.12s" changes every run
    (re.compile(r"0x[0-9a-fA-F]+"), "0x?"),           # object addresses
    (re.compile(r"\s+"), " "),
]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _tail(text: str, limit: int) -> str:
    """Keep the end: test runners print the verdict and failures last."""
    return text if len(text) <= limit else "…" + text[-limit:]


@dataclass
class ToolResult:
    """What happened when the harness handled one tool call."""

    call: ToolCall
    output: str                 # what the tool printed (or the error message)
    ok: bool                    # exit code 0 / no tool error
    exit_code: int | None = None
    is_test_run: bool = False
    blocked: str = ""           # non-empty = NOT executed, and why
    notice: str = ""            # extra guidance for the model (e.g. stall warning)
    auto_test_ok: bool | None = None  # set when the harness ran the tests after an edit

    @property
    def target(self) -> str:
        args = self.call.arguments
        return str(args.get("command") or args.get("path") or args.get("query") or "")

    def headline(self) -> str:
        """One line of facts the model can trust — computed, never generated."""

        what = f"{self.call.name} `{_clip(self.target, 120)}`" if self.target else self.call.name
        if self.blocked:
            return f"{what} → NOT RUN ({self.blocked})"
        if self.call.name == "run_shell":
            status = f"exit code {self.exit_code}"
            if self.is_test_run:
                status += " — test command " + ("PASSED" if self.ok else "FAILED")
        else:
            status = "ok" if self.ok else "error"
        if self.auto_test_ok is not None:
            status += " — tests now PASS" if self.auto_test_ok else " — tests still FAIL"
        return f"{what} → {status}"

    def for_model(self) -> str:
        """Headline + raw output (+ notice): what a model sees verbatim."""

        text = self.headline()
        if self.output.strip():
            text += "\n" + self.output.rstrip()
        if self.notice:
            text += f"\nNOTICE: {self.notice}"
        return text


class Harness:
    def __init__(
        self,
        config: Config,
        ctx: ToolContext,
        *,
        confirm: ConfirmFn | None = None,
        tracer: Any = None,
        log: Logger | None = None,
    ):
        self.config = config
        self.ctx = ctx
        self.confirm = confirm
        self.tracer = tracer
        self.log = log or (lambda _m: None)
        self.reset()

    def reset(self) -> None:
        self.tests_passed = False
        self._recent_calls: list[str] = []
        self._outcomes: list[str] = []
        self.stuck_reason = ""

    # -- facts ---------------------------------------------------------------- #
    @property
    def test_command(self) -> str:
        return self.config.loop.test_command

    def is_test_command(self, command: str) -> bool:
        # Only the exact configured command may satisfy the gate — a partial run
        # like `pytest tests/test_one.py` or `-k foo` must NOT unlock finish.
        return command.strip() == self.test_command.strip()

    def environment(self) -> str:
        return self.ctx.sandbox.describe()

    # -- guardrail helpers ---------------------------------------------------- #
    @staticmethod
    def _call_key(call: ToolCall) -> str:
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"

    def _is_repeat(self, call: ToolCall) -> bool:
        if call.name == "run_shell" and self.is_test_command(call.arguments.get("command", "")):
            return False  # re-running the tests after an edit is always legitimate
        key = self._call_key(call)
        return self._recent_calls[-2:] == [key, key]

    def _fingerprint(self, result: ToolResult) -> str:
        body = result.output
        for pattern, repl in _NOISE:
            body = pattern.sub(repl, body)
        raw = f"{result.call.name}|{result.blocked}|{result.exit_code}|{result.ok}|{body.strip()}"
        return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()

    def _check_stall(self, result: ToolResult) -> None:
        fp = self._fingerprint(result)
        self._outcomes.append(fp)
        del self._outcomes[:-STALL_WINDOW]
        seen = self._outcomes.count(fp)
        limit = self.config.loop.max_repeated_outcomes
        if limit and seen >= limit:
            self.stuck_reason = (
                f"the same result came back {seen} times in the last {STALL_WINDOW} steps "
                f"({result.headline()}) — the agent is stuck"
            )
        if seen >= STALL_WARN_AT:
            result.notice = (
                f"this exact result has now come back {seen} times in the last "
                f"{STALL_WINDOW} steps. Repeating the same approach will not help: read the "
                "error carefully and change approach (inspect the relevant file, edit the "
                "code, or run a different command)."
            )

    def _trace(self, result: ToolResult) -> None:
        if self.tracer is None:
            return
        args = {k: _clip(str(v), 2000) for k, v in result.call.arguments.items()}
        self.tracer.tool_call(
            name=result.call.name,
            arguments=args,
            summary=result.target,
            status=result.headline().split("→", 1)[-1].strip(),
            ok=result.ok,
            exit_code=result.exit_code,
            blocked=result.blocked,
            output=_clip(result.output, 4000),
            notice=result.notice,
        )

    # -- the one entry point -------------------------------------------------- #
    def execute(self, call: ToolCall) -> ToolResult:
        result = self._execute(call)
        self._check_stall(result)
        self._trace(result)
        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        if call.name not in TOOLS:
            return ToolResult(call, f"TOOL ERROR: unknown tool {call.name!r}; known tools: "
                              f"{sorted(TOOLS)}", ok=False, blocked="unknown tool")
        missing = missing_args(call.name, call.arguments)
        if missing:
            return ToolResult(call, f"TOOL ERROR ({call.name}): missing required arguments "
                              f"{missing}", ok=False, blocked="invalid call")
        # Models send lists, numbers and nulls where strings belong. Coerce
        # once, here, so nothing downstream has to defend against it.
        try:
            call = ToolCall(call.name, {k: as_text(k, v) for k, v in call.arguments.items()
                                        if k in TOOLS[call.name]["args"]})
        except ToolError as exc:
            return ToolResult(call, f"TOOL ERROR ({call.name}): {exc}", ok=False,
                              blocked="invalid call")

        if self._is_repeat(call):
            return ToolResult(
                call,
                f"REFUSED: repeated identical tool call {call.name}; "
                "choose a different action or finish.",
                ok=False, blocked="repeated identical call",
            )

        # Validated first, so --confirm never asks about a junk call.
        if self.confirm is not None and call.name in CONFIRM_TOOLS:
            if not self.confirm(call):
                return ToolResult(call, "SKIPPED by user", ok=False, blocked="declined by user")

        self._recent_calls.append(self._call_key(call))

        if call.name == "run_shell":
            command = str(call.arguments.get("command", ""))
            res = self.ctx.sandbox.run(command)
            is_test = self.is_test_command(command)
            if is_test:
                # The exit code is decisive. No model gets a vote on this.
                self.tests_passed = res.ok
            return ToolResult(call, res.output, ok=res.ok, exit_code=res.exit_code,
                              is_test_run=is_test)

        try:
            output = execute_tool(self.ctx, call.name, call.arguments)
            ok = True
        except ToolError as exc:
            output, ok = f"TOOL ERROR ({call.name}): {exc}", False
        except OSError as exc:  # read-only file, invalid name, path is a file...
            output, ok = f"TOOL ERROR ({call.name}): {exc.__class__.__name__}: {exc}", False
        result = ToolResult(call, output, ok=ok)
        if call.name in MUTATING_TOOLS and ok:
            self.tests_passed = False  # files changed since the last green run
            if self.config.loop.test_after_edit:
                self._test_after_edit(result)
        return result

    def _test_after_edit(self, result: ToolResult) -> None:
        """Run the tests right after a successful edit and report it in the same step.

        In traces, small planners often made the right fix and then never ran
        the tests — they kept "fixing" a solved problem. Deciding *when* to test
        is a judgement call they are bad at, and for a fast test suite it costs
        almost nothing to always do it (Aider does the same).
        """

        res = self.ctx.sandbox.run(self.test_command)
        self.tests_passed = res.ok
        result.auto_test_ok = res.ok
        verdict = "PASSED" if res.ok else f"FAILED (exit code {res.exit_code})"
        result.output += (f"\n\n[the harness ran the test command after this edit: "
                          f"{verdict}]\n$ {self.test_command}\n{_tail(res.output, 2500)}")

    def run_tests(self) -> ToolResult:
        """The finish gate: run the exact test command once."""

        return self.execute(ToolCall("run_shell", {"command": self.test_command}))


class CoderMemory:
    """The coder's working memory: recent raw tool output, newest last.

    The planner sees compressed observations; the coder must see *real* text
    (file contents, tracebacks) to edit correctly. Two rules keep this useful:

    * a **character budget** derived from the coder's served context window, so
      the prompt never overflows ``ctx`` (an overflow is a hard HTTP error);
    * **stale reads are dropped**: after an edit to ``calc.py``, an old copy of
      ``calc.py`` in memory would mislead the next patch.
    """

    PER_ITEM_CHARS = 6000

    def __init__(self, budget_chars: int):
        self.budget_chars = max(2000, budget_chars)
        self.items: list[dict[str, str]] = []   # {"path": ..., "text": ...}

    @staticmethod
    def budget_for(ctx_tokens: int | None, max_tokens: int, reserved_tokens: int = 2000) -> int:
        """~3 characters per token is a safe rule of thumb for code."""

        if not ctx_tokens:
            return 18000
        return max(4000, min(24000, (ctx_tokens - max_tokens - reserved_tokens) * 3))

    @staticmethod
    def _key(path: str) -> str:
        """One spelling per file: `pkg\\calc.py`, `./pkg/calc.py`, `pkg/calc.py`."""

        if not path:
            return ""
        return posixpath.normpath(path.replace("\\", "/")).lower()

    def add(self, text: str, path: str = "") -> None:
        if len(text) > self.PER_ITEM_CHARS:
            text = text[: self.PER_ITEM_CHARS] + "\n...[truncated]"
        self.items.append({"path": self._key(path), "text": text})
        while len(self.items) > 1 and sum(len(i["text"]) for i in self.items) > self.budget_chars:
            self.items.pop(0)

    def forget_file(self, path: str) -> None:
        key = self._key(path)
        if key:
            self.items = [i for i in self.items if i["path"] != key]

    def render(self) -> str:
        return "\n\n".join(i["text"] for i in self.items)

    def to_json(self) -> list[dict[str, str]]:
        return list(self.items)

    def load(self, items: list[Any]) -> None:
        # Older sessions stored plain strings.
        self.items = [i if isinstance(i, dict) else {"path": "", "text": str(i)} for i in items]


def remember(memory: CoderMemory, result: ToolResult) -> None:
    """Standard way to put a tool result into the coder's memory."""

    path = str(result.call.arguments.get("path", "")) if result.call.name in (
        "read_file", "write_file", "patch_file") else ""
    if result.call.name in MUTATING_TOOLS and result.ok:
        memory.forget_file(path)
        path = ""  # the edit confirmation isn't a copy of the file
    tag = path if result.call.name == "read_file" and result.ok else ""
    memory.add(result.for_model(), path=tag)
