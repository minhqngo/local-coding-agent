"""The agent loop.

    planner → (coder → dispatcher) → sandbox → extractor → planner → …

The planner owns strategy and picks the next move. Code changes are delegated to
the coder, whose loose action is normalised into an exact tool call (directly
when it is already clean, via the LFM2 dispatcher when it is free-form). The
sandbox executes; the extractor compresses raw output into small JSON before it
re-enters the planner's context — the single biggest win for keeping a 2B
planner coherent.

Hard guardrails: a turn limit, a separate malformed-response budget, an
anti-stall repeated-call breaker, and a mandatory passing-test gate before any
run may declare success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config
from .llm import ChatError
from .registry import ModelRegistry
from .roles import CoderAction, Dispatcher, PlannerDecision, ToolCall
from .tools import TOOLS, ToolContext, ToolError, execute_tool

Logger = Callable[[str], None]

# How much raw tool output the coder may see, and how many recent outputs.
_RAW_CONTEXT_CHARS = 6000
_RAW_CONTEXT_KEEP = 3
# Tool calls that change files invalidate a prior passing-test result.
_MUTATING_TOOLS = {"write_file", "patch_file"}
# Tool calls that require human confirmation under --confirm.
_CONFIRM_TOOLS = {"write_file", "patch_file", "run_shell"}


@dataclass
class TranscriptEntry:
    turn: int
    actor: str
    detail: str


@dataclass
class RunResult:
    success: bool
    turns: int
    summary: str
    tests_passed: bool
    transcript: list[TranscriptEntry] = field(default_factory=list)


@dataclass
class ResumeState:
    """Everything needed to continue a run after a crash or exit (P6)."""

    observations: list[str]
    raw_context: list[str]
    tests_passed: bool
    transcript: list[TranscriptEntry]
    turns_done: int = 0


class AgentLoop:
    def __init__(
        self,
        config: Config,
        registry: ModelRegistry,
        tool_ctx: ToolContext,
        logger: Logger | None = None,
        confirm: Callable[[ToolCall], bool] | None = None,
        workspace: str = "",
        session_store=None,
        session_id: str | None = None,
    ):
        self.config = config
        self.registry = registry
        self.ctx = tool_ctx
        self.log = logger or (lambda _msg: None)
        # Optional human-in-the-loop gate; called before mutating/shell calls.
        self.confirm = confirm
        # One-time repo snapshot fed into the planner's first prompt (P5).
        self.workspace = workspace
        # Optional crash-safe persistence (P6).
        self.session_store = session_store
        self.session_id = session_id
        self._reset()

    def _reset(self) -> None:
        # Compressed observations feed the PLANNER's small context window.
        self.observations: list[str] = []
        # Raw (truncated) tool output feeds the CODER, which must see real file
        # contents to rewrite them — the plan's division of labour.
        self.raw_context: list[str] = []
        self.tests_passed = False
        self.transcript: list[TranscriptEntry] = []
        # Keys of the last executed tool calls, for the repeated-call breaker.
        self._recent_calls: list[str] = []

    # -- helpers ------------------------------------------------------------- #
    def _record(self, turn: int, actor: str, detail: str) -> None:
        self.transcript.append(TranscriptEntry(turn, actor, detail))
        self.log(f"[turn {turn}] {actor}: {detail}")

    def _note(self, message: str) -> None:
        """Append an in-band corrective notice visible to planner and coder."""

        self.observations.append(message)
        self._remember_raw(message)

    def _observe(self, label: str, raw_output: str) -> dict:
        """Compress raw output via the extractor and store a compact observation."""

        compressed = self.registry.extractor.compress(raw_output)
        # sort_keys keeps the rendered prompt stable across turns (cache-friendly).
        self.observations.append(f"{label}: {json.dumps(compressed, sort_keys=True)}")
        return compressed

    def _remember_raw(self, raw_output: str) -> None:
        """Keep recent raw output so the coder can see actual file/command text."""

        text = raw_output
        if len(text) > _RAW_CONTEXT_CHARS:
            text = text[:_RAW_CONTEXT_CHARS] + "\n...[truncated]"
        self.raw_context.append(text)
        del self.raw_context[:-_RAW_CONTEXT_KEEP]

    def _invalidate_raw_for_path(self, path: str) -> None:
        """After a write/patch, drop stale prior reads of that file (P7.3)."""

        if not path:
            return
        needle = f"FILE {path}:"
        self.raw_context = [r for r in self.raw_context if not r.startswith(needle)]

    def _coder_context(self) -> str:
        return "\n\n".join(self.raw_context) if self.raw_context else ""

    def _is_exact_test_command(self, command: str) -> bool:
        # Only the exact configured command may satisfy the gate — a partial run
        # like `pytest tests/test_one.py` or `-k foo` must NOT unlock finish.
        return command.strip() == self.config.loop.test_command.strip()

    @staticmethod
    def _call_key(call: ToolCall) -> str:
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"

    def _is_repeat(self, call: ToolCall) -> bool:
        # Re-running the exact test command is always legitimate (after edits).
        if call.name == "run_shell" and self._is_exact_test_command(
            call.arguments.get("command", "")
        ):
            return False
        key = self._call_key(call)
        return self._recent_calls[-2:] == [key, key]

    def _valid_call(self, call: ToolCall) -> bool:
        """Cheap arg validation, so --confirm never prompts for a junk call (P7.1)."""

        spec = TOOLS.get(call.name)
        if spec is None:
            return False
        return all(
            arg in call.arguments
            for arg, required in spec["args"].items()
            if required
        )

    def _dispatch(self, action: CoderAction) -> ToolCall | None:
        """Turn a coder action into a strict tool call.

        Honours ``loop.dispatch_all``: when set, every call goes through the
        LFM2 dispatcher model (plan Phase-3 literal); otherwise a clean action
        is mapped deterministically and only free-form actions hit the model.
        Returns None if the action cannot be turned into a call (malformed).
        """

        if not self.config.loop.dispatch_all:
            try:
                return Dispatcher.from_coder_action(action)
            except ValueError:
                pass  # fall through to the model for unmapped actions
        intent = json.dumps(
            {
                "action": action.action,
                "path": action.path,
                "content": action.content,
                "old_text": action.old_text,
                "new_text": action.new_text,
                "command": action.command,
            }
        )
        try:
            return self.registry.dispatcher.format_call(intent)
        except ChatError:
            # Dispatcher garbled the call; try the deterministic mapping as a
            # last resort before giving up.
            try:
                return Dispatcher.from_coder_action(action)
            except ValueError:
                return None

    def _execute(self, call: ToolCall) -> tuple[str, dict]:
        """Run a tool call. Returns (raw_text, extracted).

        run_shell is special-cased so the loop reads the real exit code (and
        thus trusts the passing-test gate rather than the extractor alone).
        Raw output is remembered for the coder; a compressed copy feeds the
        planner. A mutating tool invalidates any prior passing-test result.
        """

        # Anti-stall: refuse a third identical call in a row (P4).
        if self._is_repeat(call):
            raw = (
                f"REFUSED: repeated identical tool call {call.name}; "
                "choose a different tool or finish."
            )
            self._note(raw)
            return raw, {"message": raw, "tests_passed": False}

        # Validate before prompting so --confirm never asks about a junk call (P7.1).
        if (
            self.confirm is not None
            and call.name in _CONFIRM_TOOLS
            and self._valid_call(call)
        ):
            if not self.confirm(call):
                raw = f"SKIPPED by user: {call.name}({', '.join(call.arguments)})"
                self._remember_raw(raw)
                return raw, self._observe(call.name, raw)

        self._recent_calls.append(self._call_key(call))

        if call.name == "run_shell":
            command = call.arguments.get("command", "")
            result = self.ctx.sandbox.run(command)
            status = "OK" if result.ok else f"EXIT {result.exit_code}"
            raw = f"$ {command}\n[{status}]\n{result.output}"
            self._remember_raw(raw)
            extracted = self._observe("run_shell", raw)
            if self._is_exact_test_command(command):
                # Exit code is decisive; the extractor is only a sanity signal.
                self.tests_passed = result.ok
            return raw, extracted

        try:
            raw = execute_tool(self.ctx, call.name, call.arguments)
        except ToolError as exc:
            raw = f"TOOL ERROR ({call.name}): {exc}"
        if call.name in _MUTATING_TOOLS:
            # Files changed since the last green run — must re-verify to finish.
            self.tests_passed = False
            self._invalidate_raw_for_path(call.arguments.get("path", ""))
        self._remember_raw(raw)
        extracted = self._observe(call.name, raw)
        return raw, extracted

    def _coder_step(self, turn: int, instruction: str) -> None:
        try:
            action: CoderAction = self.registry.coder.act(instruction, self._coder_context())
        except ChatError as exc:
            self._record(turn, "coder", f"malformed response: {exc}")
            self._note(
                "runtime notice: the coder returned an unparseable response; "
                "reply with exactly one code-action JSON object."
            )
            return
        self._record(turn, "coder", f"{action.action} {action.path or ''}".strip())

        call = self._dispatch(action)
        if call is None or not call.name:
            self._note(
                f"runtime notice: could not turn coder action {action.action!r} "
                "into a tool call; use action read_file/write_file/patch_file/run_shell."
            )
            return
        self._record(turn, "dispatcher", f"{call.name}({', '.join(call.arguments)})")
        self._execute(call)

    def _rag_step(self, turn: int, question: str) -> None:
        chunks: list[str] = []
        if self.ctx.search_fn is not None:
            # search_fn returns a text blob of chunks; feed it straight to RAG.
            chunks = [self.ctx.search_fn(question)]
        answer = self.registry.rag.answer(question, chunks)
        self.observations.append(f"rag_answer: {answer[:500]}")
        self._record(turn, "rag", answer[:120])

    def _run_test_gate(self, turn: int) -> None:
        call = ToolCall("run_shell", {"command": self.config.loop.test_command})
        self._record(turn, "gate", "running test command before finishing")
        self._execute(call)

    def _persist(self, task: str, turns_done: int) -> None:
        if self.session_store is None or self.session_id is None:
            return
        self.session_store.save(
            self.session_id,
            {
                "task": task,
                "turns_done": turns_done,
                "observations": self.observations,
                "raw_context": self.raw_context,
                "tests_passed": self.tests_passed,
                "transcript": [vars(e) for e in self.transcript],
            },
        )

    # -- main loop ----------------------------------------------------------- #
    def run(self, task: str, resume: ResumeState | None = None) -> RunResult:
        if resume is not None:
            self.observations = list(resume.observations)
            self.raw_context = list(resume.raw_context)
            self.tests_passed = resume.tests_passed
            self.transcript = list(resume.transcript)
            self._recent_calls = []
            turns_done = resume.turns_done
        else:
            self._reset()
            turns_done = 0

        max_turns = self.config.loop.max_turns
        # A separate budget so malformed responses can't starve real steps.
        max_attempts = max(max_turns * 3, max_turns + 4)
        turn = turns_done
        attempts = 0

        while turn < max_turns and attempts < max_attempts:
            attempts += 1
            try:
                decision: PlannerDecision = self.registry.planner.decide(
                    task, self.observations, self.tests_passed, workspace=self.workspace
                )
            except ChatError as exc:
                self._note(
                    "runtime notice: your last response was not valid JSON; "
                    "reply with exactly one next-move JSON object. "
                    f"(parse error: {exc})"
                )
                continue

            turn += 1
            self._record(turn, "planner", f"{decision.next} — {decision.thought}")

            if decision.next == "coder":
                self._coder_step(turn, decision.instruction or task)

            elif decision.next == "tool":
                call = ToolCall.from_obj(decision.tool or {})
                if not call.name:
                    self._note("error: planner emitted an empty tool call")
                else:
                    if self.config.loop.dispatch_all:
                        intent = json.dumps({"name": call.name, "arguments": call.arguments})
                        self._record(turn, "planner-tool", f"→ dispatcher: {call.name}")
                        try:
                            call = self.registry.dispatcher.format_call(intent)
                        except ChatError:
                            pass  # keep the planner's own well-formed call
                    self._record(turn, "planner-tool", f"{call.name}({', '.join(call.arguments)})")
                    self._execute(call)

            elif decision.next == "rag":
                self._rag_step(turn, decision.question or task)

            elif decision.next == "finish":
                if self.config.loop.require_passing_tests and not self.tests_passed:
                    # Enforce the gate: run tests once; only finish if they pass.
                    self._run_test_gate(turn)
                    if not self.tests_passed:
                        self._note("cannot finish: tests have not passed yet; keep working")
                        self._persist(task, turn)
                        continue
                self._persist(task, turn)
                return RunResult(
                    success=True,
                    turns=turn,
                    summary=decision.summary or "Task completed.",
                    tests_passed=self.tests_passed,
                    transcript=self.transcript,
                )
            else:
                self._note(f"error: planner returned unknown next={decision.next!r}")

            self._persist(task, turn)

        if attempts >= max_attempts and turn < max_turns:
            summary = f"Stopped: too many malformed model responses ({max_attempts} attempts)."
        else:
            summary = f"Turn limit ({max_turns}) reached without passing tests."
        return RunResult(
            success=False,
            turns=turn,
            summary=summary,
            tests_passed=self.tests_passed,
            transcript=self.transcript,
        )


def build_loop(
    config: Config,
    workdir: Path,
    responder=None,
    logger: Logger | None = None,
    enable_retrieval: bool = False,
    confirm: Callable[[ToolCall], bool] | None = None,
    session_store=None,
    session_id: str | None = None,
) -> AgentLoop:
    """Convenience constructor wiring registry + sandbox + tools + retrieval."""

    from .sandbox import Sandbox
    from .workspace import workspace_snapshot

    registry = ModelRegistry(config, responder=responder)
    sandbox = Sandbox(config.sandbox, workdir)

    search_fn = None
    if enable_retrieval:
        from .retrieval import Retriever

        retriever = Retriever(config.retrieval, workdir)
        retriever.build()

        def search_fn(query: str) -> str:  # noqa: E306
            hits = retriever.search(query)
            return "\n\n---\n\n".join(h.render() for h in hits) or "(no matches)"

    ctx = ToolContext(workdir=Path(workdir).resolve(), sandbox=sandbox, search_fn=search_fn)
    return AgentLoop(
        config,
        registry,
        ctx,
        logger=logger,
        confirm=confirm,
        workspace=workspace_snapshot(workdir),
        session_store=session_store,
        session_id=session_id,
    )
