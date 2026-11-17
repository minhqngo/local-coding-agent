"""The multi-model agent: a team of small specialists around a small coder.

    planner → (coder → dispatcher) → harness/sandbox → observe → planner → …

* The **planner** reads the task and *observations* and picks the next move.
* Code changes go to the **coder**, whose action is turned into an exact tool
  call (deterministically when clean, via the **dispatcher** model when messy).
* The **harness** (:mod:`coding_agent.harness`) executes it and enforces the
  rules — the same harness the single-model agent uses.
* Each result becomes an **observation** for the planner: a line of facts from
  the harness, then the output itself if short, or an **extractor** summary if
  long. That keeps a 2B planner's prompt small without letting a summariser
  rewrite the facts.

Two context streams, on purpose: the planner sees compact observations; the
coder sees raw text (:class:`~coding_agent.harness.CoderMemory`), because you
can't patch a file you've only seen summarised.

Compare with :mod:`coding_agent.single_model` — same harness, one model.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config
from .harness import CoderMemory, ConfirmFn, Harness, ToolResult, remember
from .llm import ChatError
from .registry import ModelRegistry
from .roles import CoderAction, Dispatcher, PlannerDecision, ToolCall
from .tools import ToolContext

Logger = Callable[[str], None]

# Output at most this long goes to the planner verbatim. Running a model to
# "compress" 3 lines costs a call and adds a chance to garble them.
VERBATIM_OBSERVATION_CHARS = 600
# Files up to this size are shown to the planner too. A planner that can't see
# the code it is planning for just asks to read it again (seen in traces).
PLANNER_FILE_CHARS = 2500


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
    """Everything needed to continue a run after a crash or exit."""

    observations: list[str]
    raw_context: list
    tests_passed: bool
    transcript: list[TranscriptEntry]
    turns_done: int = 0


def _indent(text: str) -> str:
    return textwrap.indent(text.rstrip(), "  ")


class AgentLoop:
    def __init__(
        self,
        config: Config,
        registry: ModelRegistry,
        tool_ctx: ToolContext,
        logger: Logger | None = None,
        confirm: ConfirmFn | None = None,
        workspace: str = "",
        session_store=None,
        session_id: str | None = None,
        tracer=None,
    ):
        self.config = config
        self.registry = registry
        self.ctx = tool_ctx
        self.log = logger or (lambda _msg: None)
        self.tracer = tracer
        self.harness = Harness(config, tool_ctx, confirm=confirm, tracer=tracer, log=self.log)
        # One-time repo snapshot fed into the planner's first prompt.
        self.workspace = workspace
        # Optional crash-safe persistence.
        self.session_store = session_store
        self.session_id = session_id
        coder_cfg = config.roles.get("coder")
        self._coder_budget = CoderMemory.budget_for(
            coder_cfg.ctx if coder_cfg else None,
            coder_cfg.max_tokens if coder_cfg else 2048,
        )
        planner_cfg = config.roles.get("planner")
        # Same idea for the planner: its whole prompt must fit its context.
        self._planner_budget = CoderMemory.budget_for(
            planner_cfg.ctx if planner_cfg else None,
            planner_cfg.max_tokens if planner_cfg else 2048,
            reserved_tokens=500,
        )
        self._reset()

    def _reset(self) -> None:
        # Compact observations feed the PLANNER's prompt.
        self.observations: list[str] = []
        # Raw tool output feeds the CODER (see CoderMemory).
        self.coder_memory = CoderMemory(self._coder_budget)
        self.transcript: list[TranscriptEntry] = []
        self._task = ""
        self.harness.reset()

    # The passing-test flag lives in the harness; exposed here for convenience.
    @property
    def tests_passed(self) -> bool:
        return self.harness.tests_passed

    @tests_passed.setter
    def tests_passed(self, value: bool) -> None:
        self.harness.tests_passed = value

    # -- helpers ------------------------------------------------------------- #
    def _record(self, turn: int, actor: str, detail: str) -> None:
        self.transcript.append(TranscriptEntry(turn, actor, detail))
        self.log(f"[turn {turn}] {actor}: {detail}")

    def _note(self, message: str) -> None:
        """An in-band notice that both the planner and the coder will see."""

        self.observations.append(message)
        self.coder_memory.add(message)
        if self.tracer is not None:
            self.tracer.note(message)

    def _observe(self, result: ToolResult) -> str:
        """Turn a tool result into a planner observation (facts first)."""

        obs = result.headline()
        body = result.output.strip()
        if result.call.name == "read_file" and result.ok:
            if len(body) <= PLANNER_FILE_CHARS:
                obs += "\n" + _indent(body)
            else:
                lines = body.count("\n")
                obs += (f" ({lines} lines; too long to show here — the coder has the full "
                        f"text)\n" + _indent(body[:PLANNER_FILE_CHARS // 2]) + "\n  ...")
        elif body and (result.blocked or len(body) <= VERBATIM_OBSERVATION_CHARS):
            obs += "\n" + _indent(body)
        elif body:
            summary = self.registry.extractor.compress(body)
            # sort_keys keeps the prompt stable across turns (cache-friendly).
            obs += "\n  summary: " + json.dumps(summary, sort_keys=True)
        if result.notice:
            obs += f"\n  NOTICE: {result.notice}"
        self.observations.append(obs)
        return obs

    def _execute(self, call: ToolCall) -> ToolResult:
        """Execute via the harness, then update both context streams."""

        result = self.harness.execute(call)
        remember(self.coder_memory, result)
        self._observe(result)
        return result

    def _dispatch(self, action: CoderAction) -> ToolCall | None:
        """Turn a coder action into a strict tool call.

        Honours ``loop.dispatch_all``: when set, every call goes through the
        dispatcher model; otherwise a clean action is mapped deterministically
        and only unmappable actions hit the model. Returns None if the action
        cannot be turned into a call (malformed).
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

    def _coder_step(self, turn: int, instruction: str) -> None:
        try:
            action: CoderAction = self.registry.coder.act(
                instruction, self.coder_memory.render(), task=self._task
            )
        except ChatError as exc:
            self._record(turn, "coder", f"malformed response: {exc}")
            self._note(
                "runtime notice: the coder returned an unparseable response; "
                "reply with exactly one code-action JSON object."
            )
            return
        self._record(turn, "coder", f"{action.action} {action.path or action.command}".strip())

        if action.action == "finish":
            self._note("runtime notice: the coder reports the change is done; "
                       "planner: run the test command to verify, then finish.")
            return
        call = self._dispatch(action)
        if call is None or not call.name:
            self._note(
                f"runtime notice: could not turn coder action {action.action!r} "
                "into a tool call; use action read_file/write_file/patch_file/run_shell."
            )
            return
        result = self._execute(call)
        self._record(turn, "harness", result.headline())

    def _rag_step(self, turn: int, question: str) -> None:
        chunks: list[str] = []
        if self.ctx.search_fn is not None:
            # search_fn returns a text blob of chunks; feed it straight to RAG.
            chunks = [self.ctx.search_fn(question)]
        try:
            answer = self.registry.rag.answer(question, chunks)
        except ChatError as exc:  # recoverable: say so and let the planner move on
            self._note(f"runtime notice: the repo-QA model gave no usable answer ({exc})")
            return
        self.observations.append(f"rag_answer: {answer[:500]}")
        self._record(turn, "rag", answer[:120])

    def _persist(self, task: str, turns_done: int) -> None:
        if self.session_store is None or self.session_id is None:
            return
        self.session_store.save(
            self.session_id,
            {
                "task": task,
                "turns_done": turns_done,
                "observations": self.observations,
                "raw_context": self.coder_memory.to_json(),
                "tests_passed": self.tests_passed,
                "transcript": [vars(e) for e in self.transcript],
            },
        )

    def _result(self, success: bool, turn: int, summary: str) -> RunResult:
        if self.tracer is not None:
            self.tracer.emit("run", event="end", detail=summary, success=success, turns=turn)
        return RunResult(success, turn, summary, self.tests_passed, self.transcript)

    # -- main loop ----------------------------------------------------------- #
    def run(self, task: str, resume: ResumeState | None = None) -> RunResult:
        if resume is not None:
            self._reset()
            self.observations = list(resume.observations)
            self.coder_memory.load(resume.raw_context)
            # Never trust a saved "tests passed": files may have changed since,
            # and trusting it would let a resumed run finish without any test.
            self.tests_passed = False
            self.transcript = list(resume.transcript)
            turns_done = resume.turns_done
        else:
            self._reset()
            turns_done = 0
        self._task = task
        if self.tracer is not None:
            self.tracer.emit("run", event="start", detail=task, agent="multi-model",
                             workspace=self.workspace)

        max_turns = self.config.loop.max_turns
        # A separate budget so malformed responses can't starve real steps.
        max_attempts = max(max_turns * 3, max_turns + 4)
        turn = turns_done
        attempts = 0

        while turn < max_turns and attempts < max_attempts:
            attempts += 1
            try:
                decision: PlannerDecision = self.registry.planner.decide(
                    task, self.observations, self.tests_passed, workspace=self.workspace,
                    turn=turn + 1, max_turns=max_turns, budget_chars=self._planner_budget,
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
                call = decision.tool_call()
                if not call.name:
                    self._note("runtime notice: the tool call was empty; give "
                               '{"name": ..., "arguments": {...}}')
                else:
                    if self.config.loop.dispatch_all:
                        intent = json.dumps({"name": call.name, "arguments": call.arguments})
                        try:
                            call = self.registry.dispatcher.format_call(intent)
                        except ChatError:
                            pass  # keep the planner's own well-formed call
                    result = self._execute(call)
                    self._record(turn, "harness", result.headline())

            elif decision.next == "rag":
                self._rag_step(turn, decision.question or task)

            elif decision.next == "finish":
                if self.config.loop.require_passing_tests and not self.tests_passed:
                    # Enforce the gate: run the tests once; only finish if they pass.
                    self._record(turn, "gate", "finish requested; running the test command")
                    result = self._execute(ToolCall("run_shell", {"command": self.harness.test_command}))
                    self._record(turn, "harness", result.headline())
                    if not self.tests_passed:
                        self._note("cannot finish: the test command has not passed yet; keep working")
                        self._persist(task, turn)
                        continue
                self._persist(task, turn)
                return self._result(True, turn, decision.summary or "Task completed.")
            else:
                self._note(f"runtime notice: unknown next={decision.next!r}; "
                           "use coder, tool, rag or finish")

            self._persist(task, turn)
            if self.harness.stuck_reason:
                return self._result(False, turn, f"Stopped early: {self.harness.stuck_reason}.")

        if attempts >= max_attempts and turn < max_turns:
            summary = f"Stopped: too many malformed model responses ({max_attempts} attempts)."
        else:
            summary = f"Turn limit ({max_turns}) reached without passing tests."
        return self._result(False, turn, summary)


def build_loop(
    config: Config,
    workdir: Path,
    responder=None,
    logger: Logger | None = None,
    enable_retrieval: bool = False,
    confirm: ConfirmFn | None = None,
    session_store=None,
    session_id: str | None = None,
    tracer=None,
) -> AgentLoop:
    """Convenience constructor wiring registry + sandbox + tools + retrieval."""

    from .sandbox import Sandbox
    from .workspace import workspace_snapshot

    registry = ModelRegistry(config, responder=responder, tracer=tracer)
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
    snapshot = workspace_snapshot(
        workdir,
        environment=sandbox.describe(),
        test_command=config.loop.test_command,
        skip_dirs=set(config.retrieval.exclude_dirs),
    )
    return AgentLoop(
        config,
        registry,
        ctx,
        logger=logger,
        confirm=confirm,
        workspace=snapshot,
        session_store=session_store,
        session_id=session_id,
        tracer=tracer,
    )
