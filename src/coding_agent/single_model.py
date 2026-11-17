"""The single-model agent: one coder model, one loop. Start reading here.

This is the whole idea of a coding agent in ~100 lines:

    while not done:
        action = model(task + workspace + recent tool output)   # decide
        result = harness.execute(action)                         # act
        memory.add(result)                                       # observe

Everything else in the repo is refinement of one of those three lines:
better deciding (:mod:`coding_agent.loop` splits it across specialist models),
safer acting (:mod:`coding_agent.harness`, :mod:`coding_agent.sandbox`), or
smarter observing (:class:`~coding_agent.harness.CoderMemory`, the extractor).

It is also the **experimental control**. The research question behind this repo
is *"does a team of cheap specialists around a small coder beat the small coder
alone?"* — so this agent shares the exact same harness, tools, test gate and
guardrails as the multi-model one. Only the deciding differs. Compare them with
``python eval/run_eval.py`` vs ``python eval/run_eval.py --single-model``.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .harness import CoderMemory, Harness, remember
from .llm import ChatError
from .loop import RunResult, TranscriptEntry
from .registry import ModelRegistry
from .roles import CoderAction, Dispatcher
from .tools import ToolContext


class SingleModelLoop:
    """The coder, alone, owning its own read → edit → test → finish loop."""

    def __init__(
        self,
        config: Config,
        registry: ModelRegistry,
        tool_ctx: ToolContext,
        logger=None,
        workspace: str = "",
        confirm=None,
        tracer=None,
    ):
        self.config = config
        self.registry = registry
        self.log = logger or (lambda _m: None)
        self.workspace = workspace
        self.tracer = tracer
        self.harness = Harness(config, tool_ctx, confirm=confirm, tracer=tracer, log=self.log)
        coder_cfg = config.roles.get("coder")
        # The single coder also carries the workspace snapshot in its prompt.
        self.budget = CoderMemory.budget_for(
            coder_cfg.ctx if coder_cfg else None,
            coder_cfg.max_tokens if coder_cfg else 2048,
            reserved_tokens=3000,
        )

    def run(self, task: str) -> RunResult:
        harness = self.harness
        harness.reset()
        memory = CoderMemory(self.budget)
        transcript: list[TranscriptEntry] = []
        max_turns = self.config.loop.max_turns
        max_attempts = max(max_turns * 3, max_turns + 4)  # separate malformed budget
        turn = attempts = 0
        if self.tracer is not None:
            self.tracer.emit("run", event="start", detail=task, agent="single-model",
                             workspace=self.workspace)

        def done(success: bool, summary: str) -> RunResult:
            if self.tracer is not None:
                self.tracer.emit("run", event="end", detail=summary, success=success, turns=turn)
            return RunResult(success, turn, summary, harness.tests_passed, transcript)

        while turn < max_turns and attempts < max_attempts:
            attempts += 1
            # 1. DECIDE
            try:
                action: CoderAction = self.registry.coder.act_alone(
                    task, memory.render(), harness.test_command, workspace=self.workspace
                )
            except ChatError as exc:
                memory.add("runtime notice: your last response was not a valid code action; "
                           f"reply with one JSON action. (parse error: {exc})")
                continue

            turn += 1
            detail = f"{action.action} {action.path or action.command}".strip()
            transcript.append(TranscriptEntry(turn, "coder", detail))
            self.log(f"[turn {turn}] coder: {detail}")

            if action.action == "finish":
                if self.config.loop.require_passing_tests and not harness.tests_passed:
                    result = harness.run_tests()          # the finish gate
                    remember(memory, result)
                    if not harness.tests_passed:
                        memory.add("cannot finish: the test command has not passed yet; keep working")
                        continue
                return done(True, action.rationale or "Single-model agent completed the task.")

            # 2. ACT — clean actions map to tool calls without another model.
            try:
                call = Dispatcher.from_coder_action(action)
            except ValueError:
                memory.add(f"runtime notice: unknown action {action.action!r}; use "
                           "read_file, write_file, patch_file, run_shell or finish.")
                continue
            result = harness.execute(call)
            self.log(f"           {result.headline()}")

            # 3. OBSERVE
            remember(memory, result)
            if harness.stuck_reason:
                return done(False, f"Stopped early: {harness.stuck_reason}.")

        if attempts >= max_attempts and turn < max_turns:
            return done(False, f"Stopped: too many malformed responses ({max_attempts} attempts).")
        return done(False, f"Turn limit ({max_turns}) reached without passing tests.")


def build_single_model_loop(
    config: Config, workdir: Path, responder=None, logger=None, confirm=None, tracer=None,
) -> SingleModelLoop:
    from .sandbox import Sandbox
    from .workspace import workspace_snapshot

    registry = ModelRegistry(config, responder=responder, tracer=tracer)
    sandbox = Sandbox(config.sandbox, workdir)
    ctx = ToolContext(workdir=Path(workdir).resolve(), sandbox=sandbox, search_fn=None)
    snapshot = workspace_snapshot(
        workdir,
        environment=sandbox.describe(),
        test_command=config.loop.test_command,
        skip_dirs=set(config.retrieval.exclude_dirs),
    )
    return SingleModelLoop(config, registry, ctx, logger=logger, workspace=snapshot,
                           confirm=confirm, tracer=tracer)
