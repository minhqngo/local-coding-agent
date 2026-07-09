"""Phase-2 coder-only baseline — the experimental control.

The plan's central research question is *"does a team of cheap specialists around
a small coder beat the small coder alone?"* Answering it requires a runnable
control: the coder wired by itself with read/write/patch/shell tools, no planner,
no extractor, no dispatcher. This module is that control.

It shares the sandbox, tools, passing-test gate, malformed-response recovery, and
workspace snapshot with the full loop (:class:`~coding_agent.loop.AgentLoop`) so
the only variable between the two is the presence of the glue models. Run both
over the same eval set and compare pass rates (``eval/run_eval.py --baseline``).
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .llm import ChatError
from .loop import RunResult, TranscriptEntry
from .registry import ModelRegistry
from .roles import CoderAction, Dispatcher
from .tools import ToolContext, ToolError, execute_tool

_RAW_CONTEXT_CHARS = 6000
_RAW_CONTEXT_KEEP = 4
_MUTATING = {"write_file", "patch_file"}


class CoderOnlyLoop:
    """The coder, alone, owning its own read→edit→test→finish loop."""

    def __init__(
        self,
        config: Config,
        registry: ModelRegistry,
        tool_ctx: ToolContext,
        logger=None,
        workspace: str = "",
    ):
        self.config = config
        self.registry = registry
        self.ctx = tool_ctx
        self.log = logger or (lambda _m: None)
        self.workspace = workspace

    def _run_command(self, command: str) -> tuple[str, bool]:
        result = self.ctx.sandbox.run(command)
        status = "OK" if result.ok else f"EXIT {result.exit_code}"
        raw = f"$ {command}\n[{status}]\n{result.output}"
        is_test = command.strip() == self.config.loop.test_command.strip()
        return raw, (result.ok if is_test else False)

    def run(self, task: str) -> RunResult:
        transcript: list[TranscriptEntry] = []
        raw_context: list[str] = []
        tests_passed = False
        coder = self.registry.coder
        test_command = self.config.loop.test_command
        max_turns = self.config.loop.max_turns
        max_attempts = max(max_turns * 3, max_turns + 4)
        turn = 0
        attempts = 0

        def remember(text: str) -> None:
            if len(text) > _RAW_CONTEXT_CHARS:
                text = text[:_RAW_CONTEXT_CHARS] + "\n...[truncated]"
            raw_context.append(text)
            del raw_context[:-_RAW_CONTEXT_KEEP]

        while turn < max_turns and attempts < max_attempts:
            attempts += 1
            try:
                action: CoderAction = coder.act_alone(
                    task, "\n\n".join(raw_context), test_command, workspace=self.workspace
                )
            except ChatError as exc:
                remember(
                    "runtime notice: your last response was not a valid code action; "
                    f"reply with one JSON action. (parse error: {exc})"
                )
                continue

            turn += 1
            detail = f"{action.action} {action.path or ''}".strip()
            transcript.append(TranscriptEntry(turn, "coder", detail))
            self.log(f"[turn {turn}] coder: {detail}")

            if action.action == "finish":
                if self.config.loop.require_passing_tests and not tests_passed:
                    raw, passed = self._run_command(test_command)
                    remember(raw)
                    tests_passed = passed
                    if not tests_passed:
                        remember("cannot finish: tests have not passed yet; keep working")
                        continue
                return RunResult(
                    success=True,
                    turns=turn,
                    summary=action.rationale or "Baseline coder completed the task.",
                    tests_passed=tests_passed,
                    transcript=transcript,
                )

            if action.action == "run_shell":
                raw, passed = self._run_command(action.command)
                if action.command.strip() == test_command.strip():
                    tests_passed = passed
                remember(raw)
                continue

            # read_file / write_file / patch_file → deterministic mapping.
            try:
                call = Dispatcher.from_coder_action(action)
                raw = execute_tool(self.ctx, call.name, call.arguments)
                if call.name in _MUTATING:
                    tests_passed = False  # files changed; must re-verify
            except (ValueError, ToolError) as exc:
                raw = f"ACTION ERROR ({action.action}): {exc}"
            remember(raw)

        if attempts >= max_attempts and turn < max_turns:
            summary = f"Baseline stopped: too many malformed responses ({max_attempts} attempts)."
        else:
            summary = f"Baseline turn limit ({max_turns}) reached without passing tests."
        return RunResult(
            success=False,
            turns=turn,
            summary=summary,
            tests_passed=tests_passed,
            transcript=transcript,
        )


def build_baseline_loop(config: Config, workdir: Path, responder=None, logger=None) -> CoderOnlyLoop:
    from .sandbox import Sandbox
    from .workspace import workspace_snapshot

    registry = ModelRegistry(config, responder=responder)
    sandbox = Sandbox(config.sandbox, workdir)
    ctx = ToolContext(workdir=Path(workdir).resolve(), sandbox=sandbox, search_fn=None)
    return CoderOnlyLoop(config, registry, ctx, logger=logger, workspace=workspace_snapshot(workdir))
