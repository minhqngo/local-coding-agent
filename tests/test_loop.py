import tempfile
import unittest
from pathlib import Path

from coding_agent.config import load_config
from coding_agent.loop import AgentLoop, ResumeState, TranscriptEntry, build_loop
from coding_agent.mock_brain import MockBrain
from coding_agent.registry import ModelRegistry
from coding_agent.roles import PlannerDecision, ToolCall
from coding_agent.sandbox import Sandbox
from coding_agent.session import SessionStore
from coding_agent.tools import ToolContext

BUGGY = "def safe_divide(a, b):\n    return a / b\n"
# A pytest-free check so this suite needs no extra dependencies.
CHECK_CMD = (
    'python -c "from calc import safe_divide; '
    "assert safe_divide(1, 0) == 0.0; assert safe_divide(10, 2) == 5; "
    'print(\'all checks passed\')"'
)
FIX = (
    "def safe_divide(a, b):\n"
    "    if b == 0:\n        return 0.0\n    return a / b\n"
)


class LoopTest(unittest.TestCase):
    def _workdir(self):
        d = Path(tempfile.mkdtemp(prefix="loop_test_"))
        (d / "calc.py").write_text(BUGGY)
        return d

    def test_full_loop_fixes_bug(self):
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.test_command = CHECK_CMD
        brain = MockBrain(
            coder_actions=[{"action": "write_file", "path": "calc.py", "content": FIX}],
            test_command=config.loop.test_command,
        )
        loop = build_loop(config, workdir, responder=brain)
        result = loop.run("Make safe_divide return 0.0 when dividing by zero.")

        self.assertTrue(result.success, msg=result.summary)
        self.assertTrue(result.tests_passed)
        # The fix was actually written to disk.
        self.assertIn("if b == 0", (workdir / "calc.py").read_text())
        # The loop exercised every role at least conceptually.
        actors = {e.actor for e in result.transcript}
        self.assertIn("planner", actors)
        self.assertIn("coder", actors)

    def test_dispatch_all_routes_through_dispatcher(self):
        # D2: with dispatch_all, even the planner's test-run goes through the
        # LFM2 dispatcher and still completes end-to-end.
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.test_command = CHECK_CMD
        config.loop.dispatch_all = True
        brain = MockBrain(
            coder_actions=[{"action": "write_file", "path": "calc.py", "content": FIX}],
            test_command=CHECK_CMD,
        )
        loop = build_loop(config, workdir, responder=brain)
        result = loop.run("fix")
        self.assertTrue(result.success, msg=result.summary)

    def test_turn_limit_without_fix_fails(self):
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.max_turns = 3
        config.loop.test_command = CHECK_CMD
        # No scripted fix: tests never pass, so the run must fail (not hang).
        brain = MockBrain(coder_actions=[], test_command=config.loop.test_command)
        loop = build_loop(config, workdir, responder=brain)
        result = loop.run("won't be fixed")
        self.assertFalse(result.success)
        self.assertFalse(result.tests_passed)
        self.assertEqual(result.turns, 3)

    def _bare_loop(self, config):
        """An AgentLoop with a no-op responder for white-box gate tests."""
        workdir = self._workdir()
        reg = ModelRegistry(config, responder=MockBrain(test_command=config.loop.test_command))
        ctx = ToolContext(workdir=workdir, sandbox=Sandbox(config.sandbox, workdir))
        return AgentLoop(config, reg, ctx), workdir

    def test_exact_test_command_gate(self):
        # B1: only the EXACT configured command may satisfy the finish gate.
        config = load_config(mock=True)
        config.loop.test_command = "python -m pytest -q"
        loop, _ = self._bare_loop(config)
        self.assertTrue(loop._is_exact_test_command("python -m pytest -q"))
        self.assertFalse(loop._is_exact_test_command("python -m pytest -q tests/test_one.py"))
        self.assertFalse(loop._is_exact_test_command("python -m pytest -q -k foo"))

    def test_mutating_tool_resets_tests_passed(self):
        # B2: a write after a green run invalidates the pass.
        config = load_config(mock=True)
        loop, _ = self._bare_loop(config)
        loop.tests_passed = True
        loop._execute(ToolCall("write_file", {"path": "z.py", "content": "x = 1\n"}))
        self.assertFalse(loop.tests_passed)

    def test_patch_file_resets_tests_passed(self):
        # P3/B2: a patch is mutating and invalidates a prior pass.
        config = load_config(mock=True)
        loop, workdir = self._bare_loop(config)
        (workdir / "m.py").write_text("a = 1\n")
        loop.tests_passed = True
        loop._execute(ToolCall("patch_file", {"path": "m.py", "old_text": "a = 1", "new_text": "a = 2"}))
        self.assertFalse(loop.tests_passed)

    def test_confirm_not_prompted_on_invalid_call(self):
        # P7.1: --confirm must not prompt for a call that would fail validation.
        config = load_config(mock=True)
        prompts = []
        d = self._workdir()
        reg = ModelRegistry(config, responder=MockBrain())
        ctx = ToolContext(workdir=d, sandbox=Sandbox(config.sandbox, d))
        loop = AgentLoop(config, reg, ctx, confirm=lambda c: prompts.append(c.name) or True)
        loop._execute(ToolCall("write_file", {"path": "a.py"}))  # missing content
        self.assertEqual(prompts, [])
        loop._execute(ToolCall("write_file", {"path": "a.py", "content": "x=1\n"}))
        self.assertEqual(prompts, ["write_file"])

    def test_write_invalidates_stale_read_in_coder_context(self):
        # P7.3: a write drops the pre-write read of that path from raw_context.
        config = load_config(mock=True)
        loop, workdir = self._bare_loop(config)
        (workdir / "a.py").write_text("x = 1\n")
        loop._execute(ToolCall("read_file", {"path": "a.py"}))
        self.assertTrue(any(r.startswith("FILE a.py:") for r in loop.raw_context))
        loop._execute(ToolCall("write_file", {"path": "a.py", "content": "x = 2\n"}))
        self.assertFalse(any(r.startswith("FILE a.py:") for r in loop.raw_context))

    def test_resume_continues_run(self):
        # P6: a bounded run persists; resume with a bigger budget finishes.
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.test_command = CHECK_CMD
        config.loop.max_turns = 1  # too few to finish
        store = SessionStore(workdir)
        sid = store.new_id()
        actions = [{"action": "write_file", "path": "calc.py", "content": FIX}]
        loop = build_loop(config, workdir,
                          responder=MockBrain(coder_actions=actions, test_command=CHECK_CMD),
                          session_store=store, session_id=sid)
        first = loop.run("fix")
        self.assertFalse(first.success)

        data = store.load(sid)
        config.loop.max_turns = 6
        loop2 = build_loop(config, workdir,
                           responder=MockBrain(coder_actions=actions, test_command=CHECK_CMD),
                           session_store=store, session_id=sid)
        rs = ResumeState(
            observations=data["observations"], raw_context=data["raw_context"],
            tests_passed=data["tests_passed"],
            transcript=[TranscriptEntry(**e) for e in data["transcript"]],
            turns_done=data["turns_done"],
        )
        second = loop2.run("fix", resume=rs)
        self.assertTrue(second.success, msg=second.summary)

    def test_premature_finish_is_blocked(self):
        # A planner that says "finish" before any passing test must not succeed:
        # the gate runs the (failing) test command and the loop keeps going.
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.max_turns = 2
        config.loop.test_command = CHECK_CMD  # fails: bug not fixed
        reg = ModelRegistry(config, responder=MockBrain(test_command=CHECK_CMD))
        ctx = ToolContext(workdir=workdir, sandbox=Sandbox(config.sandbox, workdir))
        loop = AgentLoop(config, reg, ctx)

        # Force the planner to always demand finish.
        loop.registry.planner.decide = lambda *a, **k: PlannerDecision(
            thought="done?", next="finish", summary="claims done"
        )
        result = loop.run("pretend it's done")
        self.assertFalse(result.success)
        self.assertFalse(result.tests_passed)


if __name__ == "__main__":
    unittest.main()
