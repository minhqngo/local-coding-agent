"""The harness guardrails: facts, gate, repeat breaker, stall detector, memory."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.config import load_config
from coding_agent.harness import STALL_WARN_AT, CoderMemory, Harness, ToolResult, remember
from coding_agent.roles import ToolCall
from coding_agent.sandbox import Sandbox
from coding_agent.tools import ToolContext


def _harness(config=None, confirm=None):
    config = config or load_config(mock=True)
    d = Path(tempfile.mkdtemp(prefix="harness_"))
    ctx = ToolContext(workdir=d, sandbox=Sandbox(config.sandbox, d))
    return Harness(config, ctx, confirm=confirm), d


class HeadlineTest(unittest.TestCase):
    def test_test_run_headline_states_the_verdict(self):
        config = load_config(mock=True)
        config.loop.test_command = 'python -c "raise SystemExit(1)"'
        h, _ = _harness(config)
        res = h.run_tests()
        self.assertTrue(res.is_test_run)
        self.assertIn("exit code 1", res.headline())
        self.assertIn("test command FAILED", res.headline())
        self.assertFalse(h.tests_passed)

    def test_blocked_call_says_not_run(self):
        h, _ = _harness()
        res = h.execute(ToolCall("nope", {}))
        self.assertIn("NOT RUN", res.headline())
        res = h.execute(ToolCall("write_file", {"path": "a.py"}))
        self.assertEqual(res.blocked, "invalid call")
        self.assertIn("missing required arguments", res.output)


class GateTest(unittest.TestCase):
    def test_only_exact_command_sets_tests_passed(self):
        config = load_config(mock=True)
        config.loop.test_command = 'python -c "print(1)"'
        h, _ = _harness(config)
        h.execute(ToolCall("run_shell", {"command": 'python -c "print(2)"'}))
        self.assertFalse(h.tests_passed)
        h.execute(ToolCall("run_shell", {"command": config.loop.test_command}))
        self.assertTrue(h.tests_passed)

    def test_successful_edit_invalidates_pass_failed_edit_does_not(self):
        h, d = _harness()
        h.tests_passed = True
        h.execute(ToolCall("patch_file", {"path": "missing.py", "old_text": "a", "new_text": "b"}))
        self.assertTrue(h.tests_passed)  # nothing changed on disk
        h.execute(ToolCall("write_file", {"path": "a.py", "content": "x = 1\n"}))
        self.assertFalse(h.tests_passed)


class StallTest(unittest.TestCase):
    def test_same_failure_with_different_commands_warns_then_stops(self):
        # The real failure this guards against: the planner tried ~15 slightly
        # different "install pip" commands that all hit the same error.
        config = load_config(mock=True)
        config.loop.max_repeated_outcomes = 5
        h, _ = _harness(config)
        results = []
        for i in range(5):
            cmd = f'python -c "import sys; sys.exit(\'No module named pip\')" {i}'
            results.append(h.execute(ToolCall("run_shell", {"command": cmd})))
        self.assertEqual(results[STALL_WARN_AT - 2].notice, "")
        self.assertIn("change approach", results[STALL_WARN_AT - 1].notice)
        self.assertIn("stuck", h.stuck_reason)

    def test_timing_noise_does_not_hide_a_repeat(self):
        h, _ = _harness()
        fp = [h._fingerprint(ToolResult(ToolCall("run_shell", {"command": "x"}),
                                        f"1 failed in {t}s", ok=False, exit_code=1))
              for t in ("0.01", "0.37")]
        self.assertEqual(fp[0], fp[1])

    def test_progress_is_not_a_stall(self):
        h, _ = _harness()
        for i in range(6):
            h.execute(ToolCall("write_file", {"path": f"f{i}.py", "content": "x"}))
        self.assertEqual(h.stuck_reason, "")

    def test_zero_disables_abort(self):
        config = load_config(mock=True)
        config.loop.max_repeated_outcomes = 0
        h, _ = _harness(config)
        for i in range(7):
            h.execute(ToolCall("run_shell", {"command": f"python -c \"exit(3)\" {i}"}))
        self.assertEqual(h.stuck_reason, "")


class TestAfterEditTest(unittest.TestCase):
    def test_edit_reports_test_result_in_the_same_step(self):
        # A correct fix is verified without the model having to ask for tests.
        config = load_config(mock=True)
        config.loop.test_command = 'python -c "from m import f; assert f() == 2"'
        h, d = _harness(config)
        (d / "m.py").write_text("def f():\n    return 1\n")
        res = h.execute(ToolCall("patch_file", {"path": "m.py", "old_text": "return 1",
                                                "new_text": "return 2"}))
        self.assertIn("tests now PASS", res.headline())
        self.assertTrue(h.tests_passed)
        res = h.execute(ToolCall("patch_file", {"path": "m.py", "old_text": "return 2",
                                                "new_text": "return 3"}))
        self.assertIn("tests still FAIL", res.headline())
        self.assertIn("AssertionError", res.output)
        self.assertFalse(h.tests_passed)

    def test_can_be_turned_off(self):
        config = load_config(mock=True)
        config.loop.test_after_edit = False
        h, _ = _harness(config)
        res = h.execute(ToolCall("write_file", {"path": "a.py", "content": "x = 1\n"}))
        self.assertIsNone(res.auto_test_ok)
        self.assertEqual(res.headline(), "write_file `a.py` → ok")


class DeletionTest(unittest.TestCase):
    def test_default_config_refuses_file_deletion(self):
        h, d = _harness()
        (d / "calc.py").write_text("x = 1\n")
        for cmd in ("del calc.py", "rm calc.py", "erase calc.py"):
            res = h.execute(ToolCall("run_shell", {"command": cmd}))
            self.assertFalse(res.ok)
            self.assertIn("REFUSED", res.output)
        self.assertTrue((d / "calc.py").exists())


class ConfirmTest(unittest.TestCase):
    def test_declined_call_is_not_executed(self):
        h, d = _harness(confirm=lambda call: False)
        res = h.execute(ToolCall("write_file", {"path": "a.py", "content": "x"}))
        self.assertEqual(res.blocked, "declined by user")
        self.assertFalse((d / "a.py").exists())


class CoderMemoryTest(unittest.TestCase):
    def test_budget_from_ctx(self):
        self.assertEqual(CoderMemory.budget_for(None, 2048), 18000)
        self.assertEqual(CoderMemory.budget_for(8192, 2048), (8192 - 2048 - 2000) * 3)
        self.assertEqual(CoderMemory.budget_for(131072, 2048), 24000)  # capped

    def test_oldest_items_evicted_to_fit_budget(self):
        mem = CoderMemory(budget_chars=2500)
        for i in range(5):
            mem.add(f"item{i} " + "x" * 1000)
        text = mem.render()
        self.assertIn("item4", text)
        self.assertNotIn("item0", text)

    def test_edit_forgets_stale_read(self):
        h, d = _harness()
        (d / "a.py").write_text("old = 1\n")
        mem = CoderMemory(10000)
        remember(mem, h.execute(ToolCall("read_file", {"path": "a.py"})))
        self.assertIn("old = 1", mem.render())
        remember(mem, h.execute(ToolCall("write_file", {"path": "a.py", "content": "new = 2\n"})))
        self.assertNotIn("old = 1", mem.render())
        self.assertIn("write_file `a.py` → ok", mem.render())


if __name__ == "__main__":
    unittest.main()
