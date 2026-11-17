"""Malformed-output recovery and the repeated-call breaker."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.config import load_config
from coding_agent.loop import AgentLoop
from coding_agent.registry import ModelRegistry
from coding_agent.roles import Extractor, ToolCall
from coding_agent.sandbox import Sandbox
from coding_agent.tools import ToolContext


def _loop(config, responder):
    d = Path(tempfile.mkdtemp(prefix="recovery_"))
    reg = ModelRegistry(config, responder=responder)
    ctx = ToolContext(workdir=d, sandbox=Sandbox(config.sandbox, d))
    return AgentLoop(config, reg, ctx)


class RecoveryTest(unittest.TestCase):
    def test_planner_garbage_then_valid_recovers(self):
        # 3 malformed planner responses, then a valid finish — must succeed
        # without consuming the (max_turns) step budget on the garbage.
        class Flaky:
            def __init__(self):
                self.n = 0

            def __call__(self, role, messages):
                if role == "planner":
                    self.n += 1
                    if self.n <= 3:
                        return "totally not json <x>"
                    return '{"next":"finish","thought":"done","summary":"ok"}'
                if role == "extractor":
                    return '{"tests_passed": true, "message": "ok"}'
                return "{}"

        config = load_config(mock=True)
        config.loop.require_passing_tests = False
        config.loop.max_turns = 2
        result = _loop(config, Flaky()).run("x")
        self.assertTrue(result.success)

    def test_garbage_only_exhausts_attempts_not_turns(self):
        # never-valid planner → distinct "malformed" failure, not turn-limit.
        class Garbage:
            def __call__(self, role, messages):
                return "not json" if role == "planner" else "{}"

        config = load_config(mock=True)
        config.loop.max_turns = 3
        result = _loop(config, Garbage()).run("x")
        self.assertFalse(result.success)
        self.assertIn("malformed", result.summary)

    def test_extractor_garbage_does_not_abort(self):
        # a garbled extractor degrades to a raw-fallback dict, never raises.
        class BadExtractor:
            def chat(self, messages, **kw):
                return "<<<not json>>>"

        obj = Extractor(BadExtractor()).compress("pytest output: 1 failed")
        self.assertIsInstance(obj, dict)
        self.assertIn("message", obj)

    def test_repeated_call_breaker(self):
        # third identical non-test call refused; test command exempt.
        config = load_config(mock=True)
        loop = _loop(config, lambda r, m: "{}")
        call = ToolCall("run_shell", {"command": "echo hi"})
        loop._execute(call)
        loop._execute(call)
        res = loop._execute(call)
        self.assertIn("REFUSED: repeated", res.output)
        self.assertEqual(res.blocked, "repeated identical call")

        test_call = ToolCall("run_shell", {"command": config.loop.test_command})
        loop._execute(test_call)
        loop._execute(test_call)
        res2 = loop._execute(test_call)
        self.assertNotIn("REFUSED", res2.output)


if __name__ == "__main__":
    unittest.main()
