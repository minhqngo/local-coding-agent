import unittest

from coding_agent.llm import MockLLMClient
from coding_agent.mock_brain import MockBrain
from coding_agent.roles import (
    Coder,
    CoderAction,
    Dispatcher,
    Extractor,
    Planner,
    RepoQA,
)


def role_client(role, brain):
    return MockLLMClient(role=role, responder=brain)


class RolesTest(unittest.TestCase):
    def test_planner_delegates_then_finishes(self):
        brain = MockBrain(coder_actions=[{"action": "write_file", "path": "x.py", "content": "1"}])
        planner = Planner(role_client("planner", brain))

        d1 = planner.decide("fix it", observations=[], tests_passed=False)
        self.assertEqual(d1.next, "coder")

        d2 = planner.decide("fix it", observations=[], tests_passed=True)
        self.assertEqual(d2.next, "finish")
        self.assertTrue(d2.summary)

    def test_planner_runs_tests_after_actions(self):
        brain = MockBrain(coder_actions=[])  # no edits scripted
        planner = Planner(role_client("planner", brain))
        d = planner.decide("fix it", observations=[], tests_passed=False)
        self.assertEqual(d.next, "tool")
        self.assertEqual(d.tool["name"], "run_shell")

    def test_coder_pops_scripted_action(self):
        brain = MockBrain(coder_actions=[{"action": "write_file", "path": "x.py", "content": "y=1"}])
        coder = Coder(role_client("coder", brain))
        action = coder.act("do it")
        self.assertIsInstance(action, CoderAction)
        self.assertEqual(action.action, "write_file")
        self.assertEqual(action.path, "x.py")

    def test_dispatcher_maps_coder_action(self):
        call = Dispatcher.from_coder_action(
            CoderAction(rationale="", action="run_shell", command="pytest")
        )
        self.assertEqual(call.name, "run_shell")
        self.assertEqual(call.arguments["command"], "pytest")

    def test_extractor_detects_failure(self):
        brain = MockBrain()
        extractor = Extractor(role_client("extractor", brain))
        raw = 'test_calc.py:5: in test_zero\nZeroDivisionError: division by zero\n1 failed, 1 passed'
        obj = extractor.compress(raw)
        self.assertFalse(obj["tests_passed"])
        self.assertEqual(obj["error_type"], "ZeroDivisionError")

    def test_extractor_detects_pass(self):
        brain = MockBrain()
        extractor = Extractor(role_client("extractor", brain))
        obj = extractor.compress("2 passed in 0.01s")
        self.assertTrue(obj["tests_passed"])

    def test_rag_answers(self):
        brain = MockBrain(rag_answer="auth lives in auth.py")
        rag = RepoQA(role_client("rag", brain))
        self.assertIn("auth", rag.answer("where is auth?", ["chunk"]))


if __name__ == "__main__":
    unittest.main()
