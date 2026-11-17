import tempfile
import unittest
from pathlib import Path

from coding_agent.single_model import build_single_model_loop
from coding_agent.config import load_config
from coding_agent.mock_brain import MockBrain

BUGGY = "def safe_divide(a, b):\n    return a / b\n"
CHECK_CMD = (
    'python -c "from calc import safe_divide; '
    "assert safe_divide(1, 0) == 0.0; print('ok')\""
)
FIX = "def safe_divide(a, b):\n    if b == 0:\n        return 0.0\n    return a / b\n"


class SingleModelTest(unittest.TestCase):
    def _workdir(self):
        d = Path(tempfile.mkdtemp(prefix="single_test_"))
        (d / "calc.py").write_text(BUGGY)
        return d

    def test_single_model_fixes_and_verifies(self):
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.test_command = CHECK_CMD
        brain = MockBrain(
            coder_actions=[{"action": "write_file", "path": "calc.py", "content": FIX}],
            test_command=CHECK_CMD,
            single_model=True,
        )
        loop = build_single_model_loop(config, workdir, responder=brain)
        result = loop.run("Fix safe_divide.")

        self.assertTrue(result.success, msg=result.summary)
        self.assertTrue(result.tests_passed)
        self.assertIn("if b == 0", (workdir / "calc.py").read_text())
        # No planner/extractor/dispatcher — only the coder acts.
        self.assertEqual({e.actor for e in result.transcript}, {"coder"})

    def test_single_model_respects_gate(self):
        workdir = self._workdir()
        config = load_config(mock=True)
        config.loop.max_turns = 3
        config.loop.test_command = CHECK_CMD
        # No fix scripted → tests never pass → must fail, not hang.
        brain = MockBrain(coder_actions=[], test_command=CHECK_CMD, single_model=True)
        loop = build_single_model_loop(config, workdir, responder=brain)
        result = loop.run("won't be fixed")
        self.assertFalse(result.success)
        self.assertFalse(result.tests_passed)


if __name__ == "__main__":
    unittest.main()
