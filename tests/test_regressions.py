"""Regression tests for bugs found by an independent review of the harness."""

import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

from coding_agent.config import SandboxConfig, load_config
from coding_agent.harness import CoderMemory, Harness, remember
from coding_agent.llm import ChatError, extract_json
from coding_agent.loop import ResumeState, build_loop
from coding_agent.roles import Planner, ToolCall
from coding_agent.sandbox import Sandbox, build_command_env
from coding_agent.tools import ToolContext, execute_tool


def _harness():
    config = load_config(mock=True)
    config.loop.test_after_edit = False
    d = Path(tempfile.mkdtemp(prefix="regress_"))
    return Harness(config, ToolContext(workdir=d, sandbox=Sandbox(config.sandbox, d))), d


class HarnessInputTest(unittest.TestCase):
    def test_non_string_command_is_a_tool_error_not_a_crash(self):
        h, _ = _harness()
        for bad in (None, 5):
            res = h.execute(ToolCall("run_shell", {"command": bad}))
            if bad is None:
                self.assertEqual(res.blocked, "invalid call")
        res = h.execute(ToolCall("run_shell", {"command": ["python", "-c", "print(7)"]}))
        self.assertIsNotNone(res.exit_code)  # coerced to text and executed

    def test_os_errors_become_tool_errors(self):
        h, d = _harness()
        (d / "calc.py").write_text("x = 1\n")
        res = h.execute(ToolCall("write_file", {"path": "calc.py/x.py", "content": "y"}))
        self.assertFalse(res.ok)
        self.assertIn("TOOL ERROR", res.output)
        ro = d / "ro.py"
        ro.write_text("a = 1\n")
        os.chmod(ro, stat.S_IREAD)
        try:
            res = h.execute(ToolCall("patch_file", {"path": "ro.py", "old_text": "a = 1",
                                                    "new_text": "a = 2"}))
            self.assertFalse(res.ok)
            self.assertIn("TOOL ERROR", res.output)
        finally:
            os.chmod(ro, stat.S_IWRITE | stat.S_IREAD)


class ResumeTest(unittest.TestCase):
    def test_resume_does_not_trust_saved_tests_passed(self):
        config = load_config(mock=True)
        config.loop.max_turns = 2
        config.loop.test_command = 'python -c "raise SystemExit(1)"'
        d = Path(tempfile.mkdtemp())

        def finish_now(role, messages):
            return json.dumps({"next": "finish", "thought": "", "summary": "done"})

        loop = build_loop(config, d, responder=finish_now)
        state = ResumeState(observations=[], raw_context=[], tests_passed=True,
                            transcript=[], turns_done=0)
        result = loop.run("x", resume=state)
        self.assertFalse(result.success)  # the gate ran the (failing) tests


class RagErrorTest(unittest.TestCase):
    def test_empty_rag_answer_does_not_crash_the_run(self):
        config = load_config(mock=True)
        config.loop.max_turns = 1

        def brain(role, messages):
            if role == "planner":
                return json.dumps({"next": "rag", "thought": "", "question": "where?"})
            raise ChatError("empty answer")

        result = build_loop(config, Path(tempfile.mkdtemp()), responder=brain).run("x")
        self.assertFalse(result.success)


class SandboxTest(unittest.TestCase):
    def test_finished_command_with_background_grandchild_returns_promptly(self):
        d = Path(tempfile.mkdtemp())
        (d / "bg.py").write_text(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print('started')\n"
        )
        sb = Sandbox(SandboxConfig(command_timeout_s=20), d)
        started = time.monotonic()
        res = sb.run(f'"{sys.executable}" bg.py')
        self.assertLess(time.monotonic() - started, 15)
        self.assertTrue(res.ok, res.output)
        self.assertIn("started", res.output)

    @unittest.skipUnless(os.name == "nt", "cmd.exe is case-insensitive")
    def test_denylist_ignores_case_on_windows(self):
        d = Path(tempfile.mkdtemp())
        sb = Sandbox(load_config(mock=True).sandbox, d)
        for cmd in ("DEL calc.py", "Rd /s /q pkg", "Erase x"):
            self.assertTrue(sb.run(cmd).refused, cmd)

    def test_store_alias_does_not_shadow_demoted_agent_venv(self):
        project = Path(tempfile.mkdtemp())
        agent = Path(tempfile.mkdtemp()) / "venv"
        agent_bin = str(agent / ("Scripts" if os.name == "nt" else "bin"))
        apps = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps"
        env = build_command_env(project, {"PATH": os.pathsep.join([agent_bin, "/usr/bin", apps]),
                                          "VIRTUAL_ENV": str(agent)})
        parts = env["PATH"].split(os.pathsep)
        self.assertLess(parts.index(agent_bin), parts.index(apps))


class FileTest(unittest.TestCase):
    def test_bom_file_can_be_patched_and_keeps_its_bom(self):
        d = Path(tempfile.mkdtemp())
        ctx = ToolContext(workdir=d, sandbox=Sandbox(SandboxConfig(), d))
        (d / "b.py").write_bytes(b"\xef\xbb\xbfx = 1\n")
        execute_tool(ctx, "patch_file", {"path": "b.py", "old_text": "x = 1", "new_text": "x = 2"})
        self.assertEqual((d / "b.py").read_bytes(), b"\xef\xbb\xbfx = 2\n")

    def test_stale_read_dropped_regardless_of_path_spelling(self):
        h, d = _harness()
        (d / "pkg").mkdir()
        (d / "pkg" / "calc.py").write_text("old = 1\n")
        mem = CoderMemory(10000)
        remember(mem, h.execute(ToolCall("read_file", {"path": "pkg\\calc.py"})))
        remember(mem, h.execute(ToolCall("write_file", {"path": "./pkg/calc.py", "content": "new = 2\n"})))
        self.assertNotIn("old = 1", mem.render())


class JsonTest(unittest.TestCase):
    def test_unclosed_bracket_in_prose(self):
        self.assertEqual(extract_json('Note [unclosed. {"next": "finish"}'), {"next": "finish"})

    def test_object_preferred_over_earlier_list(self):
        self.assertEqual(extract_json('Step [1]: {"next": "finish"}'), {"next": "finish"})


class PlannerBudgetTest(unittest.TestCase):
    def test_prompt_stays_within_budget(self):
        seen = {}

        class Client:
            def chat(self, messages, **kw):
                seen["chars"] = sum(len(m["content"]) for m in messages)
                return '{"next": "finish", "thought": ""}'

        obs = ["x" * 2600 for _ in range(8)]
        Planner(Client()).decide("task", obs, False, workspace="w" * 2000, budget_chars=10000)
        self.assertLessEqual(seen["chars"], 10000)


if __name__ == "__main__":
    unittest.main()
