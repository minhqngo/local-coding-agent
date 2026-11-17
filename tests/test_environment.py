"""Sandbox environment, encoding, line endings and timeouts — the boring things
that silently break real runs."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from coding_agent.config import SandboxConfig
from coding_agent.sandbox import Sandbox, build_command_env, find_project_venv
from coding_agent.tools import ToolContext, ToolError, execute_tool


def _fake_venv(root: Path, name: str = ".venv") -> Path:
    venv = root / name
    (venv / ("Scripts" if os.name == "nt" else "bin")).mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = x\n")
    return venv


class CommandEnvTest(unittest.TestCase):
    def test_agent_venv_is_demoted_to_end_of_path(self):
        project = Path(tempfile.mkdtemp())
        agent_venv = Path(tempfile.mkdtemp()) / "agentvenv"
        agent_bin = agent_venv / ("Scripts" if os.name == "nt" else "bin")
        base = {"PATH": os.pathsep.join([str(agent_bin), "/usr/bin"]), "VIRTUAL_ENV": str(agent_venv)}
        env = build_command_env(project, base)
        parts = [os.path.normcase(p) for p in env["PATH"].split(os.pathsep)]
        self.assertEqual(parts[-1], os.path.normcase(str(agent_bin)))
        self.assertLess(parts.index(os.path.normcase("/usr/bin")), len(parts) - 1)
        self.assertNotIn("VIRTUAL_ENV", env)

    def test_project_venv_goes_first(self):
        project = Path(tempfile.mkdtemp())
        venv = _fake_venv(project)
        self.assertEqual(find_project_venv(project), venv)
        env = build_command_env(project, {"PATH": "/usr/bin"})
        first = env["PATH"].split(os.pathsep)[0]
        self.assertTrue(first.startswith(str(venv)))
        self.assertEqual(env["VIRTUAL_ENV"], str(venv))

    def test_utf8_forced_for_child_python(self):
        env = build_command_env(Path(tempfile.mkdtemp()), {"PATH": ""})
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_describe_mentions_os_and_python(self):
        d = Path(tempfile.mkdtemp())
        text = Sandbox(SandboxConfig(), d).describe()
        self.assertIn("os:", text)
        self.assertIn("python:", text)


class RunCommandTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.sb = Sandbox(SandboxConfig(command_timeout_s=60), self.workdir)

    def test_non_ascii_output_survives(self):
        res = self.sb.run(f'"{sys.executable}" -c "print(\'→ ✓ é\')"')
        self.assertTrue(res.ok, res.output)
        self.assertIn("→ ✓ é", res.output)

    def test_stderr_is_captured(self):
        res = self.sb.run(f'"{sys.executable}" -c "import sys; sys.stderr.write(\'boom\'); sys.exit(2)"')
        self.assertEqual(res.exit_code, 2)
        self.assertIn("boom", res.output)

    def test_stdin_is_closed(self):
        # input() must get EOF immediately instead of waiting for the timeout.
        started = time.monotonic()
        res = self.sb.run(f'"{sys.executable}" -c "input()"', timeout_s=30)
        self.assertLess(time.monotonic() - started, 15)
        self.assertFalse(res.ok)

    def test_bare_python_does_not_start_a_waiting_repl(self):
        # On Windows, stdin=DEVNULL looks like a terminal: `python` would sit at
        # an interactive prompt until the timeout.
        started = time.monotonic()
        res = self.sb.run(f'"{sys.executable}"', timeout_s=30)
        self.assertLess(time.monotonic() - started, 10)
        self.assertFalse(res.timed_out)

    def test_timeout_kills_grandchildren(self):
        # The shell starts python, python starts another sleeping python: a
        # naive timeout kills only the shell and then waits forever on the pipe.
        script = self.workdir / "spawn.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "time.sleep(60)\n"
        )
        started = time.monotonic()
        res = self.sb.run(f'"{sys.executable}" spawn.py', timeout_s=2)
        self.assertTrue(res.timed_out)
        self.assertLess(time.monotonic() - started, 20)


class FileEncodingTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.ctx = ToolContext(workdir=self.workdir, sandbox=Sandbox(SandboxConfig(), self.workdir))

    def test_write_non_ascii_is_utf8(self):
        execute_tool(self.ctx, "write_file", {"path": "u.py", "content": "ARROW = '→'\n"})
        self.assertEqual((self.workdir / "u.py").read_bytes(), "ARROW = '→'\n".encode("utf-8"))

    def test_patch_preserves_crlf(self):
        path = self.workdir / "w.py"
        path.write_bytes(b"a = 1\r\nb = 2\r\n")
        execute_tool(self.ctx, "patch_file", {"path": "w.py", "old_text": "a = 1\nb = 2",
                                              "new_text": "a = 10\nb = 20"})
        self.assertEqual(path.read_bytes(), b"a = 10\r\nb = 20\r\n")

    def test_rewrite_preserves_crlf(self):
        path = self.workdir / "w.py"
        path.write_bytes(b"x = 1\r\n")
        execute_tool(self.ctx, "write_file", {"path": "w.py", "content": "x = 2\ny = 3\n"})
        self.assertEqual(path.read_bytes(), b"x = 2\r\ny = 3\r\n")

    def test_read_shows_lf(self):
        (self.workdir / "w.py").write_bytes(b"x = 1\r\n")
        self.assertNotIn("\r", execute_tool(self.ctx, "read_file", {"path": "w.py"}))

    def test_patch_refuses_non_utf8(self):
        (self.workdir / "l.py").write_bytes("café = 1\n".encode("latin-1"))
        with self.assertRaises(ToolError):
            execute_tool(self.ctx, "patch_file", {"path": "l.py", "old_text": "= 1", "new_text": "= 2"})

    def test_content_as_list_of_lines(self):
        execute_tool(self.ctx, "write_file", {"path": "x.py", "content": ["a = 1", "b = 2"]})
        self.assertEqual((self.workdir / "x.py").read_text(), "a = 1\nb = 2")


if __name__ == "__main__":
    unittest.main()
