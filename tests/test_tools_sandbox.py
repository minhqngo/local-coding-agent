import tempfile
import unittest
from pathlib import Path

from coding_agent.config import SandboxConfig
from coding_agent.sandbox import Sandbox
from coding_agent.tools import ToolContext, ToolError, execute_tool


class ToolsSandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tools_test_")
        self.workdir = Path(self.tmp)
        sandbox = Sandbox(
            SandboxConfig(backend="subprocess", denylist=["rm -rf /"]), self.workdir
        )
        self.ctx = ToolContext(workdir=self.workdir, sandbox=sandbox)

    def test_write_then_read(self):
        out = execute_tool(self.ctx, "write_file", {"path": "a.txt", "content": "hello"})
        self.assertIn("Created", out)
        read = execute_tool(self.ctx, "read_file", {"path": "a.txt"})
        self.assertIn("hello", read)

    def test_path_escape_refused(self):
        with self.assertRaises(ToolError):
            execute_tool(self.ctx, "read_file", {"path": "../../etc/passwd"})

    def test_run_shell_ok(self):
        out = execute_tool(self.ctx, "run_shell", {"command": "echo hi"})
        self.assertIn("hi", out)
        self.assertIn("[OK]", out)

    def test_denylist_refused(self):
        result = self.ctx.sandbox.run("rm -rf / --no-preserve-root")
        self.assertTrue(result.refused)
        self.assertEqual(result.exit_code, 126)

    def test_dangerous_rm_variants_refused(self):
        # B5: crude bypasses of a substring denylist are caught by the smart rm.
        for cmd in ("rm -rf ~", "rm -fr /", "rm -rf .", "rm -rf /home/x"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self.ctx.sandbox.run(cmd).refused)

    def test_denylist_no_false_positive(self):
        # B5: a command merely CONTAINING a denied word (as an argument) is fine.
        sb = Sandbox(
            SandboxConfig(backend="subprocess", denylist=["shutdown", "reboot", "dd if="]),
            self.workdir,
        )
        (self.workdir / "notes.txt").write_text("shutdown reboot")
        self.assertFalse(sb.run("grep -r shutdown .").refused)
        # ...but the actual command name is still blocked.
        self.assertTrue(sb.run("shutdown now").refused)
        self.assertTrue(sb.run("dd if=/dev/zero of=x bs=1").refused)

    def test_missing_required_arg(self):
        with self.assertRaises(ToolError):
            execute_tool(self.ctx, "write_file", {"path": "x"})

    def test_unknown_tool(self):
        with self.assertRaises(ToolError):
            execute_tool(self.ctx, "nope", {})

    def test_search_unavailable_without_index(self):
        out = execute_tool(self.ctx, "search_codebase", {"query": "auth"})
        self.assertIn("unavailable", out)

    def test_patch_file_exactly_once(self):
        execute_tool(self.ctx, "write_file", {"path": "m.py", "content": "def f():\n    return -1\n"})
        out = execute_tool(self.ctx, "patch_file",
                           {"path": "m.py", "old_text": "return -1", "new_text": "return 0"})
        self.assertIn("Patched", out)
        self.assertIn("return 0", (self.workdir / "m.py").read_text())

    def test_patch_file_zero_match_refused(self):
        execute_tool(self.ctx, "write_file", {"path": "m.py", "content": "x = 1\n"})
        with self.assertRaises(ToolError) as cm:
            execute_tool(self.ctx, "patch_file", {"path": "m.py", "old_text": "NOPE", "new_text": "y"})
        self.assertIn("found 0", str(cm.exception))
        # The current content is in the error, so the next try can copy it.
        self.assertIn("Current content of m.py:\nx = 1", str(cm.exception))

    def test_edit_that_breaks_python_is_rejected(self):
        execute_tool(self.ctx, "write_file", {"path": "c.py", "content": "def f(a, b):\n    return a / b\n"})
        with self.assertRaises(ToolError) as cm:
            # The exact one-line patch a 3B model produced in a real trace.
            execute_tool(self.ctx, "patch_file", {"path": "c.py", "old_text": "def f(a, b):",
                                                  "new_text": "def f(a, b): if b == 0:"})
        self.assertIn("NOT changed", str(cm.exception))
        self.assertIn("line 1", str(cm.exception))
        self.assertEqual((self.workdir / "c.py").read_text(), "def f(a, b):\n    return a / b\n")
        with self.assertRaises(ToolError):
            execute_tool(self.ctx, "write_file", {"path": "c.py", "content": "def f(:\n"})
        # Non-Python files are not checked.
        execute_tool(self.ctx, "write_file", {"path": "notes.txt", "content": "def f(:\n"})

    def test_patch_file_multi_match_refused(self):
        execute_tool(self.ctx, "write_file", {"path": "m.py", "content": "x=1\nx=1\n"})
        with self.assertRaises(ToolError) as cm:
            execute_tool(self.ctx, "patch_file", {"path": "m.py", "old_text": "x=1", "new_text": "x=2"})
        self.assertIn("2 times", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
