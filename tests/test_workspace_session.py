"""Workspace snapshot + session persistence/resume."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent.session import SessionStore
from coding_agent.workspace import workspace_snapshot


class WorkspaceTest(unittest.TestCase):
    def test_non_git_dir_is_benign(self):
        d = Path(tempfile.mkdtemp())
        (d / "README.md").write_text("# Title\nsome docs")
        snap = workspace_snapshot(d)
        self.assertIn("(not a git repository)", snap)
        self.assertIn("README.md", snap)

    def test_git_dir_includes_branch(self):
        d = Path(tempfile.mkdtemp())
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, env={**env, "PATH": __import__("os").environ["PATH"]})
        if run("git", "init").returncode != 0:
            self.skipTest("git unavailable")
        (d / "f.txt").write_text("hi")
        run("git", "add", "-A")
        run("git", "commit", "-m", "init")
        snap = workspace_snapshot(d)
        self.assertIn("git root:", snap)
        self.assertIn("branch:", snap)


class SessionTest(unittest.TestCase):
    def test_save_load_latest_resolve(self):
        d = Path(tempfile.mkdtemp())
        store = SessionStore(d)
        sid = store.new_id()
        store.save(sid, {"task": "t", "turns_done": 2, "observations": ["a"],
                         "raw_context": [], "tests_passed": False, "transcript": []})
        loaded = store.load(sid)
        self.assertEqual(loaded["session_id"], sid)
        self.assertEqual(loaded["turns_done"], 2)
        self.assertEqual(store.latest()["session_id"], sid)
        self.assertEqual(store.resolve("latest")["session_id"], sid)
        self.assertEqual(store.resolve(sid)["session_id"], sid)
        self.assertIsNone(store.resolve("nonexistent-id"))

    def test_save_is_atomic_and_valid_json(self):
        import json
        d = Path(tempfile.mkdtemp())
        store = SessionStore(d)
        sid = store.new_id()
        store.save(sid, {"task": "t", "transcript": [{"turn": 1, "actor": "planner", "detail": "x"}]})
        path = d / ".coding-agent" / "sessions" / f"{sid}.json"
        self.assertTrue(path.exists())
        json.loads(path.read_text())  # must be valid
        # no leftover temp file
        self.assertFalse((path.with_suffix(".json.tmp")).exists())


if __name__ == "__main__":
    unittest.main()
