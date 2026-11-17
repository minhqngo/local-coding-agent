"""End-to-end loop behaviour for the failure modes seen in real runs."""

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.config import load_config
from coding_agent.llm import ModelUnavailable
from coding_agent.loop import build_loop
from coding_agent.mock_brain import MockBrain
from coding_agent.single_model import build_single_model_loop
from coding_agent.trace import Tracer, load_trace, render_trace
from coding_agent.workspace import workspace_snapshot


def _workdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="behaviour_"))
    (d / "calc.py").write_text("def f():\n    return 1\n")
    return d


class Scripted:
    """Planner replies come from a list; other roles are recorded."""

    def __init__(self, planner_replies, extractor_reply='{"message": "summarised"}'):
        self.planner_replies = list(planner_replies)
        self.extractor_reply = extractor_reply
        self.calls = []

    def __call__(self, role, messages):
        self.calls.append(role)
        if role == "planner":
            reply = self.planner_replies.pop(0) if self.planner_replies else None
            return json.dumps(reply or {"next": "finish", "thought": "", "summary": "x"})
        if role == "extractor":
            return self.extractor_reply
        return "{}"


def _tool(cmd):
    return {"next": "tool", "thought": "run", "tool": {"name": "run_shell", "arguments": {"command": cmd}}}


class ObservationTest(unittest.TestCase):
    def test_short_output_is_verbatim_without_extractor(self):
        cfg = load_config(mock=True)
        brain = Scripted([_tool('python -c "print(42)"')])
        loop = build_loop(cfg, _workdir(), responder=brain)
        loop.config.loop.max_turns = 1
        loop.run("x")
        self.assertNotIn("extractor", brain.calls)
        obs = loop.observations[0]
        self.assertIn("exit code 0", obs)
        self.assertIn("42", obs)

    def test_long_output_goes_through_extractor_but_facts_stay(self):
        cfg = load_config(mock=True)
        long_cmd = 'python -c "import sys; print(\'x\' * 3000); sys.exit(1)"'
        brain = Scripted([_tool(long_cmd)],
                         extractor_reply='{"tests_passed": true, "message": "all good!"}')
        loop = build_loop(cfg, _workdir(), responder=brain)
        loop.config.loop.max_turns = 1
        loop.run("x")
        self.assertIn("extractor", brain.calls)
        obs = loop.observations[0]
        self.assertTrue(obs.startswith("run_shell"))
        self.assertIn("exit code 1", obs)           # the harness's fact...
        self.assertNotIn("tests_passed", obs)       # ...and the lie is gone

    def test_small_file_read_is_shown_to_planner_and_coder(self):
        # A planner that can't see the file just asks to read it again (seen
        # in a real trace), so small files are shown to both.
        cfg = load_config(mock=True)
        read = {"next": "tool", "thought": "", "tool": {"name": "read_file", "arguments": {"path": "calc.py"}}}
        loop = build_loop(cfg, _workdir(), responder=Scripted([read]))
        loop.config.loop.max_turns = 1
        loop.run("x")
        self.assertIn("return 1", loop.observations[0])
        self.assertIn("return 1", loop.coder_memory.render())

    def test_tool_name_in_next_is_normalised(self):
        # {"next": "read_file", "arguments": {...}} — a shape small planners use.
        cfg = load_config(mock=True)
        odd = {"next": "read_file", "thought": "", "arguments": {"path": "calc.py"}}
        loop = build_loop(cfg, _workdir(), responder=Scripted([odd]))
        loop.config.loop.max_turns = 1
        loop.run("x")
        self.assertTrue(loop.observations[0].startswith("read_file `calc.py` → ok"))

    def test_coder_sees_the_original_task(self):
        cfg = load_config(mock=True)
        seen = {}

        def brain(role, messages):
            if role == "planner":
                return json.dumps({"next": "coder", "thought": "", "instruction": "fix it"})
            if role == "coder":
                seen["prompt"] = messages[-1]["content"]
                return '{"action": "read_file", "path": "calc.py"}'
            return "{}"

        loop = build_loop(cfg, _workdir(), responder=brain)
        loop.config.loop.max_turns = 1
        loop.run("make f return 0.0 for zero")
        self.assertIn("make f return 0.0 for zero", seen["prompt"])
        self.assertIn("fix it", seen["prompt"])


class StopConditionsTest(unittest.TestCase):
    def test_dead_server_stops_the_run_immediately(self):
        cfg = load_config(mock=True)
        calls = []

        def dead(role, messages):
            calls.append(role)
            raise ModelUnavailable(role, "cannot reach http://127.0.0.1:8001/v1")

        loop = build_loop(cfg, _workdir(), responder=dead)
        with self.assertRaises(ModelUnavailable):
            loop.run("x")
        self.assertEqual(calls, ["planner"])  # one call, not dozens of "retries"

    def test_stuck_agent_stops_early(self):
        cfg = load_config(mock=True)
        cfg.loop.max_turns = 25
        cfg.loop.max_repeated_outcomes = 4
        cmds = [_tool(f'python -c "import sys; sys.exit(\'No module named pip\')" {i}') for i in range(25)]
        loop = build_loop(cfg, _workdir(), responder=Scripted(cmds))
        result = loop.run("install things")
        self.assertFalse(result.success)
        self.assertIn("Stopped early", result.summary)
        self.assertLessEqual(result.turns, 4)
        self.assertTrue(any("NOTICE" in o for o in loop.observations))

    def test_single_model_also_stops_when_stuck(self):
        cfg = load_config(mock=True)
        cfg.loop.max_repeated_outcomes = 3
        actions = [{"action": "run_shell", "command": f'python -c "exit(5)" {i}'} for i in range(20)]
        brain = MockBrain(coder_actions=actions, single_model=True)
        result = build_single_model_loop(cfg, _workdir(), responder=brain).run("x")
        self.assertIn("Stopped early", result.summary)


class TraceTest(unittest.TestCase):
    def test_trace_records_models_and_tools(self):
        cfg = load_config(mock=True)
        cfg.loop.test_command = 'python -c "print(1)"'
        path = Path(tempfile.mkdtemp()) / "t.trace.jsonl"
        tracer = Tracer(path)
        brain = MockBrain(coder_actions=[{"action": "write_file", "path": "a.py", "content": "x = 1\n"}],
                          test_command=cfg.loop.test_command)
        result = build_loop(cfg, _workdir(), responder=brain, tracer=tracer).run("x")
        self.assertTrue(result.success)
        events = load_trace(path)
        kinds = {e["kind"] for e in events}
        self.assertTrue({"run", "model", "tool"} <= kinds)
        planner_call = next(e for e in events if e["kind"] == "model" and e["role"] == "planner")
        self.assertIn("TASK:", planner_call["messages"][-1]["content"])  # the exact prompt
        text = render_trace(events)
        self.assertIn("MODEL planner", text)
        self.assertIn("TOOL  write_file", text)
        self.assertGreaterEqual(tracer.summary()["model_calls"], 3)


class WorkspaceTest(unittest.TestCase):
    def test_snapshot_lists_files_and_environment(self):
        d = _workdir()
        (d / ".venv" / "lib").mkdir(parents=True)
        (d / ".venv" / "lib" / "junk.py").write_text("")
        (d / "pkg").mkdir()
        (d / "pkg" / "mod.py").write_text("")
        snap = workspace_snapshot(d, environment="os: Test", test_command="pytest -q")
        self.assertIn("calc.py", snap)
        self.assertIn("pkg/mod.py", snap)
        self.assertNotIn("junk.py", snap)
        self.assertIn("os: Test", snap)
        self.assertIn("pytest -q", snap)

    def test_empty_dir_says_so(self):
        snap = workspace_snapshot(Path(tempfile.mkdtemp()))
        self.assertIn("(empty directory)", snap)


if __name__ == "__main__":
    unittest.main()
