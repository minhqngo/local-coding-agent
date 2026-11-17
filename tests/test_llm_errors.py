"""The LLM client's error taxonomy, thinking handling, and tracing."""

import unittest
from unittest import mock

from coding_agent.llm import ChatError, ModelUnavailable, OpenAICompatClient
from coding_agent.roles import ToolCall
from coding_agent.trace import Tracer


class _Resp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
        self.text = str(data)

    def json(self):
        return self._data


def _client(**kw):
    return OpenAICompatClient(role="planner", base_url="http://x/v1", model="m", **kw)


def _reply(content, finish="stop", reasoning=None, usage=None):
    msg = {"content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}}


class ClientTest(unittest.TestCase):
    def test_empty_answer_after_thinking_explains_itself(self):
        with mock.patch("requests.post", lambda *a, **k: _Resp(_reply("", "length", "hmm " * 50))):
            with self.assertRaises(ChatError) as cm:
                _client(thinking=True, max_tokens=64).chat([{"role": "user", "content": "x"}])
        self.assertIn("max_tokens=64", str(cm.exception))
        self.assertIn("thinking", str(cm.exception))

    def test_context_overflow_is_unavailable_with_hint(self):
        err = {"error": {"message": "the request exceeds the available context size"}}
        with mock.patch("requests.post", lambda *a, **k: _Resp(err, status=400)):
            with self.assertRaises(ModelUnavailable) as cm:
                _client().chat([{"role": "user", "content": "x"}])
        self.assertIn("raise ctx", str(cm.exception))

    def test_thinking_false_is_sent_explicitly(self):
        captured = {}

        def post(url, json=None, **kw):
            captured.update(json)
            return _Resp(_reply("ok"))

        with mock.patch("requests.post", post):
            _client(thinking=False).chat([{"role": "user", "content": "x"}])
        self.assertEqual(captured["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_none_sends_nothing(self):
        captured = {}

        def post(url, json=None, **kw):
            captured.update(json)
            return _Resp(_reply("ok"))

        with mock.patch("requests.post", post):
            _client().chat([{"role": "user", "content": "x"}])
        self.assertNotIn("chat_template_kwargs", captured)

    def test_tracer_records_reasoning_usage_and_errors(self):
        tracer = Tracer()
        with mock.patch("requests.post", lambda *a, **k: _Resp(_reply('{"a":1}', reasoning="because"))):
            _client(tracer=tracer).chat([{"role": "user", "content": "x"}])
        with mock.patch("requests.post", lambda *a, **k: _Resp(_reply(""))):
            with self.assertRaises(ChatError):
                _client(tracer=tracer).chat([{"role": "user", "content": "x"}])
        ok, bad = tracer.events
        self.assertEqual(ok["reasoning"], "because")
        self.assertEqual(ok["usage"]["completion_tokens"], 5)
        self.assertIn("empty", bad["error"])
        self.assertEqual(tracer.summary()["model_calls"], 2)


class ToolCallShapesTest(unittest.TestCase):
    def test_canonical(self):
        c = ToolCall.from_obj({"name": "run_shell", "arguments": {"command": "ls"}})
        self.assertEqual((c.name, c.arguments), ("run_shell", {"command": "ls"}))

    def test_flat(self):
        c = ToolCall.from_obj({"name": "read_file", "path": "a.py"})
        self.assertEqual((c.name, c.arguments), ("read_file", {"path": "a.py"}))

    def test_keyed(self):
        c = ToolCall.from_obj({"run_shell": {"command": "ls"}})
        self.assertEqual((c.name, c.arguments), ("run_shell", {"command": "ls"}))

    def test_args_alias(self):
        self.assertEqual(ToolCall.from_obj({"name": "x", "args": {"a": 1}}).arguments, {"a": 1})

    def test_planner_next_aliases(self):
        from coding_agent.roles import PlannerDecision

        d = PlannerDecision.from_obj({"next": "read", "arguments": {"path": "a.py"}})
        self.assertEqual(d.next, "tool")
        self.assertEqual((d.tool_call().name, d.tool_call().arguments), ("read_file", {"path": "a.py"}))
        self.assertEqual(PlannerDecision.from_obj({"next": "done"}).next, "finish")

    def test_garbage(self):
        self.assertEqual(ToolCall.from_obj("nope").name, "")


if __name__ == "__main__":
    unittest.main()
