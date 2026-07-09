import unittest
from unittest import mock

from coding_agent.llm import ChatError, MockLLMClient, OpenAICompatClient, extract_json


class ExtractJsonTest(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        text = 'here you go:\n```json\n{"next": "coder"}\n```\nthanks'
        self.assertEqual(extract_json(text), {"next": "coder"})

    def test_prose_wrapped(self):
        text = 'Sure! {"action": "write_file", "path": "x.py"} done.'
        self.assertEqual(extract_json(text)["action"], "write_file")

    def test_strips_think_block(self):
        text = '<think>let me reason\nabout this</think>\n{"next": "finish"}'
        self.assertEqual(extract_json(text), {"next": "finish"})

    def test_strips_unterminated_think_block(self):
        # A truncated thinking model leaves an open <think>; stray braces inside
        # must NOT be parsed as the decision (B11).
        text = 'ok\n{"next": "coder"}\n<think>hmm maybe {"next": "wrong"}'
        self.assertEqual(extract_json(text), {"next": "coder"})

    def test_iterates_past_invalid_first_span(self):
        # First balanced span is invalid JSON; a later one is the real payload.
        text = 'note {not: json} then {"next": "finish"}'
        self.assertEqual(extract_json(text), {"next": "finish"})

    def test_array(self):
        self.assertEqual(extract_json("prefix [1, 2, 3] suffix"), [1, 2, 3])

    def test_braces_inside_strings(self):
        text = '{"message": "use {curly} braces", "ok": true}'
        self.assertEqual(extract_json(text)["message"], "use {curly} braces")

    def test_no_json_raises(self):
        with self.assertRaises(ChatError):
            extract_json("no json at all here")


class MockClientTest(unittest.TestCase):
    def test_delegates_to_responder(self):
        client = MockLLMClient(role="planner", responder=lambda role, msgs: f"{role}:{len(msgs)}")
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "planner:1")


class ThinkingPayloadTest(unittest.TestCase):
    def test_thinking_sends_chat_template_kwargs(self):
        # D5: a thinking role must send enable_thinking to the server.
        captured = {}

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json=None, **kw):
            captured.update(json)
            return FakeResp()

        client = OpenAICompatClient(
            role="planner", base_url="http://x/v1", model="m", thinking=True
        )
        with mock.patch("requests.post", fake_post):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(captured["chat_template_kwargs"], {"enable_thinking": True})


if __name__ == "__main__":
    unittest.main()
