"""Remote serving: the agent's HTTP client path over real sockets, and the
config plumbing (base_url_template / per-role base_url) that points the agent
at another machine."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coding_agent.config import load_config
from coding_agent.llm import ChatError, OpenAICompatClient


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions endpoint."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        # Echo enough of the request back to assert on it.
        reply = {
            "choices": [{
                "message": {
                    "content": f"echo:{payload['model']}:{payload['messages'][-1]['content']}"
                }
            }]
        }
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


class RemoteClientTest(unittest.TestCase):
    """OpenAICompatClient against a real HTTP server (no mocked requests)."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_chat_over_real_socket(self):
        client = OpenAICompatClient(
            role="planner",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="test-model",
        )
        out = client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(out, "echo:test-model:ping")

    def test_unreachable_host_raises_chaterror(self):
        # A dead endpoint must surface as ChatError (which the loop's P1
        # recovery absorbs), never as a raw exception.
        client = OpenAICompatClient(
            role="coder",
            base_url="http://127.0.0.1:1/v1",  # port 1: nothing listens
            model="m",
            timeout_s=2,
        )
        with self.assertRaises(ChatError):
            client.chat([{"role": "user", "content": "x"}])


class RemoteConfigTest(unittest.TestCase):
    """base_url_template and per-role base_url point roles at another machine."""

    def _write_config(self, body: str) -> str:
        path = Path(tempfile.mkdtemp()) / "models.yaml"
        path.write_text(body)
        return str(path)

    def test_base_url_template_points_all_roles_remote(self):
        cfg_path = self._write_config(
            "defaults:\n"
            '  base_url_template: "http://192.168.1.50:{port}/v1"\n'
            "roles:\n"
            "  planner: {model: p, port: 8001}\n"
            "  coder: {model: c, port: 8002}\n"
        )
        cfg = load_config(cfg_path, mock=True)
        self.assertEqual(cfg.role("planner").base_url, "http://192.168.1.50:8001/v1")
        self.assertEqual(cfg.role("coder").base_url, "http://192.168.1.50:8002/v1")

    def test_per_role_base_url_override_allows_mixed_fleet(self):
        # e.g. coder on the GPU box, extractor local.
        cfg_path = self._write_config(
            "roles:\n"
            "  coder: {model: c, port: 8002, base_url: 'http://192.168.1.50:8002/v1'}\n"
            "  extractor: {model: e, port: 8004}\n"
        )
        cfg = load_config(cfg_path, mock=True)
        self.assertEqual(cfg.role("coder").base_url, "http://192.168.1.50:8002/v1")
        self.assertEqual(cfg.role("extractor").base_url, "http://127.0.0.1:8004/v1")

    def test_real_client_uses_configured_base_url(self):
        from coding_agent.llm import make_client

        cfg_path = self._write_config(
            "defaults:\n"
            '  base_url_template: "http://10.0.0.7:{port}/v1"\n'
            "roles:\n"
            "  rag: {model: r, port: 8005}\n"
        )
        cfg = load_config(cfg_path, mock=False)
        client = make_client(cfg.role("rag"), mock=False)
        self.assertEqual(client.base_url, "http://10.0.0.7:8005/v1")


if __name__ == "__main__":
    unittest.main()
