import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.config import DEFAULT_PROFILE, load_config


def _write(body: str) -> str:
    path = Path(tempfile.mkdtemp()) / "profile.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class ProfileTest(unittest.TestCase):
    def test_default_profile_is_6gb(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODING_AGENT_PROFILE", None)
            cfg = load_config()
        self.assertEqual(cfg.profile, DEFAULT_PROFILE)
        self.assertEqual(cfg.profile, "gpu-6gb")

    def test_6gb_profile_shares_the_planner(self):
        cfg = load_config("gpu-6gb")
        # rag and dispatcher reuse the planner's server: same port & model...
        for role in ("rag", "dispatcher"):
            self.assertEqual(cfg.role(role).shares, "planner")
            self.assertEqual(cfg.role(role).port, cfg.role("planner").port)
            self.assertEqual(cfg.role(role).model, cfg.role("planner").model)
            self.assertFalse(cfg.role(role).served)
        # ...but keep their own sampling settings.
        self.assertEqual(cfg.role("dispatcher").temperature, 0.0)
        self.assertEqual(sorted(r.name for r in cfg.served_roles()), ["coder", "extractor", "planner"])
        self.assertEqual(sorted(cfg.sharers_of("planner")), ["dispatcher", "rag"])

    def test_12gb_profile_full_team(self):
        cfg = load_config("gpu-12gb")
        self.assertEqual(cfg.role("rag").model, "MiniCPM5-1B")
        self.assertTrue(cfg.role("planner").thinking)
        self.assertEqual(cfg.role("extractor").gpu_layers, 0)  # glue on CPU
        self.assertEqual(len(cfg.served_roles()), 5)

    def test_agent_yaml_is_merged_under_profile(self):
        cfg = load_config("gpu-6gb")
        self.assertEqual(cfg.loop.test_command, "python -m pytest -q")
        self.assertEqual(cfg.loop.max_repeated_outcomes, 5)
        self.assertIn("shutdown", cfg.sandbox.denylist)

    def test_profile_can_override_behaviour(self):
        path = _write("roles:\n  coder: {model: c, port: 9002}\nloop:\n  max_turns: 7\n")
        cfg = load_config(path, mock=True)
        self.assertEqual(cfg.loop.max_turns, 7)
        self.assertEqual(cfg.loop.test_command, "python -m pytest -q")  # still from agent.yaml

    def test_env_var_selects_profile(self):
        with mock.patch.dict(os.environ, {"CODING_AGENT_PROFILE": "gpu-12gb"}):
            self.assertEqual(load_config().profile, "gpu-12gb")

    def test_unknown_profile_lists_available(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_config("gpu-999gb")
        self.assertIn("gpu-6gb", str(cm.exception))

    def test_share_unknown_role_is_an_error(self):
        path = _write("roles:\n  rag: {share: nobody}\n")
        with self.assertRaises(ValueError):
            load_config(path, mock=True)

    def test_thinking_unset_is_none(self):
        path = _write("roles:\n  coder: {model: c, port: 9002}\n")
        self.assertIsNone(load_config(path, mock=True).role("coder").thinking)

    def test_mock_precedence_arg_over_env_over_yaml(self):
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "0"}):
            self.assertTrue(load_config(mock=True).mock)
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "1"}):
            self.assertTrue(load_config().mock)
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "0"}):
            self.assertFalse(load_config().mock)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODING_AGENT_MOCK", None)
            self.assertFalse(load_config().mock)


if __name__ == "__main__":
    unittest.main()
