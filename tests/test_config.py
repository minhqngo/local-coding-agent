import os
import unittest
from unittest import mock

from coding_agent.config import load_config


class ConfigTest(unittest.TestCase):
    def test_ctx_and_dispatch_all_parsed(self):
        cfg = load_config()
        self.assertEqual(cfg.role("planner").ctx, 32768)
        self.assertTrue(cfg.role("planner").thinking)
        self.assertFalse(cfg.loop.dispatch_all)

    def test_rag_role_is_minicpm(self):
        # The REVIEW.md §4 swap: repo-Q&A moved to MiniCPM5-1B, others stay LFM2.
        cfg = load_config()
        self.assertEqual(cfg.role("rag").model, "MiniCPM5-1B")
        self.assertEqual(cfg.role("dispatcher").model, "LFM2-1.2B-Tool")
        self.assertEqual(cfg.role("extractor").model, "LFM2-350M-Extract")

    def test_mock_precedence_arg_over_env_over_yaml(self):
        # explicit arg wins over env
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "0"}):
            self.assertTrue(load_config(mock=True).mock)
        # env wins when no arg
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "1"}):
            self.assertTrue(load_config().mock)
        with mock.patch.dict(os.environ, {"CODING_AGENT_MOCK": "0"}):
            self.assertFalse(load_config().mock)
        # unset env, no arg → YAML default (mock absent → False)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODING_AGENT_MOCK", None)
            self.assertFalse(load_config().mock)


if __name__ == "__main__":
    unittest.main()
