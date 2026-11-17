"""GGUF metadata reading, memory estimates, and llama-server command lines."""

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent import serving
from coding_agent.config import load_config
from coding_agent.gguf import model_shape, read_metadata


def write_gguf(path: Path, kv: dict, tensors: dict | None = None) -> Path:
    """Write a GGUF v3 file with the given metadata and {name: n_bytes} tensors."""

    tensors = tensors or {}

    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for key, value in kv.items():
        out += s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + s(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
            out += b"".join(s(v) for v in value)
        elif isinstance(value, list):
            out += struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(value))
            out += b"".join(struct.pack("<I", v) for v in value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    offset = 0
    for name, n_bytes in tensors.items():
        # name, n_dims=1, dims=[n_bytes], type=0, offset
        out += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n_bytes) + struct.pack("<I", 0)
        out += struct.pack("<Q", offset)
        offset += n_bytes
    out += b"\0" * (-len(out) % 32)  # align the data section
    path.write_bytes(out + b"\0" * max(offset, 1024))
    return path


class GGUFTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_dense_model_kv(self):
        p = write_gguf(self.dir / "dense.gguf", {
            "general.architecture": "qwen2",
            "tokenizer.ggml.tokens": ["a", "b", "c"],  # skipped, not decoded
            "qwen2.block_count": 36, "qwen2.embedding_length": 2048,
            "qwen2.attention.head_count": 16, "qwen2.attention.head_count_kv": 2,
            "qwen2.context_length": 32768,
        })
        meta = read_metadata(p)
        self.assertEqual(meta["qwen2.block_count"], 36)
        shape = model_shape(p)
        # 36 layers × 2 kv heads × (128 + 128) dims × 2 bytes (f16)
        self.assertEqual(shape.kv_bytes_per_token, 36 * 2 * 256 * 2)
        self.assertEqual(shape.trained_ctx, 32768)

    def test_hybrid_per_layer_heads(self):
        # LFM2-style: 0 kv heads on convolution layers.
        p = write_gguf(self.dir / "hybrid.gguf", {
            "general.architecture": "lfm2", "lfm2.block_count": 4,
            "lfm2.embedding_length": 1024, "lfm2.attention.head_count": 16,
            "lfm2.attention.head_count_kv": [0, 8, 0, 8],
        })
        shape = model_shape(p)
        self.assertEqual(shape.n_attention_layers, 2)
        self.assertEqual(shape.kv_bytes_per_token, 2 * 8 * (64 + 64) * 2)

    def test_full_attention_interval(self):
        # Qwen3.5-style: only every 4th layer keeps a KV cache.
        p = write_gguf(self.dir / "q35.gguf", {
            "general.architecture": "qwen35", "qwen35.block_count": 8,
            "qwen35.attention.head_count": 8, "qwen35.attention.head_count_kv": 2,
            "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
            "qwen35.full_attention_interval": 4,
        })
        self.assertEqual(model_shape(p).n_attention_layers, 2)

    def test_not_gguf(self):
        p = self.dir / "x.gguf"
        p.write_bytes(b"nope")
        with self.assertRaises(ValueError):
            read_metadata(p)


class MoETest(unittest.TestCase):
    def test_tensor_sizes_and_expert_offload(self):
        from coding_agent.gguf import expert_bytes_on_cpu, tensor_sizes

        d = Path(tempfile.mkdtemp())
        p = write_gguf(d / "moe.gguf", {"general.architecture": "qwen35moe"}, {
            "token_embd.weight": 1000,
            "blk.0.attn_q.weight": 100, "blk.0.ffn_up_exps.weight": 5000,
            "blk.0.ffn_down_exps.weight": 5000, "blk.0.ffn_up_shexp.weight": 50,
            "blk.1.attn_q.weight": 100, "blk.1.ffn_up_exps.weight": 5000,
        })
        sizes = tensor_sizes(p)
        self.assertEqual(sizes["blk.0.ffn_up_exps.weight"], 5000)
        self.assertEqual(sizes["blk.1.ffn_up_exps.weight"], 5000)
        # Layer 0's routed experts only — not attention, not the shared expert.
        self.assertEqual(expert_bytes_on_cpu(p, 1), 10000)
        self.assertEqual(expert_bytes_on_cpu(p, 2), 15000)

    def test_estimate_and_command_with_cpu_moe(self):
        d = Path(tempfile.mkdtemp())
        p = write_gguf(d / "moe.gguf", {"general.architecture": "x", "x.block_count": 2}, {
            "blk.0.ffn_up_exps.weight": 8000, "blk.1.ffn_up_exps.weight": 8000,
            "blk.0.attn_q.weight": 1000,
        })
        spec = serving.ServerSpec(role="coder", serves=["coder"], port=1, host="127.0.0.1",
                                  alias="m", model_path=p, draft_path=None, ctx=None,
                                  gpu_layers=99, cpu_moe_layers=2)
        est = serving.estimate(spec)
        self.assertEqual(est.experts_on_cpu, 16000)
        self.assertLess(est.gpu_bytes, est.total - 15000)
        self.assertEqual(serving.build_command(spec, "llama-server")[-2:], ["--n-cpu-moe", "2"])

    def test_moe_profile_shares_one_server(self):
        cfg = load_config("gpu-12gb-moe")
        specs = serving.server_specs(cfg)
        self.assertEqual([s.role for s in specs], ["coder"])
        self.assertEqual(specs[0].cpu_moe_layers, 24)
        self.assertEqual(len(specs[0].serves), 5)

    def test_kv_quant_shrinks_estimate(self):
        self.assertAlmostEqual(serving._kv_ratio(["-ctk", "q8_0", "-ctv", "q8_0"]), 34 / 64)
        self.assertEqual(serving._kv_ratio([]), 1.0)


class ServingTest(unittest.TestCase):
    def test_6gb_profile_starts_three_servers(self):
        cfg = load_config("gpu-6gb")
        specs = serving.server_specs(cfg)
        self.assertEqual(sorted(s.role for s in specs), ["coder", "extractor", "planner"])
        planner = next(s for s in specs if s.role == "planner")
        self.assertEqual(sorted(planner.serves), ["dispatcher", "planner", "rag"])

    def test_naming_a_sharing_role_starts_its_server(self):
        cfg = load_config("gpu-6gb")
        self.assertEqual([s.role for s in serving.server_specs(cfg, ["rag"])], ["planner"])

    def test_cpu_override(self):
        cfg = load_config("gpu-12gb")
        self.assertTrue(all(s.gpu_layers == 0 for s in serving.server_specs(cfg, gpu_layers=0)))

    def test_command_line(self):
        cfg = load_config("gpu-6gb")
        spec = next(s for s in serving.server_specs(cfg) if s.role == "coder")
        cmd = serving.build_command(spec, "llama-server")
        self.assertIn("--parallel", cmd)
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")
        self.assertEqual(cmd[cmd.index("-c") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--alias") + 1], "Qwen2.5-Coder-3B-Instruct")

    def test_draft_flags_follow_the_binary(self):
        cfg = load_config("gpu-12gb")
        spec = next(s for s in serving.server_specs(cfg) if s.role == "coder")
        d = Path(tempfile.mkdtemp())
        spec.draft_path = d / "draft.gguf"
        spec.draft_path.write_bytes(b"GGUF")
        for help_text, flag in (("--spec-draft-n-max N", "--spec-draft-n-max"),
                                ("--draft-max N", "--draft-max")):
            serving._help_text.cache_clear()
            with mock.patch.object(serving, "_help_text", lambda b, t=help_text: t):
                cmd = serving.build_command(spec, "llama-server")
            self.assertIn("-md", cmd)
            self.assertIn(flag, cmd)

    def test_estimate_uses_file_and_kv(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "m.gguf", {
            "general.architecture": "qwen2", "qwen2.block_count": 10,
            "qwen2.embedding_length": 1024, "qwen2.attention.head_count": 8,
            "qwen2.attention.head_count_kv": 2,
        })
        spec = serving.ServerSpec(role="coder", serves=["coder"], port=1, host="127.0.0.1",
                                  alias="m", model_path=d / "m.gguf", draft_path=None,
                                  ctx=1000, gpu_layers=99)
        est = serving.estimate(spec)
        self.assertEqual(est.kv, 10 * 2 * (128 + 128) * 2 * 1000)
        self.assertEqual(est.gpu_bytes, est.total)
        spec.gpu_layers = 0
        self.assertEqual(serving.estimate(spec).gpu_bytes, 0)

    def test_missing_model_reported(self):
        cfg = load_config("gpu-6gb")
        spec = serving.server_specs(cfg)[0]
        spec.model_path = Path(tempfile.mkdtemp()) / "nope.gguf"
        self.assertEqual(serving.estimate(spec).missing, ["nope.gguf"])


if __name__ == "__main__":
    unittest.main()
