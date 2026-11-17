"""Read just enough of a GGUF file to answer "will this fit on my GPU?".

A model's memory footprint while serving is roughly

    weights            (≈ the .gguf file size)
  + KV cache           (grows linearly with the context window, ``ctx``)
  + compute buffers    (scratch space; a few hundred MB, mostly constant)

The weights are easy (``os.path.getsize``). The KV cache is the part people get
wrong: it depends on the architecture — how many layers use attention, how many
key/value heads they have, and how wide each head is. All of that is stored in
the GGUF header as key/value metadata, so we parse the header (not the tensors)
and do the arithmetic ourselves.

GGUF layout (v2/v3, little-endian)::

    magic "GGUF" | version u32 | n_tensors u64 | n_kv u64 | n_kv × (key, type, value) | ...

This module is dependency-free and only reads the header, so it is fast even
for multi-GB files (the tokenizer vocabulary array is skipped, not decoded).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

# GGUF value types -> struct format (scalars only; 8 = string, 9 = array).
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9
# Numeric arrays longer than this are skipped (we only need per-layer arrays).
_MAX_KEPT_ARRAY = 4096


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file in GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        etype = _read(f, "<I")
        count = _read(f, "<Q")
        if etype in _SCALARS and count <= _MAX_KEPT_ARRAY:
            return [_read(f, _SCALARS[etype]) for _ in range(count)]
        # Skip big arrays (e.g. the tokenizer's 150k-entry vocabulary).
        if etype in _SCALARS:
            f.seek(struct.calcsize(_SCALARS[etype]) * count, 1)
        elif etype == _STRING:
            for _ in range(count):
                f.seek(_read(f, "<Q"), 1)
        else:
            raise GGUFError(f"nested array type {etype} not supported")
        return None
    raise GGUFError(f"unknown GGUF value type {vtype}")


def _read_header(f: BinaryIO, path: str | Path) -> tuple[dict[str, Any], int]:
    if f.read(4) != b"GGUF":
        raise GGUFError(f"{path} is not a GGUF file")
    version = _read(f, "<I")
    if version < 2:
        raise GGUFError(f"GGUF v{version} is too old to read")
    n_tensors = _read(f, "<Q")
    n_kv = _read(f, "<Q")
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        key = _read_string(f)
        meta[key] = _read_value(f, _read(f, "<I"))
    return meta, n_tensors


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Return the header key/value metadata of a GGUF file."""

    with open(path, "rb") as f:
        return _read_header(f, path)[0]


def tensor_sizes(path: str | Path) -> dict[str, int]:
    """Bytes per tensor, from the tensor-info table that follows the metadata.

    Each entry is (name, n_dims, dims…, type, offset); a tensor's size is the
    distance to the next tensor's offset. Used to see exactly how much of an
    MoE model stays in system RAM with ``--n-cpu-moe``.
    """

    path = Path(path)
    with open(path, "rb") as f:
        meta, n_tensors = _read_header(f, path)
        infos: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read(f, "<I")
            f.seek(8 * n_dims, 1)  # dims
            _read(f, "<I")         # ggml type
            infos.append((name, _read(f, "<Q")))
        alignment = int(meta.get("general.alignment", 32) or 32)
        data_start = -(-f.tell() // alignment) * alignment
    data_size = path.stat().st_size - data_start
    ordered = sorted(infos, key=lambda t: t[1])
    sizes: dict[str, int] = {}
    for i, (name, offset) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else data_size
        sizes[name] = max(0, end - offset)
    return sizes


def expert_bytes_on_cpu(path: str | Path, n_cpu_moe: int) -> int:
    """Bytes of routed-expert weights llama.cpp keeps on the CPU for ``--n-cpu-moe N``.

    In a mixture-of-experts model most parameters live in per-layer "expert"
    FFN tensors (``blk.<i>.ffn_*_exps``), but each token only uses a few
    experts. Keeping the experts of the first N layers in system RAM frees most
    of the VRAM while the attention layers (used by every token) stay on the GPU.
    """

    import re

    pattern = re.compile(r"blk\.(\d+)\.ffn_\w*_exps\b")
    total = 0
    for name, size in tensor_sizes(path).items():
        m = pattern.match(name)
        if m and int(m.group(1)) < n_cpu_moe:
            total += size
    return total


@dataclass
class ModelShape:
    architecture: str
    n_layers: int
    n_attention_layers: int       # layers that actually keep a KV cache
    kv_bytes_per_token: int       # f16 K+V across all attention layers
    trained_ctx: int | None

    def kv_cache_bytes(self, ctx: int) -> int:
        return self.kv_bytes_per_token * ctx


def model_shape(path: str | Path) -> ModelShape:
    """Work out how much KV cache one token of context costs for this model."""

    meta = read_metadata(path)
    arch = str(meta.get("general.architecture", "unknown"))

    def get(name: str, default: Any = None) -> Any:
        return meta.get(f"{arch}.{name}", default)

    n_layers = int(get("block_count", 0) or 0)
    n_head = get("attention.head_count", 0) or 0
    n_head_kv = get("attention.head_count_kv", n_head)
    n_embd = int(get("embedding_length", 0) or 0)

    def per_layer(value: Any) -> list[int]:
        # Some hybrid models (e.g. LFM2) store one value per layer, with 0 for
        # layers that are convolutions rather than attention.
        if isinstance(value, list):
            return [int(v) for v in value]
        return [int(value or 0)] * n_layers

    heads = per_layer(n_head)
    kv_heads = per_layer(n_head_kv)
    first_heads = next((h for h in heads if h), 1)
    key_len = int(get("attention.key_length", 0) or (n_embd // first_heads if n_embd else 0))
    value_len = int(get("attention.value_length", 0) or key_len)

    # Hybrid linear-attention models (e.g. Qwen3.5's Gated DeltaNet layers) keep
    # a KV cache only on every Nth layer; the others keep a small fixed state.
    interval = int(get("full_attention_interval", 0) or 0)
    attn_layers = [i for i, h in enumerate(kv_heads) if h > 0]
    if interval > 1:
        attn_layers = [i for i in attn_layers if (i + 1) % interval == 0]

    f16 = 2
    kv_per_token = sum(kv_heads[i] * (key_len + value_len) * f16 for i in attn_layers)
    trained = get("context_length")
    return ModelShape(
        architecture=arch,
        n_layers=n_layers,
        n_attention_layers=len(attn_layers),
        kv_bytes_per_token=kv_per_token,
        trained_ctx=int(trained) if trained else None,
    )
