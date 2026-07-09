#!/usr/bin/env python3
"""Turn per-role examples into chat-format SFT data (JSONL).

Liquid explicitly recommends fine-tuning LFM2 on narrow use cases. The two most
worthwhile targets in this system are the glue roles whose failures are cheap to
label:

* **extractor** — (raw tool output) -> (structured JSON). Label a few dozen real
  tracebacks/test logs with the JSON you wish the model had produced.
* **dispatcher** — (intent) -> (exact tool call JSON).

Input: a JSONL file of ``{"input": "...", "output": "..."}`` pairs (plus an
optional shared ``system`` prompt). Output: chat-format JSONL ready for TRL /
Unsloth ``SFTTrainer`` with a ``messages`` field.

    python finetune/prepare_data.py --role extractor \
        --in data/extractor_pairs.jsonl --out data/extractor_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse the exact production system prompts so training matches inference.
from coding_agent.roles import (  # noqa: E402
    DISPATCHER_SYSTEM,
    EXTRACTOR_SYSTEM,
)

ROLE_SYSTEM = {
    "extractor": EXTRACTOR_SYSTEM,
    "dispatcher": DISPATCHER_SYSTEM,
}


def to_chat(pair: dict, system: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["output"]},
        ]
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=sorted(ROLE_SYSTEM))
    p.add_argument("--in", dest="infile", required=True, help="JSONL of {input, output} pairs")
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--system", default=None, help="override the role's default system prompt")
    args = p.parse_args(argv)

    system = args.system or ROLE_SYSTEM[args.role]
    src = Path(args.infile).read_text().splitlines()
    n = 0
    with Path(args.outfile).open("w") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            out.write(json.dumps(to_chat(pair, system)) + "\n")
            n += 1
    print(f"wrote {n} chat examples to {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
