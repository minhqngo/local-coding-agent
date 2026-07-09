#!/usr/bin/env python3
"""Tiny helper so the shell scripts share one source of truth (models.yaml).

Usage:
  python3 scripts/_config.py roles                 # role names, one per line
  python3 scripts/_config.py field <role> <field>  # e.g. port, gguf, hf_repo
  python3 scripts/_config.py serve-lines           # "role port gguf draft_gguf"
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _roles() -> dict:
    data = yaml.safe_load(CONFIG.read_text()) or {}
    return data.get("roles", {}) or {}


def main(argv: list[str]) -> int:
    roles = _roles()
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "roles":
        print("\n".join(roles))
    elif cmd == "field" and len(argv) == 3:
        role, field = argv[1], argv[2]
        print(roles.get(role, {}).get(field, ""))
    elif cmd == "serve-lines":
        for name, spec in roles.items():
            print(f"{name}\t{spec.get('port','')}\t{spec.get('gguf','')}\t{spec.get('draft_gguf','')}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
