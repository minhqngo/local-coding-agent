"""Crash-safe session persistence and resume.

The multi-model loop writes its full state (task, transcript, observations,
coder memory, passing-test flag) to ``<repo>/.coding-agent/sessions/<id>.json``
after every turn, so a crash, Ctrl-C or a dead model server can be resumed with
``--resume latest|<id>``. Writes are atomic (temp file + ``os.replace``) so a
kill mid-write never corrupts the file.

Next to each session lives its trace, ``<id>.trace.jsonl`` (see
:mod:`coding_agent.trace`). Together they are also the raw material for
fine-tuning the glue models (``finetune/``): real tool output paired with what
the models made of it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

SESSIONS_SUBDIR = Path(".coding-agent") / "sessions"
TRACE_SUFFIX = ".trace.jsonl"


class SessionStore:
    def __init__(self, workdir: str | Path):
        self.dir = Path(workdir).resolve() / SESSIONS_SUBDIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def trace_path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}{TRACE_SUFFIX}"

    def save(self, session_id: str, data: dict) -> None:
        payload = dict(data)
        payload["session_id"] = session_id
        tmp = self._path(session_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path(session_id))  # atomic

    def load(self, session_id: str) -> dict:
        return json.loads(self._path(session_id).read_text(encoding="utf-8"))

    def latest(self) -> dict | None:
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None

    def resolve(self, ref: str) -> dict | None:
        """Resolve 'latest' or a specific session id to its stored data."""

        if ref == "latest":
            return self.latest()
        path = self._path(ref)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def find_trace(self, ref: str = "latest") -> Path | None:
        if ref != "latest":
            path = self.trace_path(ref)
            return path if path.exists() else None
        traces = sorted(self.dir.glob(f"*{TRACE_SUFFIX}"), key=lambda p: p.stat().st_mtime)
        return traces[-1] if traces else None
