"""Crash-safe session persistence and resume (P6).

The loop writes its full state (task, transcript, observations, raw context,
passing-test flag) to ``<repo>/.coding-agent/sessions/<id>.json`` after every
turn, so a crash or Ctrl-C can be resumed with ``--resume latest|<id>``. Writes
are atomic (temp file + ``os.replace``) so a kill mid-write never corrupts the
file.

Beyond recovery, these transcripts are exactly the labelled data the Phase-5
fine-tuning workflow wants (raw tool output → extracted JSON pairs); today they
would otherwise die with the process.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

SESSIONS_SUBDIR = Path(".coding-agent") / "sessions"


class SessionStore:
    def __init__(self, workdir: str | Path):
        self.dir = Path(workdir).resolve() / SESSIONS_SUBDIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session_id: str, data: dict) -> None:
        payload = dict(data)
        payload["session_id"] = session_id
        tmp = self._path(session_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self._path(session_id))  # atomic

    def load(self, session_id: str) -> dict:
        return json.loads(self._path(session_id).read_text())

    def latest(self) -> dict | None:
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return json.loads(files[-1].read_text()) if files else None

    def resolve(self, ref: str) -> dict | None:
        """Resolve 'latest' or a specific session id to its stored data."""

        if ref == "latest":
            return self.latest()
        path = self._path(ref)
        return json.loads(path.read_text()) if path.exists() else None
