"""Run traces: a JSONL record of everything that happened in a run.

If you are learning how agents work, this is the most useful file to read the
*output* of. Every model call (the exact messages sent, the raw reply, any
hidden reasoning, latency and token counts) and every tool execution (the call,
exit code, output) is appended as one JSON line to::

    <repo>/.coding-agent/sessions/<session-id>.trace.jsonl

and ``coding-agent trace latest`` renders it as a readable timeline. When a run
goes wrong, the answer is almost always visible in what a model was *shown*.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Tracer:
    """Append-only event log. ``path=None`` keeps events in memory only."""

    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **fields: Any) -> None:
        event = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **fields}
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    # Convenience wrappers so call sites read like what they record.
    def model_call(self, **fields: Any) -> None:
        self.emit("model", **fields)

    def tool_call(self, **fields: Any) -> None:
        self.emit("tool", **fields)

    def note(self, text: str, **fields: Any) -> None:
        self.emit("note", text=text, **fields)

    def summary(self) -> dict[str, Any]:
        """Aggregate stats: calls/tokens/seconds per role, tool count."""

        calls: Counter = Counter()
        seconds: Counter = Counter()
        prompt_tok: Counter = Counter()
        completion_tok: Counter = Counter()
        tools = 0
        for e in self.events:
            if e["kind"] == "model":
                role = e.get("role", "?")
                calls[role] += 1
                seconds[role] += e.get("latency_s") or 0.0
                usage = e.get("usage") or {}
                prompt_tok[role] += usage.get("prompt_tokens") or 0
                completion_tok[role] += usage.get("completion_tokens") or 0
            elif e["kind"] == "tool":
                tools += 1
        return {
            "model_calls": sum(calls.values()),
            "model_seconds": round(sum(seconds.values()), 1),
            "prompt_tokens": sum(prompt_tok.values()),
            "completion_tokens": sum(completion_tok.values()),
            "tool_calls": tools,
            "per_role": {
                r: {
                    "calls": calls[r],
                    "seconds": round(seconds[r], 1),
                    "prompt_tokens": prompt_tok[r],
                    "completion_tokens": completion_tok[r],
                }
                for r in sorted(calls)
            },
        }


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _clip(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f" …[+{len(text) - limit} chars]"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines()) or prefix


def render_trace(events: list[dict[str, Any]], full: bool = False) -> str:
    """Human-readable timeline. ``full=True`` also prints every prompt."""

    limit = 0 if full else 400
    out: list[str] = []
    for e in events:
        t = f"[{e.get('t', 0):7.1f}s]"
        kind = e.get("kind")
        if kind == "model":
            usage = e.get("usage") or {}
            tok = ""
            if usage:
                tok = f", {usage.get('prompt_tokens', '?')}→{usage.get('completion_tokens', '?')} tok"
            head = f"{t} MODEL {e.get('role')} ({e.get('latency_s', 0):.1f}s{tok})"
            if e.get("error"):
                head += f"  ERROR: {e['error']}"
            out.append(head)
            if full:
                for m in e.get("messages", []):
                    out.append(f"  ┌─ {m.get('role')} ─")
                    out.append(_indent(m.get("content", "")))
            if e.get("reasoning"):
                out.append("  ┌─ reasoning ─")
                out.append(_indent(_clip(e["reasoning"], limit)))
            out.append("  ┌─ reply ─")
            out.append(_indent(_clip(e.get("response", ""), limit)))
        elif kind == "tool":
            status = e.get("status", "")
            out.append(f"{t} TOOL  {e.get('name')} {_clip(e.get('summary', ''), 160)}  → {status}")
            if full or not e.get("ok", True):
                out.append(_indent(_clip(e.get("output", ""), limit)))
        elif kind == "note":
            out.append(f"{t} NOTE  {e.get('text')}")
        elif kind == "run":
            out.append(f"{t} RUN   {e.get('event')}: {_clip(str(e.get('detail', '')), 300)}")
        else:
            out.append(f"{t} {kind}: {e}")
    return "\n".join(out)
