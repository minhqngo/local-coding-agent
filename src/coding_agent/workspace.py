"""Repo snapshot for the planner's first prompt (P5).

A small, one-time picture of the workspace — git branch/status/recent commits
plus the head of a couple of project files — so the planner doesn't burn turns
(and extractor hops) rediscovering the repo. Everything is best-effort with
short timeouts and silent fallbacks; a non-git directory yields a benign block.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_TIMEOUT_S = 5
_STATUS_CLIP = 1500
_FILE_HEAD_CHARS = 800
_DOC_FILES = ["AGENTS.md", "README.md", "pyproject.toml", "package.json"]


def _git(workdir: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[clipped]"


def workspace_snapshot(workdir: str | Path) -> str:
    workdir = Path(workdir).resolve()
    lines: list[str] = [f"cwd: {workdir}"]

    root = _git(workdir, "rev-parse", "--show-toplevel")
    if root:
        branch = _git(workdir, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git(workdir, "status", "--short")
        log = _git(workdir, "log", "--oneline", "-5")
        lines.append(f"git root: {root}")
        if branch:
            lines.append(f"branch: {branch}")
        if status:
            lines.append("git status --short:\n" + _clip(status, _STATUS_CLIP))
        if log:
            lines.append("recent commits:\n" + log)
    else:
        lines.append("(not a git repository)")

    for name in _DOC_FILES:
        path = workdir / name
        if path.is_file():
            try:
                head = path.read_text(errors="replace")[:_FILE_HEAD_CHARS]
            except OSError:
                continue
            lines.append(f"--- {name} (head) ---\n{head}")

    return "\n".join(lines)
