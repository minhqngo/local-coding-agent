"""Workspace snapshot: what the model knows about the repo before turn 1.

A one-time picture, put at the top of the first prompt, so the agent doesn't
burn turns rediscovering basics. Each part is there because a run once failed
without it:

* **environment** — OS, shell, and *which Python* commands will use. Without
  it, a model on Windows emits bash (``./venv/bin/activate``), and nobody
  notices the tests are running under the wrong interpreter.
* **test command** — the exact command the finish gate will run.
* **file listing** — so a task that names ``foo.py`` in a repo without one is
  obviously impossible at turn 1, not turn 17.
* **git status / recent commits** and the head of README/pyproject.

Everything is best-effort with short timeouts; a non-git directory yields a
benign block.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_GIT_TIMEOUT_S = 5
_STATUS_CLIP = 1500
_FILE_HEAD_CHARS = 800
_DOC_FILES = ["AGENTS.md", "README.md", "pyproject.toml", "package.json"]
_TREE_MAX_ENTRIES = 60
_TREE_MAX_DEPTH = 3
_ALWAYS_SKIP = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".coding-agent",
    ".index", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "dist", "build",
}


def _git(workdir: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[clipped]"


def file_listing(workdir: Path, skip: set[str] | None = None) -> list[str]:
    """Relative paths (dirs end in '/'), breadth-limited, junk dirs skipped."""

    skip = _ALWAYS_SKIP | (skip or set())
    entries: list[str] = []
    for root, dirs, files in os.walk(workdir):
        rel_root = Path(root).relative_to(workdir)
        depth = len(rel_root.parts)
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.endswith(".egg-info"))
        if depth >= _TREE_MAX_DEPTH:
            dirs[:] = []
        for name in sorted(files):
            entries.append(str((rel_root / name).as_posix()))
        for name in dirs:
            entries.append(str((rel_root / name).as_posix()) + "/")
        if len(entries) > _TREE_MAX_ENTRIES:
            break
    entries.sort()
    if len(entries) > _TREE_MAX_ENTRIES:
        extra = len(entries) - _TREE_MAX_ENTRIES
        entries = entries[:_TREE_MAX_ENTRIES] + [f"... (+{extra} more)"]
    return entries


def workspace_snapshot(
    workdir: str | Path,
    *,
    environment: str = "",
    test_command: str = "",
    skip_dirs: set[str] | None = None,
) -> str:
    workdir = Path(workdir).resolve()
    lines: list[str] = [f"cwd: {workdir}"]
    if environment:
        lines.append(environment)
    if test_command:
        lines.append(f"test command (the finish gate runs exactly this): {test_command}")

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

    listing = file_listing(workdir, skip_dirs)
    lines.append("files:\n" + ("\n".join(f"  {e}" for e in listing) if listing else "  (empty directory)"))

    for name in _DOC_FILES:
        path = workdir / name
        if path.is_file():
            try:
                head = path.read_text(encoding="utf-8", errors="replace")[:_FILE_HEAD_CHARS]
            except OSError:
                continue
            lines.append(f"--- {name} (head) ---\n{head}")

    return "\n".join(lines)
