"""The agent-computer interface: the small, fixed set of tools the models drive.

Kept deliberately tiny (read / write / shell / search) — a narrow interface is
what small models handle reliably. Every tool operates relative to the repo
workdir and returns a plain-text result string; the extractor is responsible for
compressing that text before it reaches the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .sandbox import Sandbox

MAX_READ_CHARS = 16000


@dataclass
class ToolContext:
    workdir: Path
    sandbox: Sandbox
    # Set by the loop when Phase-4 retrieval is enabled; may be None.
    search_fn: Callable[[str], str] | None = None


class ToolError(Exception):
    pass


def _resolve_in_workdir(workdir: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``workdir`` and refuse escapes via ``..``/abs paths."""

    target = (workdir / rel).resolve()
    if target != workdir and workdir not in target.parents:
        raise ToolError(f"path escapes workdir: {rel!r}")
    return target


def read_file(ctx: ToolContext, path: str) -> str:
    target = _resolve_in_workdir(ctx.workdir, path)
    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"DIRECTORY {path}:\n" + "\n".join(entries)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    text = target.read_text(errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return f"FILE {path}:\n{text}"


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = _resolve_in_workdir(ctx.workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content)
    verb = "Updated" if existed else "Created"
    return f"{verb} {path} ({len(content)} chars, {content.count(chr(10)) + 1} lines)."


def patch_file(ctx: ToolContext, path: str, old_text: str, new_text: str) -> str:
    """Anchored edit: replace ``old_text`` (which must occur exactly once)."""

    target = _resolve_in_workdir(ctx.workdir, path)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    text = target.read_text(errors="replace")
    count = text.count(old_text)
    if count != 1:
        raise ToolError(
            f"old_text must occur exactly once in {path}, found {count}; "
            "include more surrounding context to make it unique"
        )
    target.write_text(text.replace(old_text, new_text, 1))
    return f"Patched {path} (replaced 1 occurrence, {len(new_text)} chars new)."


def run_shell(ctx: ToolContext, command: str) -> str:
    result = ctx.sandbox.run(command)
    status = "OK" if result.ok else f"EXIT {result.exit_code}"
    return f"$ {command}\n[{status}]\n{result.output}"


def search_codebase(ctx: ToolContext, query: str) -> str:
    if ctx.search_fn is None:
        return (
            "search_codebase is unavailable: retrieval index not built "
            "(enable Phase-4 retrieval to use this tool)."
        )
    return ctx.search_fn(query)


# name -> (callable, {arg: required?}) — the dispatcher targets these names.
TOOLS: dict[str, dict[str, Any]] = {
    "read_file": {
        "fn": read_file,
        "args": {"path": True},
        "doc": "Read a file (or list a directory) relative to the repo root.",
    },
    "write_file": {
        "fn": write_file,
        "args": {"path": True, "content": True},
        "doc": "Write the full contents of a file relative to the repo root.",
    },
    "patch_file": {
        "fn": patch_file,
        "args": {"path": True, "old_text": True, "new_text": True},
        "doc": "Replace an exactly-once snippet in a file (surgical edit).",
    },
    "run_shell": {
        "fn": run_shell,
        "args": {"command": True},
        "doc": "Run a shell command in the sandbox (e.g. the test command).",
    },
    "search_codebase": {
        "fn": search_codebase,
        "args": {"query": True},
        "doc": "Ask a natural-language question about the codebase (RAG).",
    },
}


def execute_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}; known tools: {sorted(TOOLS)}")
    missing = [a for a, required in spec["args"].items() if required and a not in arguments]
    if missing:
        raise ToolError(f"tool {name} missing required arguments: {missing}")
    kwargs = {a: arguments[a] for a in spec["args"] if a in arguments}
    return spec["fn"](ctx, **kwargs)
