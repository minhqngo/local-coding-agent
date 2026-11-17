"""The agent-computer interface: the small, fixed set of tools the models drive.

Kept deliberately tiny (read / write / patch / shell / search) — a narrow
interface is what small models handle reliably. Every tool works relative to
the repo directory and returns plain text.

File I/O details that matter more than they look:

* **Encoding** — always UTF-8. Python's default on Windows is the locale code
  page, so ``Path.write_text("→")`` crashes there.
* **Line endings** — models always see ``\\n``. When a file on disk uses
  ``\\r\\n`` we convert back on write, so a one-line patch doesn't silently turn
  every line of a Windows file into a diff.
* **Path safety** — every path is resolved under the repo; ``..`` and absolute
  paths that escape it are refused.
* **Edits that break Python syntax are rejected** (see :func:`check_syntax`),
  and a patch whose ``old_text`` isn't found shows the current file content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .sandbox import Sandbox

MAX_READ_CHARS = 16000
PATCH_ERROR_SHOW_CHARS = 3000


@dataclass
class ToolContext:
    workdir: Path
    sandbox: Sandbox
    # Set when retrieval is enabled (--enable-retrieval); may be None.
    search_fn: Callable[[str], str] | None = None


class ToolError(Exception):
    pass


def _resolve_in_workdir(workdir: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``workdir`` and refuse escapes via ``..``/abs paths."""

    workdir = Path(workdir).resolve()
    target = (workdir / rel).resolve()
    if target != workdir and workdir not in target.parents:
        raise ToolError(f"path escapes workdir: {rel!r}")
    return target


_BOM = b"\xef\xbb\xbf"


def has_bom(path: Path) -> bool:
    """UTF-8 byte-order mark, as Notepad / Visual Studio often write it."""

    with open(path, "rb") as f:
        return f.read(3) == _BOM


def read_text(path: Path, strict: bool = False) -> tuple[str, str]:
    """Return ``(text_with_LF_newlines, original_newline)``; any BOM is stripped."""

    data = path.read_bytes()
    if data.startswith(_BOM):
        data = data[len(_BOM):]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if strict:
            raise ToolError(f"{path.name} is not UTF-8 text; refusing to edit it") from None
        text = data.decode("utf-8", errors="replace")
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), newline


def write_text(path: Path, text: str, newline: str = "\n", bom: bool = False) -> None:
    text = text.replace("\r\n", "\n")
    if newline != "\n":
        text = text.replace("\n", newline)
    path.write_bytes((_BOM if bom else b"") + text.encode("utf-8"))


def read_file(ctx: ToolContext, path: str) -> str:
    target = _resolve_in_workdir(ctx.workdir, path)
    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"DIRECTORY {path}:\n" + "\n".join(entries)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    text, _ = read_text(target)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return f"FILE {path}:\n{text}"


def check_syntax(path: str, text: str) -> None:
    """Refuse an edit that would leave a Python file unparseable.

    The lint-on-edit idea from SWE-agent: a small model that breaks a file's
    syntax tends to keep patching the wreckage. Rejecting the edit (the file
    stays as it was) and saying exactly where the error is works much better.
    """

    if not path.endswith(".py"):
        return
    try:
        compile(text, path, "exec", dont_inherit=True)
    except SyntaxError as exc:
        line = (exc.text or "").rstrip()
        where = f"line {exc.lineno}" + (f": {line.strip()}" if line else "")
        raise ToolError(
            f"edit rejected — {path} would not be valid Python ({exc.msg}, {where}). "
            "The file was NOT changed. Put each statement on its own line with "
            "correct indentation."
        ) from None


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = _resolve_in_workdir(ctx.workdir, path)
    if target.is_dir():
        raise ToolError(f"{path} is a directory")
    check_syntax(path, content)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    newline = read_text(target)[1] if existed else "\n"
    write_text(target, content, newline, bom=existed and has_bom(target))
    verb = "Updated" if existed else "Created"
    return f"{verb} {path} ({len(content)} chars, {content.count(chr(10)) + 1} lines)."


def patch_file(ctx: ToolContext, path: str, old_text: str, new_text: str) -> str:
    """Anchored edit: replace ``old_text`` (which must occur exactly once)."""

    target = _resolve_in_workdir(ctx.workdir, path)
    if not target.is_file():
        raise ToolError(f"file not found: {path}")
    if not old_text:
        raise ToolError("old_text is empty: the reply had no <old_text>...</old_text> block. "
                        "Include <old_text> (lines copied from the file) and <new_text>, or "
                        "use write_file with a <content> block to rewrite the whole file.")
    text, newline = read_text(target, strict=True)
    old_text = old_text.replace("\r\n", "\n")
    count = text.count(old_text)
    if count > 1:
        raise ToolError(f"old_text occurs {count} times in {path}; include more "
                        "surrounding lines so it matches exactly once")
    if count == 0:
        # Small models often patch from memory. Showing the real text right in
        # the error lets the next attempt copy it, instead of guessing again.
        if len(text) <= PATCH_ERROR_SHOW_CHARS:
            shown = f"\nCurrent content of {path}:\n{text}"
        else:
            shown = f" Read {path} again; it is too long to show here."
        raise ToolError(f"old_text was not found in {path} (found 0 times); copy it "
                        f"exactly, including indentation.{shown}")
    updated = text.replace(old_text, new_text.replace("\r\n", "\n"), 1)
    check_syntax(path, updated)
    write_text(target, updated, newline, bom=has_bom(target))
    return f"Patched {path} (replaced 1 occurrence, {len(new_text)} chars new)."


def run_shell(ctx: ToolContext, command: str) -> str:
    result = ctx.sandbox.run(command)
    status = "OK" if result.ok else f"EXIT {result.exit_code}"
    return f"$ {command}\n[{status}]\n{result.output}"


def search_codebase(ctx: ToolContext, query: str) -> str:
    if ctx.search_fn is None:
        return (
            "search_codebase is unavailable: the retrieval index is not built "
            "(run with --enable-retrieval to use this tool)."
        )
    return ctx.search_fn(query)


# name -> {fn, args: {arg: required?}, doc}. The dispatcher targets these names.
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
        "doc": "Search the codebase for chunks relevant to a question.",
    },
}

MUTATING_TOOLS = {"write_file", "patch_file"}


def missing_args(name: str, arguments: dict[str, Any]) -> list[str]:
    spec = TOOLS.get(name)
    if spec is None:
        return []
    return [a for a, required in spec["args"].items() if required and a not in arguments]


def execute_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}; known tools: {sorted(TOOLS)}")
    missing = missing_args(name, arguments)
    if missing:
        raise ToolError(f"tool {name} missing required arguments: {missing}")
    kwargs = {a: as_text(a, arguments[a]) for a in spec["args"] if a in arguments}
    return spec["fn"](ctx, **kwargs)


def as_text(name: str, value: Any) -> str:
    """Small models sometimes send a list of lines or a number; coerce sanely."""

    if isinstance(value, str):
        return value
    if value is None:
        raise ToolError(f"argument {name!r} is null")
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    return str(value)
