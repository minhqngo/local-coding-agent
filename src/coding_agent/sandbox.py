"""Command execution sandbox.

Two backends:

* ``subprocess`` — runs commands locally with the working directory pinned to
  the repo root. Fast; the denylist is **best-effort only** (see below) — real
  isolation comes from the docker backend. Fine for a personal agent on code
  you trust.
* ``docker`` — runs commands inside a single long-lived, ``--network none``
  container per run, with the repo bind-mounted at ``/workspace``. State
  persists between commands (so ``pip install`` then ``pytest`` works, given a
  base image that can reach a package index), and the container is force-removed
  when the sandbox closes or the process exits. Use a base image with your test
  deps baked in (see ``scripts/Dockerfile``); ``--network none`` blocks installs
  at run time by design.

The denylist is a defence-in-depth convenience, not a security boundary: it
lexes each command with ``shlex`` and blocks argv-prefix matches plus obviously
destructive ``rm``/fork-bomb forms, which avoids both the substring false
positives (``grep -r shutdown``) and the crudest bypasses (``rm -rf ~``). It
cannot catch obfuscated commands — that is what the docker backend is for.
"""

from __future__ import annotations

import atexit
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig

MAX_OUTPUT_CHARS = 12000
# Command separators we split on to inspect each sub-command independently.
_SEP_RE = re.compile(r"&&|\|\||[;|&]")


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str
    timed_out: bool = False
    refused: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.refused


def _lex(part: str) -> list[str]:
    try:
        return shlex.split(part)
    except ValueError:
        return part.split()


def _is_dangerous_rm(argv: list[str]) -> bool:
    if not argv or argv[0] != "rm":
        return False
    flags = "".join(a[1:] for a in argv if a.startswith("-") and not a.startswith("--"))
    recursive = "r" in flags or "R" in flags or "--recursive" in argv
    force = "f" in flags or "--force" in argv
    if not (recursive or force):
        return False
    targets = [a for a in argv[1:] if not a.startswith("-")]
    dangerous = {"/", "//", "~", "~/", ".", "..", "*", "/*"}
    return any(t in dangerous or t.startswith("/") or t.startswith("~") for t in targets)


class Sandbox:
    def __init__(self, config: SandboxConfig, workdir: Path):
        self.config = config
        self.workdir = Path(workdir).resolve()
        if not self.workdir.exists():
            raise FileNotFoundError(f"sandbox workdir does not exist: {self.workdir}")
        self._container: str | None = None
        atexit.register(self.close)

    # -- safety -------------------------------------------------------------- #
    def _denied(self, command: str) -> str | None:
        """Return a reason string if the command should be refused, else None.

        Matches on the command NAME (argv[0], basename, so ``mkfs.ext4`` matches
        a ``mkfs`` entry) plus any extra tokens in the entry appearing in the
        argument tail. This avoids substring false positives (``grep -r
        shutdown`` — argv[0] is grep) while still catching the real targets.
        """

        deny = [_lex(entry) for entry in self.config.denylist]
        deny = [d for d in deny if d]
        for part in _SEP_RE.split(command):
            if ":(){" in part.replace(" ", ""):
                return "fork bomb"
            argv = _lex(part)
            if not argv:
                continue
            if _is_dangerous_rm(argv):
                return "destructive rm on a root/home/broad target"
            cmd0 = os.path.basename(argv[0])
            tail = " ".join(argv[1:])
            for entry in deny:
                ename = os.path.basename(entry[0])
                if cmd0 != ename and not cmd0.startswith(ename + "."):
                    continue
                extras = [tok.rstrip("=") for tok in entry[1:]]
                if all(x in tail for x in extras):
                    return f"matched denylist entry {' '.join(entry)!r}"
        return None

    # -- execution ----------------------------------------------------------- #
    def run(self, command: str, timeout_s: int | None = None) -> CommandResult:
        timeout_s = timeout_s or self.config.command_timeout_s
        reason = self._denied(command)
        if reason is not None:
            return CommandResult(
                command=command,
                exit_code=126,
                output=f"REFUSED ({reason}): {command}",
                refused=True,
            )
        if self.config.backend == "docker":
            return self._run_docker(command, timeout_s)
        return self._run_subprocess(command, timeout_s)

    def _run_subprocess(self, command: str, timeout_s: int) -> CommandResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return CommandResult(command, proc.returncode, _truncate(output))
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            return CommandResult(
                command, 124, _truncate(partial + f"\n[timed out after {timeout_s}s]"), timed_out=True
            )

    # -- docker: one long-lived container per run ---------------------------- #
    def _ensure_container(self) -> str | None:
        if self._container:
            return self._container
        name = f"coding-agent-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "-d", "--name", name,
                    "--network", "none",
                    "-v", f"{self.workdir}:/workspace",
                    "-w", "/workspace",
                    self.config.docker_image,
                    "sleep", "infinity",
                ],
                capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError:
            return None
        if proc.returncode != 0:
            return None
        self._container = name
        return name

    def _run_docker(self, command: str, timeout_s: int) -> CommandResult:
        name = self._ensure_container()
        if name is None:
            return CommandResult(
                command, 127,
                "docker unavailable or container failed to start; "
                "set sandbox.backend: subprocess or check Docker.",
                refused=True,
            )
        try:
            proc = subprocess.run(
                ["docker", "exec", name, "bash", "-lc", command],
                capture_output=True, text=True, timeout=timeout_s,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return CommandResult(command, proc.returncode, _truncate(output))
        except subprocess.TimeoutExpired:
            # The exec'd process keeps running inside the container; kill the
            # whole container so nothing leaks past the timeout.
            self.close()
            return CommandResult(
                command, 124, _truncate(f"[docker command timed out after {timeout_s}s]"), timed_out=True
            )

    def close(self) -> None:
        if self._container:
            subprocess.run(
                ["docker", "rm", "-f", self._container],
                capture_output=True, text=True,
            )
            self._container = None

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...[truncated {len(text) - limit} chars]...\n{tail}"
