"""Command execution sandbox — where the agent's shell commands actually run.

Two backends:

* ``subprocess`` — runs commands on this machine with the working directory
  pinned to the repo. Fast; the denylist is **best-effort only** (see below).
  Fine for a personal agent on code you trust.
* ``docker`` — runs commands inside one long-lived, ``--network none`` container
  per run, with the repo bind-mounted at ``/workspace``. Use an image with your
  test deps baked in (``scripts/Dockerfile.sandbox``); the container is removed
  when the sandbox closes.

Things that look trivial but break real runs (each one has bitten this project):

1. **Whose Python?** The agent itself usually runs inside *its* virtualenv. A
   child process inherits that environment, so ``python -m pytest`` would run
   the *agent's* interpreter — which doesn't have the project's packages. We
   strip the agent's venv from ``PATH`` and, if the project has its own
   ``.venv/`` or ``venv/``, put that first instead.
2. **Encoding.** On Windows the default text encoding is the locale code page
   (cp1252), so non-ASCII test output can crash the decoder. We force UTF-8 for
   child Pythons and decode with ``errors="replace"``.
3. **Timeouts must kill the whole process tree.** ``subprocess.run(timeout=…)``
   kills only the shell; a hung ``pytest`` grandchild keeps the output pipe
   open and the agent hangs forever (on Windows especially). We start each
   command in its own process group and kill the group.
4. **No stdin.** A command that waits for input (a prompt, a REPL) gets EOF
   immediately instead of hanging until the timeout. On Windows that takes a
   closed pipe: ``DEVNULL`` claims to be a terminal, and a bare ``python``
   then sits at an interactive prompt.
5. **No stale bytecode.** Two edits within one second that keep a file's size
   (``return 2`` → ``return 3``) fool Python's ``.pyc`` cache, so the tests run
   the old code. Child Pythons get ``PYTHONDONTWRITEBYTECODE=1``.

The denylist is a defence-in-depth convenience, not a security boundary: it
lexes each command with ``shlex`` and blocks argv-prefix matches plus obviously
destructive ``rm``/fork-bomb forms. It cannot catch obfuscated commands — that
is what the docker backend is for.
"""

from __future__ import annotations

import atexit
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig

MAX_OUTPUT_CHARS = 12000
# Command separators we split on to inspect each sub-command independently.
_SEP_RE = re.compile(r"&&|\|\||[;|&]")
_VENV_NAMES = (".venv", "venv")


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


# --------------------------------------------------------------------------- #
# Denylist
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Environment for child processes
# --------------------------------------------------------------------------- #

def _venv_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def find_project_venv(workdir: Path) -> Path | None:
    for name in _VENV_NAMES:
        candidate = workdir / name
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    return None


def build_command_env(workdir: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every sandboxed command runs with (see module docstring)."""

    env = dict(os.environ if base is None else base)
    project_venv = find_project_venv(workdir)

    # 1. Demote the agent's own virtualenv to the END of PATH: a system Python
    #    wins over it, but it still works as a last resort when it is the only
    #    Python on the machine (common with uv-managed interpreters).
    agent_venvs = set()
    if sys.prefix != sys.base_prefix:
        agent_venvs.add(Path(sys.prefix))
    if env.get("VIRTUAL_ENV"):
        agent_venvs.add(Path(env["VIRTUAL_ENV"]))
    demote = {
        _norm(_venv_bin(v)) for v in agent_venvs
        if project_venv is None or _norm(v) != _norm(project_venv)
    }
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    path_parts = [p for p in parts if _norm(p) not in demote]
    demoted = [p for p in parts if _norm(p) in demote]
    # ...but ahead of Windows' WindowsApps folder, whose python.exe is only a
    # Microsoft Store installer stub, not an interpreter.
    stub = next((i for i, p in enumerate(path_parts) if "windowsapps" in p.lower()), len(path_parts))
    path_parts[stub:stub] = demoted
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)

    # 2. Activate the project's own virtualenv, if it has one.
    if project_venv is not None:
        path_parts.insert(0, str(_venv_bin(project_venv)))
        env["VIRTUAL_ENV"] = str(project_venv)
    env["PATH"] = os.pathsep.join(path_parts)

    # 3. UTF-8 for any Python we start, regardless of the OS code page.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # 4. No .pyc files. Python trusts a cached .pyc when the source's mtime (in
    #    whole seconds) and size match. An agent edits faster than once a second,
    #    and `return 2` → `return 3` keeps the size, so tests would silently run
    #    the OLD code. (Found by this repo's own test for test-after-edit.)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _read_capped(path: Path, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Read a (possibly huge) output file, keeping only its head and tail."""

    size = path.stat().st_size
    with open(path, "rb") as f:
        if size <= limit * 2:
            data = f.read()
        else:
            head = f.read(limit)
            f.seek(-limit, os.SEEK_END)
            data = head + b"\n...[output truncated]...\n" + f.read()
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def run_command(command: str, cwd: Path, env: dict[str, str] | None, timeout_s: int) -> CommandResult:
    """Run a shell command with merged stdout/stderr, UTF-8, and a tree-killing timeout.

    Output goes to a temporary FILE rather than a pipe. With a pipe, "done"
    means "every process holding the pipe has exited" — so a background
    grandchild (a test that starts a server) makes a finished command look
    hung. With a file, done means the shell exited.
    """

    import tempfile

    # stdin is a pipe we close at once: immediate EOF. (DEVNULL is not enough on
    # Windows: NUL reports itself as a terminal, so a bare `python` starts an
    # interactive prompt and hangs until the timeout.)
    kwargs: dict = dict(shell=True, cwd=str(cwd), env=env, stdin=subprocess.PIPE)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    fd, out_name = tempfile.mkstemp(prefix="agent-cmd-", suffix=".log")
    out_path = Path(out_name)
    proc = None
    try:
        with os.fdopen(fd, "wb") as out_file:
            proc = subprocess.Popen(command, stdout=out_file, stderr=subprocess.STDOUT, **kwargs)
            proc.stdin.close()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                return CommandResult(
                    command, 124,
                    _truncate(_read_capped(out_path) + f"\n[timed out after {timeout_s}s]"),
                    timed_out=True,
                )
        return CommandResult(command, proc.returncode, _truncate(_read_capped(out_path)))
    finally:
        # Ctrl-C (or any other exception) must not leave a hung test running.
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
        try:
            out_path.unlink()
        except OSError:
            pass  # Windows: a lingering grandchild may still hold the file open


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #

class Sandbox:
    def __init__(self, config: SandboxConfig, workdir: Path):
        self.config = config
        self.workdir = Path(workdir).resolve()
        if not self.workdir.exists():
            raise FileNotFoundError(f"sandbox workdir does not exist: {self.workdir}")
        self.env = build_command_env(self.workdir)
        self._container: str | None = None
        atexit.register(self.close)

    # -- what the model should know about this environment ------------------ #
    def describe(self) -> str:
        """A few lines for the model's prompt: OS, shell, which Python runs."""

        if self.config.backend == "docker":
            return (f"commands run in docker image {self.config.docker_image} "
                    "(Linux, bash, no network)")
        system = platform.system() or os.name
        if os.name == "nt":
            shell = "cmd.exe — use Windows commands (dir, type), not ls/cat"
        else:
            shell = "/bin/sh"
        lines = [f"os: {system} (shell: {shell})"]
        python = shutil.which("python", path=self.env.get("PATH"))
        venv = find_project_venv(self.workdir)
        if python is None:
            where = "(no `python` on PATH)"
        elif "windowsapps" in python.lower():
            where = (f"{python} — only the Microsoft Store installer stub, not a real "
                     "Python; give the project a .venv")
        elif venv is not None:
            where = f"{python} — the project's virtualenv {venv.name}/"
        elif sys.prefix != sys.base_prefix and _norm(python).startswith(_norm(sys.prefix)):
            where = (f"{python} — the AGENT's own virtualenv (the project has no .venv/; "
                     "its test dependencies may be missing)")
        else:
            where = python
        lines.append(f"python: {where}")
        return "\n".join(lines)

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
            if os.name == "nt":  # cmd.exe doesn't care about case: DEL == del
                cmd0 = cmd0.lower()
            tail = " ".join(argv[1:])
            for entry in deny:
                ename = os.path.basename(entry[0])
                if os.name == "nt":
                    ename = ename.lower()
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
        return run_command(command, self.workdir, self.env, timeout_s)

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
        except (FileNotFoundError, subprocess.TimeoutExpired):
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
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, stdin=subprocess.DEVNULL,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return CommandResult(command, proc.returncode, _truncate(output))
        except subprocess.TimeoutExpired:
            # The exec'd process keeps running inside the container; kill the
            # whole container so nothing leaks past the timeout.
            self.close()
            return CommandResult(
                command, 124, _truncate(f"[docker command timed out after {timeout_s}s]"),
                timed_out=True,
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
    """Keep head AND tail: pytest prints its verdict ("1 failed") at the end."""

    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...[truncated {len(text) - limit} chars]...\n{tail}"
