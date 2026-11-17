"""Serving the models locally with llama.cpp's ``llama-server``.

One ``llama-server`` process per *served* role (roles that ``share`` another
role's model reuse its server). Cross-platform — no bash needed on Windows:

    coding-agent serve start [ROLE ...] [--cpu] [--host 0.0.0.0] [--force]
    coding-agent serve status
    coding-agent serve stop
    coding-agent doctor [--bench]
    coding-agent download [ROLE ...]

The interesting part, educationally, is the **memory budget**. Before starting
anything we estimate what each server will need (weights + KV cache + compute
buffers, see :mod:`coding_agent.gguf`) and compare it with free VRAM. Starting
more than fits does not fail loudly: on Windows the NVIDIA driver quietly spills
into shared system RAM and generation becomes *slower than running on the CPU*.
That is exactly what happened to this project's first real run (planner at
9 tok/s; the same model does 112 tok/s when it fits).

Binary lookup order: ``$LLAMA_SERVER`` → ``server.binary`` in the profile →
``llama-server`` on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import REPO_ROOT, Config

RUNS_DIR = REPO_ROOT / "runs"
GiB = 1024 ** 3
MiB = 1024 ** 2
# Compute/scratch buffers per server, on top of weights + KV. Calibrated
# against nvidia-smi on this repo's profiles (measured 6.3 GiB for an estimated
# 6.4 GiB of weights+KV across 3 GPU servers), rounded up for safety.
COMPUTE_OVERHEAD_BYTES = 256 * MiB
# Bytes per KV element relative to f16, for --cache-type-k/-v values.
_KV_TYPE_RATIO = {"f16": 1.0, "bf16": 1.0, "f32": 2.0, "q8_0": 34 / 64, "q5_1": 24 / 64,
                  "q5_0": 22 / 64, "q4_1": 20 / 64, "q4_0": 18 / 64}
# Keep this much VRAM free for the desktop, driver and fragmentation.
VRAM_HEADROOM_BYTES = 512 * MiB
READY_TIMEOUT_S = 180
# Below this, a GPU-placed model is suspiciously slow (spilled or no GPU backend).
SLOW_GPU_TOK_S = 25.0


# --------------------------------------------------------------------------- #
# What to start
# --------------------------------------------------------------------------- #

@dataclass
class ServerSpec:
    role: str
    serves: list[str]
    port: int
    host: str
    alias: str
    model_path: Path
    draft_path: Path | None
    ctx: int | None
    gpu_layers: int | None
    extra_args: list[str] = field(default_factory=list)
    cpu_moe_layers: int | None = None

    @property
    def on_gpu(self) -> bool:
        return self.gpu_layers is None or self.gpu_layers > 0

    @property
    def client_host(self) -> str:
        return "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host

    @property
    def base(self) -> str:
        return f"http://{self.client_host}:{self.port}"


def models_dir(config: Config) -> Path:
    d = Path(config.server.models_dir)
    return d if d.is_absolute() else REPO_ROOT / d


def server_specs(
    config: Config,
    roles: list[str] | None = None,
    *,
    gpu_layers: int | None = None,
    host: str | None = None,
) -> list[ServerSpec]:
    """One spec per served role. ``roles`` may name sharing roles too."""

    wanted: set[str] | None = None
    if roles:
        wanted = set()
        for name in roles:
            role = config.role(name)
            wanted.add(role.shares or role.name)
    mdir = models_dir(config)
    specs = []
    for role in config.served_roles():
        if wanted is not None and role.name not in wanted:
            continue
        specs.append(ServerSpec(
            role=role.name,
            serves=[role.name, *config.sharers_of(role.name)],
            port=role.port,
            host=host or config.server.host,
            alias=role.model,
            model_path=mdir / role.gguf,
            draft_path=(mdir / role.draft_gguf) if role.draft_gguf else None,
            ctx=role.ctx,
            gpu_layers=gpu_layers if gpu_layers is not None else role.gpu_layers,
            extra_args=list(role.server_args),
            cpu_moe_layers=role.cpu_moe_layers,
        ))
    return specs


def find_llama_server(config: Config) -> str | None:
    for candidate in (os.environ.get("LLAMA_SERVER"), config.server.binary):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return shutil.which("llama-server")


@lru_cache(maxsize=4)
def _help_text(binary: str) -> str:
    try:
        proc = subprocess.run([binary, "--help"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
        return proc.stdout + proc.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def build_command(spec: ServerSpec, binary: str) -> list[str]:
    cmd = [
        binary,
        "--model", str(spec.model_path),
        "--host", spec.host,
        "--port", str(spec.port),
        "--alias", spec.alias,
        # The agent is strictly sequential: one slot means one KV cache, and
        # the prompt prefix cache is never evicted by a parallel request.
        "--parallel", "1",
    ]
    if spec.gpu_layers is not None:
        cmd += ["-ngl", str(spec.gpu_layers)]
    if spec.ctx:
        cmd += ["-c", str(spec.ctx)]
    if spec.cpu_moe_layers and spec.on_gpu:
        cmd += ["--n-cpu-moe", str(spec.cpu_moe_layers)]
    if spec.draft_path is not None and spec.draft_path.is_file():
        cmd += ["-md", str(spec.draft_path)]
        if spec.gpu_layers is not None:
            cmd += ["-ngld", str(spec.gpu_layers)]
        # The flag was renamed in 2026 llama.cpp builds; support both.
        help_text = _help_text(binary)
        if "--spec-draft-n-max" in help_text:
            cmd += ["--spec-draft-n-max", "16", "--spec-draft-n-min", "1"]
        elif "--draft-max" in help_text:
            cmd += ["--draft-max", "16", "--draft-min", "1"]
    cmd += spec.extra_args
    return cmd


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #

@dataclass
class Estimate:
    spec: ServerSpec
    weights: int = 0
    kv: int = 0
    overhead: int = 0
    gpu_fraction: float = 0.0
    ctx_used: int | None = None
    experts_on_cpu: int = 0       # MoE expert weights kept in RAM (--n-cpu-moe)
    missing: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.weights + self.kv + self.overhead

    @property
    def gpu_bytes(self) -> int:
        if self.gpu_fraction <= 0:
            return 0
        on_gpu = self.weights - self.experts_on_cpu + self.kv
        return int(on_gpu * self.gpu_fraction) + self.overhead

    @property
    def cpu_bytes(self) -> int:
        return self.total - self.gpu_bytes


def _kv_ratio(args: list[str]) -> float:
    """Average KV-cache size factor from -ctk/-ctv flags (1.0 = f16)."""

    ratios = []
    for flags in (("-ctk", "--cache-type-k"), ("-ctv", "--cache-type-v")):
        value = "f16"
        for i, a in enumerate(args[:-1]):
            if a in flags:
                value = args[i + 1]
        ratios.append(_KV_TYPE_RATIO.get(value, 1.0))
    return sum(ratios) / 2


def estimate(spec: ServerSpec) -> Estimate:
    from .gguf import GGUFError, expert_bytes_on_cpu, model_shape

    est = Estimate(spec)
    # Placement is known even if the header can't be parsed below; without this
    # an unreadable GGUF would silently skip the VRAM check.
    est.gpu_fraction = 1.0 if spec.on_gpu else 0.0
    for path in (spec.model_path, spec.draft_path):
        if path is None:
            continue
        if not path.is_file():
            est.missing.append(path.name)
            continue
        est.weights += path.stat().st_size
        try:
            shape = model_shape(path)
        except (GGUFError, OSError):
            continue
        ctx = spec.ctx or shape.trained_ctx or 4096
        est.ctx_used = est.ctx_used or ctx
        est.kv += int(shape.kv_cache_bytes(ctx) * _kv_ratio(spec.extra_args))
        if path == spec.model_path and spec.cpu_moe_layers:
            try:
                est.experts_on_cpu = expert_bytes_on_cpu(path, spec.cpu_moe_layers)
            except (GGUFError, OSError):
                pass
        if path == spec.model_path:
            n_layers = max(shape.n_layers, 1)
            gl = spec.gpu_layers
            est.gpu_fraction = 1.0 if gl is None else max(0.0, min(1.0, gl / (n_layers + 1)))
    est.overhead = COMPUTE_OVERHEAD_BYTES
    return est


@dataclass
class GpuInfo:
    name: str
    total: int
    used: int
    free: int


def gpu_info() -> GpuInfo | None:
    """First NVIDIA GPU via nvidia-smi, or None (no NVIDIA GPU / driver)."""

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    name, total, used, free = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    return GpuInfo(name, int(total) * MiB, int(used) * MiB, int(free) * MiB)


def _gb(n: int) -> str:
    return f"{n / GiB:.1f} GiB"


# --------------------------------------------------------------------------- #
# Process management
# --------------------------------------------------------------------------- #

def _pidfile(role: str) -> Path:
    return RUNS_DIR / f"{role}.pid"


def _logfile(role: str) -> Path:
    return RUNS_DIR / f"{role}.log"


def _read_pid(role: str) -> int | None:
    try:
        return int(_pidfile(role).read_text().strip())
    except (OSError, ValueError):
        return None


def _is_llama_process(pid: int) -> bool:
    """True if ``pid`` is alive AND is a llama-server (guards against PID reuse)."""

    if os.name == "nt":
        # NB: os.kill(pid, 0) on Windows *terminates* the process. Use tasklist.
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return "llama-server" in out.lower()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        comm = Path(f"/proc/{pid}/comm").read_text()
    except OSError:
        comm = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                              capture_output=True, text=True).stdout
    return "llama" in comm


def health(spec: ServerSpec, timeout: float = 2.0) -> str:
    """'ok', 'loading', or 'down'."""

    try:
        with urllib.request.urlopen(f"{spec.base}/health", timeout=timeout) as resp:
            return "ok" if resp.status == 200 else "loading"
    except urllib.error.HTTPError as exc:
        return "loading" if exc.code == 503 else "down"
    except (urllib.error.URLError, OSError):
        return "down"


def _log_tail(role: str, n: int = 15) -> str:
    try:
        lines = _logfile(role).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join("    " + line for line in lines[-n:])


def _spawn(cmd: list[str], role: str) -> subprocess.Popen:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(_logfile(role), "w", encoding="utf-8")
    kwargs: dict = dict(stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        # Keep running after this CLI exits, with no console window.
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.DETACHED_PROCESS)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    _pidfile(role).write_text(str(proc.pid))
    return proc


def start(
    config: Config,
    roles: list[str] | None = None,
    *,
    cpu: bool = False,
    host: str | None = None,
    force: bool = False,
    out=print,
) -> int:
    binary = find_llama_server(config)
    if binary is None:
        out("✗ llama-server not found. Set LLAMA_SERVER=<path to llama-server(.exe)> or put it "
            "on PATH. See docs/running-locally.md for installing llama.cpp.")
        return 2
    specs = server_specs(config, roles, gpu_layers=0 if cpu else None, host=host)
    if not specs:
        out("nothing to start (no served roles match)")
        return 1

    todo = []
    for spec in specs:
        if not spec.model_path.is_file():
            out(f"✗ {spec.role}: {spec.model_path} not found — run `coding-agent download`")
            return 2
        state = health(spec)
        if state != "down":
            out(f"• {spec.role}: already running on :{spec.port} ({state})")
            continue
        todo.append(spec)
    if not todo:
        return 0

    # The memory budget check — the reason this launcher exists.
    need = sum(estimate(s).gpu_bytes for s in todo)
    gpu = gpu_info()
    if need and gpu is not None:
        out(f"VRAM: need ≈{_gb(need)}, free {_gb(gpu.free)} of {_gb(gpu.total)} ({gpu.name})")
        if need > gpu.free - VRAM_HEADROOM_BYTES and not force:
            out("✗ refusing to start: this would overflow VRAM. On Windows the driver then spills "
                "into system RAM and generation gets slower than CPU-only.\n"
                "  Options: a smaller profile (--profile gpu-6gb), smaller `ctx`, "
                "`--cpu`, stop other GPU apps, or --force to try anyway.")
            return 3
    elif need and gpu is None:
        out("note: nvidia-smi not found; skipping the VRAM check")

    procs = {}
    for spec in todo:
        cmd = build_command(spec, binary)
        where = "CPU" if not spec.on_gpu else f"GPU (ngl={spec.gpu_layers if spec.gpu_layers is not None else 'auto'})"
        shared = f" [also serves: {', '.join(spec.serves[1:])}]" if len(spec.serves) > 1 else ""
        out(f"▶ {spec.role}: {spec.model_path.name} on :{spec.port}, {where}, "
            f"ctx={spec.ctx or 'model default'}{shared}")
        procs[spec.role] = (spec, _spawn(cmd, spec.role))

    deadline = time.monotonic() + READY_TIMEOUT_S
    failed = 0
    for role, (spec, proc) in procs.items():
        while True:
            if proc.poll() is not None:
                out(f"✗ {role} exited with code {proc.returncode}. Last log lines "
                    f"({_logfile(role)}):\n{_log_tail(role)}")
                failed += 1
                break
            if health(spec) == "ok":
                out(f"✓ {role} ready on :{spec.port}")
                break
            if time.monotonic() > deadline:
                out(f"✗ {role} not ready after {READY_TIMEOUT_S}s; see {_logfile(role)}")
                failed += 1
                break
            time.sleep(0.5)
    if failed:
        return 1
    out("all servers ready. Check memory/speed with: coding-agent doctor --bench")
    return 0


def stop(config: Config, out=print) -> int:
    stopped = 0
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for pidfile in sorted(RUNS_DIR.glob("*.pid")):
        role = pidfile.stem
        pid = _read_pid(role)
        if pid is not None and _is_llama_process(pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                import signal

                os.kill(pid, signal.SIGTERM)
            out(f"■ stopped {role} (pid {pid})")
            stopped += 1
        pidfile.unlink(missing_ok=True)
    if not stopped:
        out("no running servers started by this tool")
    return 0


def status(config: Config, out=print) -> int:
    specs = server_specs(config)
    out(f"profile: {config.profile}")
    down = 0
    for spec in specs:
        state = health(spec)
        down += state != "ok"
        pid = _read_pid(spec.role)
        roles = ", ".join(spec.serves)
        out(f"  {spec.role:<11} :{spec.port}  {state:<8} pid={pid or '-':<7} roles: {roles}")
    gpu = gpu_info()
    if gpu:
        out(f"GPU: {gpu.name} — {_gb(gpu.used)} used of {_gb(gpu.total)}")
    return 0 if not down else 1


# --------------------------------------------------------------------------- #
# Doctor & bench
# --------------------------------------------------------------------------- #

def bench_server(spec: ServerSpec, max_tokens: int = 128) -> float | None:
    """Tokens/second for one short generation (llama-server's own timing)."""

    body = json.dumps({
        "model": spec.alias,
        "messages": [{"role": "user", "content": "Count from 1 to 60, separated by spaces."}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{spec.base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    tps = (data.get("timings") or {}).get("predicted_per_second")
    if tps:
        return float(tps)
    n = (data.get("usage") or {}).get("completion_tokens") or 0
    return n / max(time.monotonic() - started, 1e-6) if n else None


def doctor(config: Config, bench: bool = False, out=print) -> int:
    problems = 0
    out(f"profile   : {config.profile}  ({config.profile_path})")
    out(f"python    : {sys.version.split()[0]} at {sys.executable}")

    binary = find_llama_server(config)
    if binary:
        devices = ""
        try:
            proc = subprocess.run([binary, "--list-devices"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30)
            devices = "; ".join(line.strip() for line in (proc.stdout + proc.stderr).splitlines()
                                if ":" in line and ("MiB" in line or "CUDA" in line or "Vulkan" in line))
        except (OSError, subprocess.SubprocessError):
            pass
        out(f"llama-server: {binary}")
        out(f"  devices : {devices or 'none listed (CPU-only build?)'}")
        if not devices and any(s.on_gpu for s in server_specs(config)):
            out("  ! this build reports no GPU device; GPU roles will run on the CPU")
            problems += 1
    else:
        out("llama-server: NOT FOUND — set LLAMA_SERVER or add it to PATH")
        problems += 1

    gpu = gpu_info()
    out(f"GPU       : {gpu.name}, {_gb(gpu.used)} used / {_gb(gpu.total)}" if gpu
        else "GPU       : no NVIDIA GPU detected (nvidia-smi unavailable)")

    out("\nservers (estimated memory = weights + KV cache at ctx + ~0.25 GiB buffers):")
    gpu_total = cpu_total = 0
    incomplete = False
    for spec in server_specs(config):
        est = estimate(spec)
        shared = f"  (+ {', '.join(spec.serves[1:])})" if len(spec.serves) > 1 else ""
        if spec.model_path.name in est.missing:
            incomplete = True
            out(f"  {spec.role:<11}:{spec.port} {spec.model_path.name:<42} not downloaded{shared}")
            out(f"    ! run `coding-agent download {spec.role}` (then re-run doctor for the budget)")
            problems += 1
            continue
        gpu_total += est.gpu_bytes
        cpu_total += est.cpu_bytes
        place = "GPU" if est.gpu_fraction >= 1 else ("CPU" if est.gpu_fraction <= 0 else f"{est.gpu_fraction:.0%} GPU")
        if est.experts_on_cpu and est.gpu_fraction > 0:
            place = "GPU+RAM"
        line = (f"  {spec.role:<11}:{spec.port} {spec.model_path.name:<42} {place:<7} "
                f"ctx {est.ctx_used or '?':>6}  weights {_gb(est.weights)}  kv {_gb(est.kv)}  "
                f"→ ≈{_gb(est.total)}{shared}")
        out(line)
        if est.experts_on_cpu and est.gpu_fraction > 0:
            out(f"    MoE: {_gb(est.experts_on_cpu)} of expert weights stay in RAM "
                f"(--n-cpu-moe {spec.cpu_moe_layers}); ≈{_gb(est.gpu_bytes)} on the GPU")
        for name in est.missing:
            out(f"    ! missing {name} — run `coding-agent download {spec.role}`")
            problems += 1

    budget = f"≈{_gb(gpu_total)} VRAM, ≈{_gb(cpu_total)} RAM"
    if incomplete:
        out(f"\ntotal     : {budget} so far — incomplete until every model is downloaded")
    elif gpu:
        verdict = "fits" if gpu_total <= gpu.total - VRAM_HEADROOM_BYTES else "DOES NOT FIT"
        out(f"\ntotal     : {budget} — {verdict} in {_gb(gpu.total)} of VRAM")
        if verdict != "fits":
            problems += 1
    else:
        out(f"\ntotal     : {budget}")

    out("\nhealth:")
    running = []
    for spec in server_specs(config):
        state = health(spec)
        out(f"  {spec.role:<11} {spec.base}  {state}")
        if state == "ok":
            running.append(spec)
    if not running:
        out("  (nothing running — start with `coding-agent serve start`)")

    if bench and running:
        out("\nbench (128 tokens, greedy):")
        for spec in running:
            tps = bench_server(spec)
            if tps is None:
                out(f"  {spec.role:<11} failed")
                problems += 1
                continue
            flag = ""
            if spec.on_gpu and tps < SLOW_GPU_TOK_S:
                flag = ("  ! slow for a GPU: VRAM may have spilled into system RAM, or the "
                        "build has no GPU backend")
                problems += 1
            out(f"  {spec.role:<11} {tps:6.1f} tok/s{flag}")
        if gpu:
            now = gpu_info()
            if now:
                out(f"  GPU memory in use now: {_gb(now.used)} of {_gb(now.total)}")

    out(f"\n{'no problems found' if not problems else f'{problems} problem(s) found'}")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download(config: Config, roles: list[str] | None = None, out=print) -> int:
    mdir = models_dir(config)
    mdir.mkdir(parents=True, exist_ok=True)
    wanted = {config.role(r).shares or r for r in roles} if roles else None
    files: list[tuple[str, str]] = []
    for role in config.served_roles():
        if wanted is not None and role.name not in wanted:
            continue
        if role.gguf and role.hf_repo:
            files.append((role.hf_repo, role.gguf))
        if role.draft_gguf and role.draft_hf_repo:
            files.append((role.draft_hf_repo, role.draft_gguf))

    failures = 0
    for repo, filename in files:
        if (mdir / filename).is_file():
            out(f"✓ {filename} (already downloaded)")
            continue
        out(f"↓ {repo} :: {filename}")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(mdir))
            continue
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            out(f"  ✗ {exc}")
            failures += 1
            continue
        hf = shutil.which("hf") or shutil.which("huggingface-cli")
        if hf is None:
            out("  ✗ install the Hugging Face hub client:  pip install -U huggingface_hub")
            return 2
        proc = subprocess.run([hf, "download", repo, filename, "--local-dir", str(mdir)])
        failures += proc.returncode != 0
    return 0 if not failures else 1
