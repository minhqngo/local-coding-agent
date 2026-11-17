"""Configuration loading.

Configuration comes from two YAML files merged together:

* ``config/agent.yaml`` — how the agent *behaves* (sandbox, loop limits, test
  command, retrieval). Identical on every machine.
* a **hardware profile** — which models run *where* (ports, GPU layers, context
  sizes). One per machine class, under ``config/profiles/``:

  - ``gpu-6gb``  — 6 GB GPU / 16 GB RAM laptop (GTX 1660 Ti class). Default.
  - ``gpu-12gb`` — 12 GB GPU laptop (RTX PRO 3000 class).

Pick a profile with ``--profile NAME`` (or a path to any YAML file), or set the
``CODING_AGENT_PROFILE`` environment variable once per machine.

Everything downstream reads a :class:`Config` object; nothing else touches YAML.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is a declared dependency, but keep the import failure legible.
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "PyYAML is required to load config. Install with `pip install PyYAML`."
    ) from exc


# Repository root = two levels up from this file (src/coding_agent/config.py).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_DIR = REPO_ROOT / "config"
AGENT_CONFIG_PATH = CONFIG_DIR / "agent.yaml"
PROFILES_DIR = CONFIG_DIR / "profiles"
DEFAULT_PROFILE = "gpu-6gb"
PROFILE_ENV_VAR = "CODING_AGENT_PROFILE"


@dataclass
class RoleConfig:
    """Everything needed to talk to (or serve) one model in one role."""

    name: str
    model: str
    port: int
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: int = 120
    # None = don't tell the chat template anything; True/False = send
    # chat_template_kwargs.enable_thinking (Qwen3-family thinking switch).
    thinking: bool | None = None
    # --- serving (read by `coding-agent serve`, unused at inference time) ---
    ctx: int | None = None            # context window passed to llama-server -c
    gpu_layers: int | None = None     # llama-server -ngl (99 = all, 0 = CPU)
    # MoE models only: keep the expert weights of the first N layers in system
    # RAM (llama-server --n-cpu-moe N). Big MoE models on small GPUs live on this.
    cpu_moe_layers: int | None = None
    gguf: str | None = None
    hf_repo: str | None = None
    draft_gguf: str | None = None
    draft_hf_repo: str | None = None
    server_args: list[str] = field(default_factory=list)  # extra llama-server flags
    # Name of the role whose server this role reuses (see `share:` in profiles).
    shares: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def served(self) -> bool:
        """True if this role needs its own llama-server process."""
        return self.shares is None and bool(self.gguf)


@dataclass
class ServerConfig:
    models_dir: str = "models"
    host: str = "127.0.0.1"
    binary: str | None = None  # llama-server path; usually set via LLAMA_SERVER


@dataclass
class SandboxConfig:
    backend: str = "subprocess"
    docker_image: str = "python:3.12-slim"
    command_timeout_s: int = 120
    denylist: list[str] = field(default_factory=list)


@dataclass
class LoopConfig:
    max_turns: int = 25
    require_passing_tests: bool = True
    test_command: str = "python -m pytest -q"
    # Abort when the identical outcome repeats this often (0 = never abort).
    max_repeated_outcomes: int = 5
    # Run the test command automatically after every successful edit.
    test_after_edit: bool = True
    # Route EVERY tool call through the dispatcher model (off: clean coder
    # actions are mapped deterministically, which is faster and more reliable).
    dispatch_all: bool = False


@dataclass
class RetrievalConfig:
    chunk_lines: int = 60
    chunk_overlap: int = 10
    top_k: int = 6
    index_dir: str = ".index"
    include_globs: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)


@dataclass
class Config:
    roles: dict[str, RoleConfig]
    sandbox: SandboxConfig
    loop: LoopConfig
    retrieval: RetrievalConfig
    server: ServerConfig = field(default_factory=ServerConfig)
    mock: bool = False
    profile: str = ""          # profile name (or file stem) this config came from
    profile_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        try:
            return self.roles[name]
        except KeyError:
            raise KeyError(
                f"No role '{name}' in profile '{self.profile}'. "
                f"Known roles: {sorted(self.roles)}"
            ) from None

    def served_roles(self) -> list[RoleConfig]:
        """Roles that need their own model server (shared roles excluded)."""
        return [r for r in self.roles.values() if r.served]

    def sharers_of(self, name: str) -> list[str]:
        return [r.name for r in self.roles.values() if r.shares == name]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def resolve_profile_path(profile: str | Path | None = None) -> Path:
    """Map a profile name or path to a YAML file.

    ``None`` → ``$CODING_AGENT_PROFILE`` → ``DEFAULT_PROFILE``. A value that is
    an existing file is used as-is; otherwise it names ``config/profiles/<name>.yaml``.
    """

    chosen = profile or os.environ.get(PROFILE_ENV_VAR) or DEFAULT_PROFILE
    as_path = Path(chosen)
    if as_path.is_file():
        return as_path
    named = PROFILES_DIR / f"{chosen}.yaml"
    if named.is_file():
        return named
    available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
    raise FileNotFoundError(
        f"No profile {str(chosen)!r}: not a file, and not one of {available} "
        f"in {PROFILES_DIR}"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    # Explicit UTF-8: on Windows the default is the locale code page (cp1252).
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _env_mock() -> bool | None:
    """Return the env-var mock setting, or None when the var is unset."""

    raw = os.environ.get("CODING_AGENT_MOCK")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_ROLE_KEYS = {
    "model", "port", "temperature", "max_tokens", "timeout_s", "thinking",
    "ctx", "gpu_layers", "cpu_moe_layers", "gguf", "hf_repo", "draft_gguf", "draft_hf_repo",
    "base_url", "server_args", "share",
}
# Keys a sharing role inherits from the role it shares (it is the same server).
_SHARED_KEYS = ("model", "port", "base_url", "ctx", "gpu_layers", "cpu_moe_layers")


def _build_roles(data: dict[str, Any]) -> dict[str, RoleConfig]:
    defaults = data.get("defaults", {}) or {}
    template = defaults.get("base_url_template", "http://127.0.0.1:{port}/v1")
    default_timeout = int(defaults.get("timeout_s", 120))
    default_max_tokens = int(defaults.get("max_tokens", 2048))
    specs: dict[str, dict[str, Any]] = {
        name: dict(spec or {}) for name, spec in (data.get("roles") or {}).items()
    }

    # Resolve `share:` first: copy the server identity from the target role.
    for name, spec in specs.items():
        target = spec.get("share")
        if target is None:
            continue
        if target not in specs:
            raise ValueError(f"role {name!r} shares unknown role {target!r}")
        if specs[target].get("share"):
            raise ValueError(f"role {name!r} shares {target!r}, which itself shares a role")
        for key in _SHARED_KEYS:
            if key in specs[target] and key not in spec:
                spec[key] = specs[target][key]

    roles: dict[str, RoleConfig] = {}
    for name, spec in specs.items():
        port = int(spec.get("port", 8000))
        thinking = spec.get("thinking")
        roles[name] = RoleConfig(
            name=name,
            model=spec.get("model", name),
            port=port,
            base_url=spec.get("base_url", template.format(port=port)),
            temperature=float(spec.get("temperature", 0.2)),
            max_tokens=int(spec.get("max_tokens", default_max_tokens)),
            timeout_s=int(spec.get("timeout_s", default_timeout)),
            thinking=None if thinking is None else bool(thinking),
            ctx=int(spec["ctx"]) if spec.get("ctx") is not None else None,
            gpu_layers=int(spec["gpu_layers"]) if spec.get("gpu_layers") is not None else None,
            cpu_moe_layers=(int(spec["cpu_moe_layers"])
                            if spec.get("cpu_moe_layers") is not None else None),
            gguf=None if spec.get("share") else spec.get("gguf"),
            hf_repo=spec.get("hf_repo"),
            draft_gguf=spec.get("draft_gguf"),
            draft_hf_repo=spec.get("draft_hf_repo"),
            server_args=[str(a) for a in spec.get("server_args", []) or []],
            shares=spec.get("share"),
            extra={k: v for k, v in spec.items() if k not in _ROLE_KEYS},
        )
    return roles


def load_config(
    path: str | Path | None = None,
    mock: bool | None = None,
) -> Config:
    """Load ``config/agent.yaml`` merged with a hardware profile.

    ``path`` is a profile name (``gpu-6gb``) or a path to a YAML file; see
    :func:`resolve_profile_path`. ``mock`` precedence: explicit argument >
    ``CODING_AGENT_MOCK`` env var > the ``mock`` key in the YAML > ``False``.
    """

    profile_path = resolve_profile_path(path)
    base = _read_yaml(AGENT_CONFIG_PATH) if AGENT_CONFIG_PATH.is_file() else {}
    data = _deep_merge(base, _read_yaml(profile_path))

    sb = data.get("sandbox", {}) or {}
    sandbox = SandboxConfig(
        backend=sb.get("backend", "subprocess"),
        docker_image=sb.get("docker_image", "python:3.12-slim"),
        command_timeout_s=int(sb.get("command_timeout_s", 120)),
        denylist=list(sb.get("denylist", [])),
    )

    lp = data.get("loop", {}) or {}
    loop = LoopConfig(
        max_turns=int(lp.get("max_turns", 25)),
        require_passing_tests=bool(lp.get("require_passing_tests", True)),
        test_command=lp.get("test_command", "python -m pytest -q"),
        max_repeated_outcomes=int(lp.get("max_repeated_outcomes", 5)),
        test_after_edit=bool(lp.get("test_after_edit", True)),
        dispatch_all=bool(lp.get("dispatch_all", False)),
    )

    rt = data.get("retrieval", {}) or {}
    retrieval = RetrievalConfig(
        chunk_lines=int(rt.get("chunk_lines", 60)),
        chunk_overlap=int(rt.get("chunk_overlap", 10)),
        top_k=int(rt.get("top_k", 6)),
        index_dir=rt.get("index_dir", ".index"),
        include_globs=list(rt.get("include_globs", ["**/*.py"])),
        exclude_dirs=list(rt.get("exclude_dirs", [".git", "__pycache__"])),
    )

    sv = data.get("server", {}) or {}
    server = ServerConfig(
        models_dir=str(sv.get("models_dir", "models")),
        host=str(sv.get("host", "127.0.0.1")),
        binary=sv.get("binary"),
    )

    # Precedence: explicit arg > CODING_AGENT_MOCK env var > YAML `mock` > False.
    if mock is not None:
        resolved_mock = mock
    elif (env_mock := _env_mock()) is not None:
        resolved_mock = env_mock
    else:
        resolved_mock = bool(data.get("mock", False))

    return Config(
        roles=_build_roles(data),
        sandbox=sandbox,
        loop=loop,
        retrieval=retrieval,
        server=server,
        mock=resolved_mock,
        profile=profile_path.stem,
        profile_path=profile_path,
        raw=data,
    )
