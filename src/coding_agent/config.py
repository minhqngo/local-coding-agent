"""Configuration loading.

The single source of truth is ``config/models.yaml``. Everything downstream
reads a :class:`Config` object; nothing else touches the raw YAML.
"""

from __future__ import annotations

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
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"


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
    thinking: bool = False
    # Served context window (passed to llama-server as -c; unused at inference).
    ctx: int | None = None
    # Serving / download hints (unused at inference time, read by scripts).
    gguf: str | None = None
    hf_repo: str | None = None
    draft_gguf: str | None = None
    draft_hf_repo: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


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
    # When true, EVERY tool call (planner-issued and clean coder actions) is
    # routed through the LFM2 dispatcher model, honouring the plan's Phase-3
    # "route every tool invocation through LFM2-1.2B-Tool" literally. Off by
    # default: clean coder actions are mapped deterministically to save a hop.
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
    mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        try:
            return self.roles[name]
        except KeyError:  # pragma: no cover - defensive
            raise KeyError(
                f"No role '{name}' in config. Known roles: {sorted(self.roles)}"
            ) from None


def _env_mock() -> bool | None:
    """Return the env-var mock setting, or None when the var is unset."""

    raw = os.environ.get("CODING_AGENT_MOCK")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str | Path | None = None, mock: bool | None = None) -> Config:
    """Load and validate the agent configuration.

    ``mock`` precedence: explicit argument > ``CODING_AGENT_MOCK`` env var >
    the ``mock`` key in the YAML > ``False``.
    """

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text()) or {}
    defaults = data.get("defaults", {})
    base_url_template = defaults.get(
        "base_url_template", "http://127.0.0.1:{port}/v1"
    )
    default_timeout = int(defaults.get("timeout_s", 120))
    default_max_tokens = int(defaults.get("max_tokens", 2048))

    roles: dict[str, RoleConfig] = {}
    for name, spec in (data.get("roles") or {}).items():
        spec = dict(spec)
        port = int(spec.get("port", 8000))
        known = {
            "model", "port", "temperature", "max_tokens", "timeout_s",
            "thinking", "ctx", "gguf", "hf_repo", "draft_gguf", "draft_hf_repo",
            "base_url",
        }
        extra = {k: v for k, v in spec.items() if k not in known}
        roles[name] = RoleConfig(
            name=name,
            model=spec.get("model", name),
            port=port,
            base_url=spec.get("base_url", base_url_template.format(port=port)),
            temperature=float(spec.get("temperature", 0.2)),
            max_tokens=int(spec.get("max_tokens", default_max_tokens)),
            timeout_s=int(spec.get("timeout_s", default_timeout)),
            thinking=bool(spec.get("thinking", False)),
            ctx=int(spec["ctx"]) if spec.get("ctx") is not None else None,
            gguf=spec.get("gguf"),
            hf_repo=spec.get("hf_repo"),
            draft_gguf=spec.get("draft_gguf"),
            draft_hf_repo=spec.get("draft_hf_repo"),
            extra=extra,
        )

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

    # Precedence: explicit arg > CODING_AGENT_MOCK env var > YAML `mock` > False.
    if mock is not None:
        resolved_mock = mock
    elif (env_mock := _env_mock()) is not None:
        resolved_mock = env_mock
    else:
        resolved_mock = bool(data.get("mock", False))

    return Config(
        roles=roles,
        sandbox=sandbox,
        loop=loop,
        retrieval=retrieval,
        mock=resolved_mock,
        raw=data,
    )
