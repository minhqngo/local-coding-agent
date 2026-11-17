#!/usr/bin/env python3
"""Run the agent against a task set and report pass rate.

Each task in eval/tasks/*.yaml declares its setup files, a natural-language
prompt, an independent ``verify.test_command``, and (for offline runs) a
``mock_actions`` script. A task passes only if the verify command exits 0 in the
task's workdir *after* the agent finishes — the agent's own success claim is
recorded but never trusted.

    python eval/run_eval.py --mock                      # offline, deterministic
    python eval/run_eval.py                             # real models, multi-model agent
    python eval/run_eval.py --single-model              # real models, coder alone
    python eval/run_eval.py --profile gpu-12gb --tasks patch_off_by_one --keep

The multi-model vs single-model comparison is the experiment this repo exists
for: same tasks, same harness, same gate — only the deciding differs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Make `import coding_agent` work when run straight from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coding_agent.cli import _check_servers, roles_needed  # noqa: E402
from coding_agent.config import load_config  # noqa: E402
from coding_agent.llm import ModelUnavailable  # noqa: E402
from coding_agent.loop import build_loop  # noqa: E402
from coding_agent.sandbox import build_command_env, run_command  # noqa: E402
from coding_agent.single_model import build_single_model_loop  # noqa: E402
from coding_agent.trace import Tracer  # noqa: E402

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
_VERIFY_TIMEOUT_S = 120


@dataclass
class TaskOutcome:
    name: str
    agent_success: bool
    verify_passed: bool
    turns: int
    seconds: float
    model_calls: int
    tokens: int
    summary: str
    workdir: Path | None = None

    @property
    def passed(self) -> bool:
        return self.verify_passed


def _materialize(task: dict, workdir: Path) -> None:
    for rel, content in (task.get("setup", {}).get("files", {}) or {}).items():
        target = workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))


def _verify(command: str, workdir: Path) -> bool:
    # Same environment rules and tree-killing timeout as the agent's sandbox, so
    # an agent-authored infinite loop fails the task instead of hanging the eval.
    result = run_command(command, workdir, build_command_env(workdir), _VERIFY_TIMEOUT_S)
    return result.ok


def run_task(task_path: Path, args: argparse.Namespace) -> TaskOutcome:
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    config = load_config(args.profile, mock=args.mock or None)
    if args.max_turns:
        config.loop.max_turns = args.max_turns
    test_command = task.get("verify", {}).get("test_command", config.loop.test_command)
    config.loop.test_command = test_command

    responder = None
    if config.mock:
        from coding_agent.mock_brain import MockBrain

        responder = MockBrain(
            coder_actions=task.get("mock_actions", []),
            test_command=test_command,
            single_model=args.single_model,
        )

    workdir = Path(tempfile.mkdtemp(prefix=f"eval_{task['name']}_"))
    tracer = Tracer(workdir / ".coding-agent" / "eval.trace.jsonl" if args.keep else None)
    started = time.monotonic()
    try:
        _materialize(task, workdir)
        if args.single_model:
            loop = build_single_model_loop(config, workdir, responder=responder, tracer=tracer)
        else:
            loop = build_loop(config, workdir, responder=responder, tracer=tracer)
        try:
            result = loop.run(task["description"].strip())
            success, turns, summary = result.success, result.turns, result.summary
        except ModelUnavailable as exc:
            success, turns, summary = False, 0, f"model unavailable: {exc}"
        stats = tracer.summary()
        return TaskOutcome(
            name=task["name"],
            agent_success=success,
            verify_passed=_verify(test_command, workdir),
            turns=turns,
            seconds=time.monotonic() - started,
            model_calls=stats["model_calls"],
            tokens=stats["prompt_tokens"] + stats["completion_tokens"],
            summary=summary,
            workdir=workdir if args.keep else None,
        )
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", "--config", dest="profile", default=None,
                   help="hardware profile (gpu-6gb, gpu-12gb) or YAML path")
    p.add_argument("--mock", action="store_true", help="run offline with each task's mock_actions")
    p.add_argument("--tasks", nargs="*", help="task names to run (default: all)")
    p.add_argument("--single-model", "--baseline", dest="single_model", action="store_true",
                   help="run the single-model agent (the control) instead of the team")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--keep", action="store_true",
                   help="keep each task's workdir and trace for inspection")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    task_files = sorted(TASKS_DIR.glob("*.yaml"))
    if args.tasks:
        wanted = set(args.tasks)
        task_files = [t for t in task_files if t.stem in wanted]
    if not task_files:
        print("no matching tasks", file=sys.stderr)
        return 2

    config = load_config(args.profile, mock=args.mock or None)
    _check_servers(config, roles_needed(config, single_model=args.single_model, retrieval=False))

    mode = "single-model" if args.single_model else "multi-model"
    print(f"running {len(task_files)} task(s) — agent: {mode} — profile: {config.profile} "
          f"— mock: {config.mock}\n")
    outcomes: list[TaskOutcome] = []
    for tf in task_files:
        o = run_task(tf, args)
        outcomes.append(o)
        mark = "PASS" if o.passed else "FAIL"
        print(f"[{mark}] {o.name:22s} turns={o.turns:2d}  {o.seconds:6.1f}s  "
              f"calls={o.model_calls:3d}  tokens={o.tokens:6d}  agent_success={o.agent_success}")
        if not o.passed:
            print(f"       {o.summary}")
        if o.workdir:
            print(f"       kept: {o.workdir}")
            print(f"       view: coding-agent trace {o.workdir / '.coding-agent' / 'eval.trace.jsonl'}")

    passed = sum(o.passed for o in outcomes)
    total = len(outcomes)
    secs = sum(o.seconds for o in outcomes)
    print(f"\npass rate: {passed}/{total} = {passed / total:.0%}   total time {secs:.0f}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
