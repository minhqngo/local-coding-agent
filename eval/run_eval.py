#!/usr/bin/env python3
"""Phase 5 — run the agent against a task set and report pass rate.

Each task in eval/tasks/*.yaml declares its setup files, a natural-language
prompt, an independent `verify.test_command`, and (for offline runs) a
`mock_actions` script. A task passes only if the verify command exits 0 in the
task's workdir *after* the agent finishes — the agent's own success claim is
never trusted.

    python eval/run_eval.py                 # real models (needs servers up)
    python eval/run_eval.py --mock          # deterministic offline dry-run
    python eval/run_eval.py --mock --tasks fix_divide_by_zero

This is the control described in the plan: build 20–30 tasks from your own
projects here, then compare the full multi-model system against a coder-only
baseline (see --baseline) to answer whether the glue models actually help.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

# Make `import coding_agent` work when run straight from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coding_agent.baseline import build_baseline_loop  # noqa: E402
from coding_agent.config import load_config  # noqa: E402
from coding_agent.loop import build_loop  # noqa: E402

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
_VERIFY_TIMEOUT_S = 120


@dataclass
class TaskOutcome:
    name: str
    agent_success: bool
    verify_passed: bool
    turns: int

    @property
    def passed(self) -> bool:
        return self.verify_passed


def _materialize(task: dict, workdir: Path) -> None:
    for rel, content in (task.get("setup", {}).get("files", {}) or {}).items():
        target = workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _verify(task: dict, workdir: Path) -> bool:
    cmd = task.get("verify", {}).get("test_command", "python -m pytest -q")
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=workdir, capture_output=True, text=True,
            timeout=_VERIFY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False  # an agent-authored infinite test counts as a failure
    return proc.returncode == 0


def run_task(task_path: Path, mock: bool, baseline: bool) -> TaskOutcome:
    """Run one task through the full loop, or the coder-only control (baseline).

    The baseline is a *real* control: the coder alone (no planner/extractor/
    dispatcher) with the same tools and passing-test gate — the only variable is
    the presence of the glue models. In mock mode it uses the same scripted fix
    as the full run, so both are exercised end to end rather than a no-op 0%.
    """

    task = yaml.safe_load(task_path.read_text())
    config = load_config(mock=mock or None)
    test_command = task.get("verify", {}).get("test_command", config.loop.test_command)
    config.loop.test_command = test_command

    responder = None
    if config.mock:
        from coding_agent.mock_brain import MockBrain

        responder = MockBrain(
            coder_actions=task.get("mock_actions", []),
            test_command=test_command,
            single_model=baseline,
        )

    workdir = Path(tempfile.mkdtemp(prefix=f"eval_{task['name']}_"))
    try:
        _materialize(task, workdir)
        if baseline:
            loop = build_baseline_loop(config, workdir, responder=responder)
        else:
            loop = build_loop(config, workdir, responder=responder)
        result = loop.run(task["description"].strip())
        verify_passed = _verify(task, workdir)
        return TaskOutcome(
            name=task["name"],
            agent_success=result.success,
            verify_passed=verify_passed,
            turns=result.turns,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true", help="run offline with each task's mock_actions")
    p.add_argument("--tasks", nargs="*", help="task names to run (default: all)")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="run the Phase-2 coder-only control (no planner/extractor/dispatcher)",
    )
    args = p.parse_args(argv)

    task_files = sorted(TASKS_DIR.glob("*.yaml"))
    if args.tasks:
        wanted = set(args.tasks)
        task_files = [t for t in task_files if t.stem in wanted]
    if not task_files:
        print("no matching tasks", file=sys.stderr)
        return 2

    mode = "coder-only baseline" if args.baseline else "full multi-model"
    print(f"running {len(task_files)} task(s) — mode: {mode} — mock: {args.mock}\n")
    outcomes: list[TaskOutcome] = []
    for tf in task_files:
        outcome = run_task(tf, mock=args.mock, baseline=args.baseline)
        outcomes.append(outcome)
        mark = "PASS" if outcome.passed else "FAIL"
        print(f"[{mark}] {outcome.name:24s} turns={outcome.turns:2d} agent_success={outcome.agent_success}")

    passed = sum(o.passed for o in outcomes)
    total = len(outcomes)
    print(f"\npass rate: {passed}/{total} = {passed / total:.0%}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
