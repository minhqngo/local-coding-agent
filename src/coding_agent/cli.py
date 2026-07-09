"""Command-line entrypoint.

    coding-agent run "fix the failing test in calc.py" --repo ./myproject
    coding-agent run "..." --mock --mock-script fix.json   # offline dry-run
    coding-agent run "..." --single-model                  # Phase-2 coder-only control
    coding-agent run "..." --confirm                       # ask before edits/commands
    coding-agent ask "where is auth handled?" --repo ./myproject

Global flags (--config/--repo/--mock/--verbose) may be given before OR after the
subcommand. In real mode each role talks to its llama-server port (see
config/models.yaml and scripts/serve.sh); in --mock mode the whole fleet is
replaced by a deterministic offline brain, so nothing is downloaded or served.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .loop import ResumeState, TranscriptEntry, build_loop
from .registry import ModelRegistry
from .roles import ToolCall


def _make_responder(mock: bool, mock_script: str | None, test_command: str, single_model: bool):
    if not mock:
        return None
    from .mock_brain import MockBrain

    coder_actions = []
    if mock_script:
        coder_actions = json.loads(Path(mock_script).read_text())
        if not isinstance(coder_actions, list):
            raise SystemExit("--mock-script must contain a JSON list of coder actions")
    return MockBrain(
        coder_actions=coder_actions, test_command=test_command, single_model=single_model
    )


def _confirm_fn(call: ToolCall) -> bool:
    desc = call.arguments.get("command") or call.arguments.get("path") or ""
    try:
        reply = input(f"  allow {call.name}: {desc!r}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return reply in {"y", "yes"}


def _print_result(result) -> int:
    print("\n── result ─────────────────────────────")
    print(f"success       : {result.success}")
    print(f"turns         : {result.turns}")
    print(f"tests_passed  : {result.tests_passed}")
    print(f"summary       : {result.summary}")
    return 0 if result.success else 1


def _resume_state(data: dict) -> ResumeState:
    return ResumeState(
        observations=list(data.get("observations", [])),
        raw_context=list(data.get("raw_context", [])),
        tests_passed=bool(data.get("tests_passed", False)),
        transcript=[TranscriptEntry(**e) for e in data.get("transcript", [])],
        turns_done=int(data.get("turns_done", 0)),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    if args.mock_script and not args.mock:
        raise SystemExit("--mock-script only applies with --mock")
    if args.single_model and args.resume:
        raise SystemExit("--resume is not supported for the coder-only baseline")

    config = load_config(args.config, mock=args.mock or None)
    if args.max_turns is not None:
        config.loop.max_turns = args.max_turns
    if args.backend:
        config.sandbox.backend = args.backend
    if args.dispatch_all:
        config.loop.dispatch_all = True

    workdir = Path(args.repo).resolve()
    responder = _make_responder(
        config.mock, args.mock_script, config.loop.test_command, args.single_model
    )
    logger = (lambda m: print(m, file=sys.stderr)) if args.verbose else None

    # Resolve session state / task (P6).
    store = None
    session_id = None
    resume = None
    task = args.task
    if not args.single_model and not args.no_session:
        from .session import SessionStore

        store = SessionStore(workdir)
        if args.resume:
            data = store.resolve(args.resume)
            if data is None:
                raise SystemExit(f"no session matching --resume {args.resume!r}")
            session_id = data["session_id"]
            resume = _resume_state(data)
            task = task or data.get("task")
            print(f"↻ resuming session {session_id} (turns done: {resume.turns_done})")
        else:
            session_id = store.new_id()
    if not task:
        raise SystemExit("a task is required (or use --resume to continue one)")

    mode = "coder-only baseline" if args.single_model else "multi-model"
    print(f"▶ task: {task}")
    print(f"  repo: {workdir}  |  mock: {config.mock}  |  mode: {mode}  |  max_turns: {config.loop.max_turns}")

    if args.single_model:
        from .baseline import build_baseline_loop

        loop = build_baseline_loop(config, workdir, responder=responder, logger=logger)
        return _print_result(loop.run(task))

    loop = build_loop(
        config,
        workdir,
        responder=responder,
        logger=logger,
        enable_retrieval=args.enable_retrieval,
        confirm=_confirm_fn if args.confirm else None,
        session_store=store,
        session_id=session_id,
    )
    return _print_result(loop.run(task, resume=resume))


def _cmd_ask(args: argparse.Namespace) -> int:
    config = load_config(args.config, mock=args.mock or None)
    workdir = Path(args.repo).resolve()
    responder = _make_responder(config.mock, None, config.loop.test_command, False)

    registry = ModelRegistry(config, responder=responder)
    from .retrieval import Retriever

    retriever = Retriever(config.retrieval, workdir).build()
    hits = retriever.search(args.question)
    chunks = [h.render() for h in hits]
    answer = registry.rag.answer(args.question, chunks)

    print(answer)
    if args.verbose:
        print("\n── sources ────────────", file=sys.stderr)
        for h in hits:
            print(f"  {h.chunk.path}:{h.chunk.start_line}-{h.chunk.end_line} ({h.score:.3f})", file=sys.stderr)
    return 0


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the global flags. On subparsers use suppress=True so a value given
    before the subcommand isn't clobbered by the subparser's default."""

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--config", default=default(None), help="path to models.yaml (default: bundled config)")
    parser.add_argument("--repo", default=default("."), help="repository / working directory")
    parser.add_argument("--mock", action="store_true", default=default(False), help="run fully offline with the deterministic brain")
    parser.add_argument("--verbose", action="store_true", default=default(False), help="stream the transcript to stderr")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coding-agent", description=__doc__)
    _add_global_flags(p, suppress=False)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on a coding task")
    _add_global_flags(run, suppress=True)  # allow globals after the subcommand too
    run.add_argument("task", nargs="?", default=None, help="the coding task, in natural language")
    run.add_argument("--mock-script", default=None, help="JSON list of scripted coder actions (mock mode)")
    run.add_argument("--max-turns", type=int, default=None)
    run.add_argument("--backend", choices=["subprocess", "docker"], default=None)
    run.add_argument("--enable-retrieval", action="store_true", help="build the repo index and expose search_codebase")
    run.add_argument("--single-model", action="store_true", help="Phase-2 coder-only baseline (no planner/extractor/dispatcher)")
    run.add_argument("--dispatch-all", action="store_true", help="route every tool call through the LFM2 dispatcher model")
    run.add_argument("--confirm", action="store_true", help="prompt before each write_file / patch_file / run_shell")
    run.add_argument("--resume", default=None, metavar="latest|ID", help="resume a saved session and continue it")
    run.add_argument("--no-session", action="store_true", help="do not persist a resumable session")
    run.set_defaults(func=_cmd_run)

    ask = sub.add_parser("ask", help="ask a question about the codebase (RAG)")
    _add_global_flags(ask, suppress=True)
    ask.add_argument("question", help="natural-language question about the repo")
    ask.set_defaults(func=_cmd_ask)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
