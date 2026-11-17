"""Command-line entrypoint.

    coding-agent run "fix the failing test in calc.py" --repo ./myproject
    coding-agent run "..." --single-model             # one model, no team
    coding-agent run "..." --mock --mock-script f.json # offline, scripted
    coding-agent ask "where is auth handled?" --repo ./myproject
    coding-agent trace latest --repo ./myproject      # what did the models see?

    coding-agent serve start | status | stop          # llama-server per role
    coding-agent doctor [--bench]                     # will it fit? is it fast?
    coding-agent download                             # fetch the GGUF weights

Global flags (--profile/--repo/--mock/--verbose) work before OR after the
subcommand. --profile picks a hardware profile (config/profiles/*.yaml) or any
YAML path; the default comes from $CODING_AGENT_PROFILE, else gpu-6gb.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .llm import ChatError, ModelUnavailable
from .loop import ResumeState, TranscriptEntry, build_loop
from .registry import ModelRegistry
from .roles import ToolCall


def _make_responder(mock: bool, mock_script: str | None, test_command: str, single_model: bool):
    if not mock:
        return None
    from .mock_brain import MockBrain

    coder_actions = []
    if mock_script:
        coder_actions = json.loads(Path(mock_script).read_text(encoding="utf-8"))
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


def _print_result(result, trace_path: Path | None = None, tracer=None) -> int:
    print("\n── result ─────────────────────────────")
    print(f"success       : {result.success}")
    print(f"turns         : {result.turns}")
    print(f"tests_passed  : {result.tests_passed}")
    print(f"summary       : {result.summary}")
    if tracer is not None:
        s = tracer.summary()
        print(f"model calls   : {s['model_calls']} ({s['model_seconds']}s, "
              f"{s['prompt_tokens']}→{s['completion_tokens']} tokens)")
    if trace_path is not None:
        print(f"trace         : {trace_path}")
    return 0 if result.success else 1


def _resume_state(data: dict) -> ResumeState:
    return ResumeState(
        observations=list(data.get("observations", [])),
        raw_context=list(data.get("raw_context", [])),
        tests_passed=bool(data.get("tests_passed", False)),
        transcript=[TranscriptEntry(**e) for e in data.get("transcript", [])],
        turns_done=int(data.get("turns_done", 0)),
    )


def roles_needed(config, *, single_model: bool, retrieval: bool) -> list[str]:
    if single_model:
        return ["coder"]
    roles = ["planner", "coder", "extractor"]
    if config.loop.dispatch_all:
        roles.append("dispatcher")
    if retrieval:
        roles.append("rag")
    return roles


def preflight(config, roles: list[str]) -> list[str]:
    """Return a list of problems with the model endpoints these roles need."""

    import urllib.error
    import urllib.request

    problems = []
    checked: dict[str, str] = {}
    for name in roles:
        role = config.role(name)
        url = role.base_url.rstrip("/") + "/models"
        if url not in checked:
            try:
                with urllib.request.urlopen(url, timeout=3):
                    checked[url] = ""
            except (urllib.error.URLError, OSError) as exc:
                checked[url] = f"{exc.__class__.__name__}"
        if checked[url]:
            problems.append(f"{name}: {role.base_url} unreachable ({checked[url]})")
    return problems


def _check_servers(config, roles: list[str]) -> None:
    if config.mock:
        return
    problems = preflight(config, roles)
    if problems:
        lines = "\n  ".join(problems)
        raise SystemExit(
            f"✗ model servers not reachable (profile {config.profile}):\n  {lines}\n"
            "Start them with `coding-agent serve start` (same --profile), "
            "or run offline with --mock."
        )


def _cmd_run(args: argparse.Namespace) -> int:
    if args.mock_script and not args.mock:
        raise SystemExit("--mock-script only applies with --mock")
    if args.single_model and args.resume:
        raise SystemExit("--resume is not supported for the single-model agent")

    config = load_config(args.profile, mock=args.mock or None)
    if args.max_turns is not None:
        config.loop.max_turns = args.max_turns
    if args.backend:
        config.sandbox.backend = args.backend
    if args.dispatch_all:
        config.loop.dispatch_all = True
    if args.test_command:
        config.loop.test_command = args.test_command

    workdir = Path(args.repo).resolve()
    if not workdir.is_dir():
        raise SystemExit(f"--repo {workdir} is not a directory")
    _check_servers(config, roles_needed(config, single_model=args.single_model,
                                        retrieval=args.enable_retrieval))
    responder = _make_responder(
        config.mock, args.mock_script, config.loop.test_command, args.single_model
    )
    logger = (lambda m: print(m, file=sys.stderr)) if args.verbose else None

    from .session import SessionStore
    from .trace import Tracer

    store = SessionStore(workdir)
    resume = None
    task = args.task
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
    trace_path = store.trace_path(session_id)
    tracer = Tracer(trace_path)

    mode = "single-model" if args.single_model else "multi-model"
    print(f"▶ task: {task}")
    print(f"  repo: {workdir}  |  profile: {config.profile}  |  mock: {config.mock}  |  "
          f"mode: {mode}  |  max_turns: {config.loop.max_turns}")

    try:
        if args.single_model:
            from .single_model import build_single_model_loop

            loop = build_single_model_loop(
                config, workdir, responder=responder, logger=logger,
                confirm=_confirm_fn if args.confirm else None, tracer=tracer,
            )
            result = loop.run(task)
        else:
            loop = build_loop(
                config,
                workdir,
                responder=responder,
                logger=logger,
                enable_retrieval=args.enable_retrieval,
                confirm=_confirm_fn if args.confirm else None,
                session_store=None if args.no_session else store,
                session_id=session_id,
                tracer=tracer,
            )
            result = loop.run(task, resume=resume)
    except ModelUnavailable as exc:
        print(f"\n✗ stopped: a model is unavailable.\n  {exc}", file=sys.stderr)
        if not args.single_model and not args.no_session:
            print(f"  progress was saved; continue with: --resume {session_id}", file=sys.stderr)
        print(f"  trace: {trace_path}", file=sys.stderr)
        return 3
    return _print_result(result, trace_path, tracer)


def _cmd_ask(args: argparse.Namespace) -> int:
    config = load_config(args.profile, mock=args.mock or None)
    workdir = Path(args.repo).resolve()
    _check_servers(config, ["rag"])
    responder = _make_responder(config.mock, None, config.loop.test_command, False)

    registry = ModelRegistry(config, responder=responder)
    from .retrieval import Retriever

    retriever = Retriever(config.retrieval, workdir).build()
    hits = retriever.search(args.question)
    chunks = [h.render() for h in hits]
    try:
        answer = registry.rag.answer(args.question, chunks)
    except ModelUnavailable as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 3
    except ChatError as exc:
        print(f"✗ the repo-QA model gave no usable answer: {exc}", file=sys.stderr)
        return 1

    print(answer)
    if args.verbose:
        print("\n── sources ────────────", file=sys.stderr)
        for h in hits:
            print(f"  {h.chunk.path}:{h.chunk.start_line}-{h.chunk.end_line} ({h.score:.3f})", file=sys.stderr)
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    from .session import SessionStore
    from .trace import load_trace, render_trace

    if Path(args.ref).is_file():
        path = Path(args.ref)
    else:
        store = SessionStore(Path(args.repo).resolve())
        path = store.find_trace(args.ref)
        if path is None:
            raise SystemExit(f"no trace matching {args.ref!r} under {store.dir}")
    print(f"# {path}")
    print(render_trace(load_trace(path), full=args.full))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from . import serving

    config = load_config(args.profile, mock=False)
    if args.action == "start":
        return serving.start(config, args.roles or None, cpu=args.cpu, host=args.host,
                             force=args.force)
    if args.action == "stop":
        return serving.stop(config)
    return serving.status(config)


def _cmd_doctor(args: argparse.Namespace) -> int:
    from . import serving

    return serving.doctor(load_config(args.profile, mock=False), bench=args.bench)


def _cmd_download(args: argparse.Namespace) -> int:
    from . import serving

    return serving.download(load_config(args.profile, mock=False), args.roles or None)


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the global flags. On subparsers use suppress=True so a value given
    before the subcommand isn't clobbered by the subparser's default."""

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--profile", "--config", dest="profile", default=default(None),
                        help="hardware profile name (gpu-6gb, gpu-12gb) or path to a YAML file "
                             "(default: $CODING_AGENT_PROFILE or gpu-6gb)")
    parser.add_argument("--repo", default=default("."), help="repository / working directory")
    parser.add_argument("--mock", action="store_true", default=default(False),
                        help="run fully offline with the deterministic mock brain")
    parser.add_argument("--verbose", "-v", action="store_true", default=default(False),
                        help="stream the transcript to stderr")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coding-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_global_flags(p, suppress=False)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on a coding task")
    _add_global_flags(run, suppress=True)  # allow globals after the subcommand too
    run.add_argument("task", nargs="?", default=None, help="the coding task, in natural language")
    run.add_argument("--single-model", action="store_true",
                     help="one model (the coder) does everything — no planner/extractor")
    run.add_argument("--test-command", default=None,
                     help="override loop.test_command (the finish gate) for this run")
    run.add_argument("--max-turns", type=int, default=None)
    run.add_argument("--mock-script", default=None, help="JSON list of scripted coder actions (mock mode)")
    run.add_argument("--backend", choices=["subprocess", "docker"], default=None)
    run.add_argument("--enable-retrieval", action="store_true", help="build the repo index and expose search_codebase")
    run.add_argument("--dispatch-all", action="store_true", help="route every tool call through the dispatcher model")
    run.add_argument("--confirm", action="store_true", help="ask before each write_file / patch_file / run_shell "
                          "(the configured test command after an edit runs without asking)")
    run.add_argument("--resume", default=None, metavar="latest|ID", help="resume a saved session and continue it")
    run.add_argument("--no-session", action="store_true", help="do not persist a resumable session")
    run.set_defaults(func=_cmd_run)

    ask = sub.add_parser("ask", help="ask a question about the codebase (RAG)")
    _add_global_flags(ask, suppress=True)
    ask.add_argument("question", help="natural-language question about the repo")
    ask.set_defaults(func=_cmd_ask)

    trace = sub.add_parser("trace", help="show what every model saw and said in a run")
    _add_global_flags(trace, suppress=True)
    trace.add_argument("ref", nargs="?", default="latest",
                       help="'latest', a session id, or a path to a .trace.jsonl file")
    trace.add_argument("--full", action="store_true", help="print full prompts and outputs")
    trace.set_defaults(func=_cmd_trace)

    serve = sub.add_parser("serve", help="start/stop/status of the local llama-server fleet")
    _add_global_flags(serve, suppress=True)
    serve.add_argument("action", choices=["start", "stop", "status"])
    serve.add_argument("roles", nargs="*", help="only these roles (default: all served roles)")
    serve.add_argument("--cpu", action="store_true", help="put every model on the CPU (-ngl 0)")
    serve.add_argument("--host", default=None, help="bind address (default from profile: 127.0.0.1)")
    serve.add_argument("--force", action="store_true", help="start even if the VRAM estimate does not fit")
    serve.set_defaults(func=_cmd_serve)

    doctor = sub.add_parser("doctor", help="check binary, models, memory budget and server health")
    _add_global_flags(doctor, suppress=True)
    doctor.add_argument("--bench", action="store_true", help="also measure tokens/s of running servers")
    doctor.set_defaults(func=_cmd_doctor)

    download = sub.add_parser("download", help="download the profile's GGUF weights into models/")
    _add_global_flags(download, suppress=True)
    download.add_argument("roles", nargs="*", help="only these roles (default: all)")
    download.set_defaults(func=_cmd_download)

    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/pipes default to cp1252; never crash on a "▶" or "✓".
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
