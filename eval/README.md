# Evaluation

The question this project exists to answer: **does a team of cheap specialists
around a small coder beat the small coder alone?** Both agents share the same
harness, tools and test gate, so running them on the same tasks isolates the
one thing that differs — how they decide.

## Task format

Each `tasks/*.yaml` is one small, realistic task:

```yaml
name: fix_divide_by_zero
description: >
  Natural-language prompt handed to the agent.
setup:
  files:                 # created in a fresh temp directory before the run
    calc.py: |
      ...
verify:
  test_command: "python -m pytest -q"   # run AFTER the agent; exit 0 == pass
mock_actions:            # scripted fix for offline (--mock) runs
  - action: write_file
    path: calc.py
    content: |
      ...
```

A task **passes** only if `verify.test_command` exits 0 once the agent stops.
The agent's own success claim is printed (`agent_success=`) but never trusted:
the agent can be right without knowing it (it ran out of turns after a correct
edit) or claim success wrongly.

## Running

```bash
python eval/run_eval.py --mock                         # offline, scripted, no models
python eval/run_eval.py                                # multi-model, default profile
python eval/run_eval.py --single-model                 # the coder alone (the control)
python eval/run_eval.py --profile gpu-12gb --tasks patch_off_by_one --keep
```

| Flag | Meaning |
|---|---|
| `--profile NAME` | hardware profile (servers must be running) |
| `--single-model` | run the single-model agent (alias: `--baseline`) |
| `--tasks A B` | only these task names |
| `--max-turns N` | override the turn limit |
| `--keep` | keep each task's directory and trace (printed paths) to inspect failures |

Each line reports turns, wall time, model calls and tokens, so you can compare
cost as well as pass rate.

## Reading failures

Run with `--keep`, then render the kept trace:

```bash
coding-agent trace <path to eval.trace.jsonl>          # add --full for complete prompts
```

Every improvement in `docs/how-it-works.md` ("Lessons from real runs") came from
reading traces of failed tasks this way.

## Making the answer meaningful

Three tasks find harness bugs; they don't rank agents. Runs at temperature > 0
vary from run to run. To answer the question:

- grow the set to 20–30 tasks from your own projects — bug fixes, missing tests,
  single-file refactors;
- keep the tasks' code unlike the prompt examples in `roles.py` (see lesson 6);
- run each configuration several times and compare pass rates *and* time.
