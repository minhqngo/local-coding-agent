# Evaluation (Phase 5)

The empirical question this project exists to answer: **does a team of cheap
specialists around a small coder beat the small coder alone?**

## Task format

Each `tasks/*.yaml` is one realistic mini-task:

```yaml
name: fix_divide_by_zero
description: >
  Natural-language prompt handed to the agent's planner.
setup:
  files:                 # created in a fresh temp workdir before the run
    calc.py: |
      ...
verify:
  test_command: "python -m pytest -q"   # run AFTER the agent; exit 0 == pass
mock_actions:            # deterministic fix for offline (--mock) runs
  - action: write_file
    path: calc.py
    content: |
      ...
```

A task **passes** only if `verify.test_command` exits 0 in its workdir once the
agent finishes. The agent's own "success" claim is recorded but never trusted.

## Running

```bash
# Offline, deterministic — no models, no servers. Good for CI / smoke testing.
python eval/run_eval.py --mock

# Real models — start the servers first (scripts/serve.sh) then:
python eval/run_eval.py

# A subset:
python eval/run_eval.py --mock --tasks fix_divide_by_zero
```

## Measuring the multi-model hypothesis

```bash
python eval/run_eval.py               # full system: planner+coder+dispatcher+extractor
python eval/run_eval.py --baseline    # coder-only control (same tasks, same gate)
```

`--baseline` runs the **real** Phase-2 control (`coding_agent.baseline`): the
coder alone with read/write/shell tools and the same passing-test gate — the
only variable is the presence of the glue models. Compare the two pass rates to
answer *"does the glue beat the coder alone?"*

Grow this set to 20–30 tasks drawn from your own projects — small bug fixes,
missing tests, single-file refactors, and repo-Q&A tasks. That is where the
answer actually lives; two toy tasks only prove the harness runs.
