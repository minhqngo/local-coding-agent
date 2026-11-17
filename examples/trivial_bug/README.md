# Trivial-bug example

`calc.py` has a one-line bug: `safe_divide(1, 0)` raises instead of returning
`0.0`. `test_calc.py` catches it. This is the smallest end-to-end run.

The agent edits files in place, so copy the example to a scratch directory first:

```bash
tmp=$(mktemp -d) && cp examples/trivial_bug/calc.py examples/trivial_bug/test_calc.py "$tmp"/
```

## Offline (no models)

`fix.json` is a scripted coder action. The mock brain drives the real loop —
planner → coder → harness → planner — which writes the fix to disk; the test
gate then verifies it.

```bash
coding-agent run "Make safe_divide return 0.0 when dividing by zero." \
    --repo "$tmp" --mock --mock-script examples/trivial_bug/fix.json -v
coding-agent trace latest --repo "$tmp"      # every prompt, reply and tool call
```

## Real models

With servers running (`coding-agent serve start`, see `docs/running-locally.md`):

```bash
coding-agent run "Make safe_divide return 0.0 when dividing by zero." --repo "$tmp"
coding-agent run "Make safe_divide return 0.0 when dividing by zero." --repo "$tmp" --single-model
```

Copy a fresh `$tmp` between runs. On the RTX PRO 3000 laptop with `gpu-6gb`, the
multi-model agent fixed it in 10–20 turns (15–35 s), the single-model agent in 3.
