# Trivial-bug example (Phase 2 end-to-end)

`calc.py` has a one-line bug: `safe_divide(1, 0)` raises instead of returning
`0.0`. `test_calc.py` catches it. This is the smallest end-to-end run.

## Offline (no models needed)

`fix.json` is a scripted coder action, so the mock brain drives the real loop —
planner → coder → dispatcher → sandbox → extractor → planner — and actually
writes the fix to disk, then the passing-test gate verifies it.

> Note: the agent edits files in place. Copy the example to a scratch dir first
> so you can re-run it.

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo ../..)"   # repo root
tmp=$(mktemp -d) && cp examples/trivial_bug/* "$tmp"/
python -m coding_agent --repo "$tmp" --mock --verbose \
    run "Make safe_divide return 0.0 when dividing by zero." \
    --mock-script examples/trivial_bug/fix.json
python -m pytest -q "$tmp"     # confirm the fix stuck
```

(Set `PYTHONPATH=src` or `pip install -e .` first so `coding_agent` imports.)

## Real models

Start the servers, then drop `--mock`/`--mock-script`:

```bash
./scripts/serve.sh start && ./scripts/healthcheck.sh
python -m coding_agent --repo "$tmp" run "Fix the zero-division bug in calc.py"
```
