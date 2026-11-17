# Exercises

Hands-on ways to learn how the agent works by changing it. The first five run
offline (`--mock`, no GPU); the rest need served models. After each change run
`python -m unittest discover -t . -s tests` — a failing test usually tells you
which guardrail you just removed.

A scratch copy of the example bug, used below:

```bash
tmp=$(mktemp -d) && cp examples/trivial_bug/calc.py examples/trivial_bug/test_calc.py "$tmp"/
```

## Offline

**1. Read a trace.** Run the mock example and open the trace with `--full`:

```bash
coding-agent run "Make safe_divide return 0.0 when dividing by zero." \
    --repo "$tmp" --mock --mock-script examples/trivial_bug/fix.json
coding-agent trace latest --repo "$tmp" --full
```

Find the planner's first prompt. Which sections does it have, in what order, and
why is the observation block last? (Hint: `Planner.decide` in `roles.py`.)

**2. Remove the test gate.** `test_premature_finish_is_blocked` in
`tests/test_loop.py` forces a planner that says "finish" on every turn while the
bug is still there. Add `config.loop.require_passing_tests = False` to that test
and run it:

```bash
python -m unittest tests.test_loop.LoopTest.test_premature_finish_is_blocked
```

The run now "succeeds" on turn 1 with failing tests. Small models declare victory
like this all the time; what would an eval that trusted the agent's claim have
recorded? Undo the change.

**3. Make the extractor lie.** `tests/test_loop_behaviour.py` has a test where the
extractor claims `"tests_passed": true` about a failing command. Find the line in
`roles.Extractor.compress` that deletes that claim and remove it. Which tests
fail, and what would the planner have seen?

**4. Write an eval task.** Copy `eval/tasks/patch_off_by_one.yaml` and make a task
of your own: a small bug, a test that catches it, a `mock_actions` fix. Check it
with `python eval/run_eval.py --mock --tasks <name>`. Then write a deliberately
wrong `mock_actions` fix and confirm the task fails even though the agent says
"success" — the verify command is what counts.

**5. Add a tool.** Add `list_files(pattern)` to `tools.py` (a glob under the
repo, at most 100 results). Register it in `TOOLS`, mention it in the planner's
system prompt, and write a test. Notice everything you *didn't* have to touch:
validation, confirmation, tracing and stall detection come from the harness.

## With models

**6. Thinking on vs off.** Run the eval with the planner's `thinking` on and off
(edit the profile) and compare time and pass rate:

```bash
python eval/run_eval.py --profile gpu-6gb
```

Then open a trace and read the `reasoning` blocks. How much of it changes the
decision?

**7. Single vs multi.** The research question of this repo. Run both agents on the
same tasks, three times each:

```bash
python eval/run_eval.py --profile gpu-6gb
python eval/run_eval.py --profile gpu-6gb --single-model
```

Look at `calls=` and `tokens=` as well as the pass rate. Where does the team
spend its extra calls? Is any of it useful?

**8. Break the prompt on purpose.** In `roles._CODER_FORMAT`, put the patch example
back on one line (`header {...} + <old_text>…</old_text><new_text>…</new_text>`)
and run the single-model eval. Then read `docs/how-it-works.md`, lesson 5. Small
models copy the shape of examples; this is the cheapest way to feel it.

**9. Turn off the stall detector.** Set `max_repeated_outcomes: 0` and give the
agent an impossible task (`"fix foo.py"` in an empty directory). How many turns
does it waste? Put it back and compare.

**10. Squeeze the context.** Halve the coder's `ctx` in the profile and re-run
`coding-agent doctor` to see the VRAM saved. Then run the eval. `CoderMemory`
shrinks its budget automatically — at what point does the coder stop seeing the
files it needs?

**11. Fit a bigger model.** On the work laptop, download the `gpu-12gb-moe`
profile's model and find the lowest `cpu_moe_layers` that `doctor` says fits.
Measure tok/s at that value and at +4 with `doctor --bench`, then compare pass
rates against `gpu-12gb`.

**12. Fine-tune the extractor.** Traces contain every (tool output → extractor
JSON) pair from your runs. Collect the bad ones, write the JSON you wanted, and
follow `finetune/README.md`.
