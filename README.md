# Local Coding Agent

A small coding agent that runs entirely on a laptop, written to be read. It fixes
bugs in a repository by reading files, editing them and running the tests, using
local open-weight models served by llama.cpp.

There are two agents in here, and they share one harness:

- **single-model** — one coder model decides everything
  (`src/coding_agent/single_model.py`, about 100 lines of loop).
- **multi-model** — a team of specialists: a planner, a coder, a dispatcher, an
  extractor and a repo-QA model (`src/coding_agent/loop.py`).

Both run on the same **harness** (`src/coding_agent/harness.py`), which executes
tool calls and enforces the rules: tests must pass before "done", exit codes are
facts no model can overrule, stuck runs get stopped. Comparing the two agents on
the same tasks is the experiment this repo is built for.

Hardware profiles are included for a **6 GB GPU / 16 GB RAM** laptop (GTX 1660 Ti
class) and a **12 GB GPU** laptop (RTX PRO 3000 class), plus a launcher that
checks a fleet of models fits in VRAM before starting it.

## Learning path

1. Run it offline (below) and look at the trace.
2. Read [`docs/how-it-works.md`](docs/how-it-works.md): the loop, the harness, and
   ten failures from real runs that shaped the design.
3. Run it on your GPU with [`docs/running-locally.md`](docs/running-locally.md).
4. Do some of the [`docs/exercises.md`](docs/exercises.md).

Suggested reading order in `src/coding_agent/`: `single_model.py` → `harness.py` →
`tools.py` → `sandbox.py` → `llm.py` → `roles.py` → `loop.py` → `trace.py`.

## Quick start (offline, no models)

`--mock` replaces every model with a deterministic script; everything else (loop,
tools, sandbox, test gate, traces) is the real code.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e .

python -m unittest discover -t . -s tests       # ~150 tests, no models needed
python eval/run_eval.py --mock                  # the eval harness, scripted
```

Drive the agent through the bundled bug, then see what each "model" saw:

```bash
# bash / Git Bash
tmp=$(mktemp -d) && cp examples/trivial_bug/calc.py examples/trivial_bug/test_calc.py "$tmp"/
coding-agent run "Make safe_divide return 0.0 when dividing by zero." \
    --repo "$tmp" --mock --mock-script examples/trivial_bug/fix.json -v
coding-agent trace latest --repo "$tmp"
```

```powershell
# PowerShell
$tmp = Join-Path $env:TEMP ([guid]::NewGuid()); New-Item -ItemType Directory $tmp | Out-Null
Copy-Item examples\trivial_bug\calc.py, examples\trivial_bug\test_calc.py $tmp
coding-agent run "Make safe_divide return 0.0 when dividing by zero." `
    --repo $tmp --mock --mock-script examples\trivial_bug\fix.json -v
coding-agent trace latest --repo $tmp
```

(`coding-agent` and `python -m coding_agent` are the same thing.)

## Running with real models

You need a GPU build of llama.cpp's `llama-server` — see
[`docs/running-locally.md`](docs/running-locally.md) for each laptop. Then, once
per machine:

```powershell
$env:LLAMA_SERVER = "C:\path\to\llama-server.exe"   # or put it on PATH
$env:CODING_AGENT_PROFILE = "gpu-6gb"              # or gpu-12gb on the work laptop
```

```bash
coding-agent download            # GGUF weights for the profile → models/
coding-agent doctor              # binary, GPU, files, VRAM budget
coding-agent serve start         # one llama-server per model, checked to fit
coding-agent doctor --bench      # tokens/s per server
coding-agent run "fix the failing test in calc.py" --repo path/to/project
coding-agent run "..." --repo path/to/project --single-model
coding-agent serve stop
```

## Profiles

| Profile | For | Models (role) | VRAM | Status |
|---|---|---|---|---|
| `gpu-6gb` (default) | GTX 1660 Ti, 16 GB RAM | Qwen2.5-Coder-3B (coder), Qwen3.5-2B (planner, shared by rag + dispatcher), LFM2-350M on CPU (extractor) | 4.2 GiB measured | tested |
| `gpu-12gb` | RTX PRO 3000, 12 GB | + MiniCPM5-1B (rag), LFM2-1.2B-Tool on CPU (dispatcher), 0.5B draft model, planner thinking on | 6.3 GiB measured | tested |
| `gpu-6gb-9b` | GTX 1660 Ti | Ornith-1.5-9B (all roles but extractor) | ≈5.3 GiB est. | experimental |
| `gpu-12gb-moe` | 12 GB + 64 GB RAM | Qwen3.6-35B-A3B MoE, experts in RAM (all roles) | ≈10 GB reported on similar 12 GB cards | experimental |

The experimental profiles use newer, stronger models picked in a November 2026
survey (sources in `docs/running-locally.md`); their files are verified to exist
but they have not been benchmarked on the target laptops yet.

## Measured results

The three bundled eval tasks, real models, RTX PRO 3000 laptop, `--max-turns 15`
(small sample, nonzero temperature — treat as indicative):

| Agent | Profile | Original code | Final code (pass rate, time per run) |
|---|---|---|---|
| multi-model | gpu-12gb | 1/3 | 2/3 (142 s) |
| single-model | gpu-12gb | 0/3 | 2/3 (19 s) |
| multi-model | gpu-6gb | — | 2/3, 3/3, 2/3 over three runs (27–56 s) |
| single-model | gpu-6gb | — | 3/3, 2/3, 2/3 over three runs (15–23 s) |

Every gain came from reading traces and fixing the harness or the prompts, not
from changing models; `docs/how-it-works.md` walks through each one. The task
that still fails is `implement_slugify`: it needs a correct regex, and the 3B
coder rarely gets there — a model limit, which is what the experimental
profiles' larger models are for.

## Commands

```
coding-agent [--profile NAME|FILE] [--repo DIR] [--mock] [-v] <command>

  run [TASK]   --single-model --test-command CMD --max-turns N --confirm
               --enable-retrieval --dispatch-all --backend subprocess|docker
               --resume latest|ID --no-session --mock-script FILE
  ask QUESTION                         repo Q&A over retrieved chunks
  trace [latest|ID|FILE] [--full]      what every model saw and said
  serve start|stop|status [ROLE...] [--cpu] [--host H] [--force]
  doctor [--bench]                     binary, GPU, files, VRAM budget, speed
  download [ROLE...]                   fetch the profile's GGUF files
```

Every run writes `<repo>/.coding-agent/sessions/<id>.json` (resumable state) and
`<id>.trace.jsonl` (every prompt, reply, tool call and exit code).

## Configuration

- `config/agent.yaml` — behaviour: test command, turn limit, stall limit,
  sandbox, retrieval. Same on every machine.
- `config/profiles/*.yaml` — hardware: which model serves which role, port,
  `gpu_layers`, `ctx`, `cpu_moe_layers`, `share: <role>` to reuse another role's
  server, `thinking`, sampling. A profile can override anything in `agent.yaml`.

Choose with `--profile` or `CODING_AGENT_PROFILE`; set `CODING_AGENT_MOCK=1` to
force offline mode. Any OpenAI-compatible server works (Ollama, LM Studio, vLLM):
set `base_url` and `model` per role; `serve`/`doctor` only manage llama-server.

## Safety

Commands run through a sandbox with a timeout that kills the whole process tree,
stdin closed, and a best-effort denylist (`rm -rf /`, `mkfs`, `shutdown`, …). The
default `subprocess` backend is **not** a security boundary — it runs in your
repo with your user's rights. For untrusted code use the docker backend
(`--backend docker`, image from `scripts/build_sandbox_image.sh`), which runs
with `--network none`. `--confirm` asks before every edit and command.

llama-server has no authentication; profiles bind to `127.0.0.1`. See
`docs/running-locally.md` before serving to another machine.

## Licensing

This code is MIT. Model licenses: Qwen3.5, Qwen3.6, MiniCPM5 — Apache-2.0;
Ornith-1.5 — MIT; Qwen2.5-Coder-3B — Qwen Research license (non-commercial);
LFM2/LFM3 — LFM Open License, commercial use only for organisations under
$10M annual revenue. The `gpu-12gb` profile uses LFM models: fine for personal
projects, not for work at a larger company (use `gpu-12gb-moe` there).
