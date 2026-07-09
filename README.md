# Local Multi-Model Coding Agent

A personal coding agent built entirely from small open-weight models (all ≤ 3B
parameters, with exactly one model at the 3B cap). The division of labour:

> **Qwen writes the code. LFM2 handles the structured glue** (tool dispatch,
> context compression, embeddings). **MiniCPM handles long-context reasoning
> over the repo** (Q&A).

It is fully **config-driven** and ships a deterministic **`--mock` mode**, so the
entire agent loop — and its test suite — runs offline with zero model weights
downloaded. Point it at real `llama-server` (or any OpenAI-compatible) endpoints
when you want to run the models for real.

## The model fleet

| Role | Model | Port | Job |
|---|---|---|---|
| coder | Qwen2.5-Coder-3B-Instruct | 8002 | Writes code. Nothing else. (the single 3B slot) |
| planner | Qwen3.5-2B (thinking) | 8001 | Owns strategy, delegates; never writes code |
| dispatcher | LFM2-1.2B-Tool | 8003 | Formats intents into exact tool calls (temp 0) |
| extractor | LFM2-350M-Extract | 8004 | Compresses raw tool output into small JSON |
| rag | MiniCPM5-1B | 8005 | Repo Q&A over retrieved chunks (131K context) |
| embedder | LFM2-ColBERT-350M | 8006 | Optional dense retrieval |

The repo-Q&A role uses MiniCPM5-1B for its long (131K) native context and strong
reasoning at ~1.2 GB (Q8); the dispatcher, extractor, and embedder use Liquid's
LFM2 models, which are tuned for exactly these structured, non-coding roles.
`config/models.yaml` is the single source of truth for roles, ports, models,
quantization, and context sizes.

## How it works

```
planner → (coder → dispatcher) → sandbox → extractor → planner → …
```

The planner reads the task plus the compressed observations so far and picks the
next move. Code changes are delegated to the coder, whose action is normalised
into an exact tool call. The sandbox executes it; the extractor compresses the
raw output to small JSON before it re-enters the planner's context window — the
single biggest practical win for keeping a 2B planner coherent. When the planner
needs to understand the codebase it asks the RAG role a question instead of
dumping files into its own context.

Hard guardrails: a per-run turn limit and a **mandatory passing-test gate** — a
run can only declare success after the exact test command has been executed and
passed, and any file edit afterwards invalidates that pass.

### Robustness for small models

Small (2–3B) models fail in ways large models don't, so the harness is built to
absorb those failures:

- **Malformed-output recovery** — an unparseable planner/coder response does not
  crash the run. A corrective notice is fed back and the model retries, on a
  *separate* attempt budget so malformed responses never starve real steps.
- **Tagged content blocks** — the coder puts multi-line file content and patch
  text in raw `<content>` / `<old_text>` / `<new_text>` tags rather than escaping
  it inside a JSON string (the most common small-model failure). Plain JSON is
  still accepted.
- **Anchored `patch_file`** — surgical, exactly-once edits to existing files;
  cheaper and safer than whole-file rewrites. `write_file` remains for new files.
- **Repeated-call breaker** — a third identical tool call in a row is refused
  (the exact test command is exempt), so the agent can't stall in a loop.
- **Workspace snapshot** — the first prompt carries a one-time git
  branch/status/log and project-file heads, so the agent doesn't spend turns
  rediscovering the repo.

## Installation

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Runtime dependencies are `requests` and `PyYAML`. The offline test suite needs
only the standard library plus PyYAML (no pytest required).

## Quick start (offline, no downloads)

`--mock` replaces the entire model fleet with a deterministic offline stand-in.
The loop, tools, sandbox, extraction, retrieval, and test gate are all the real
code paths — only the model calls are stubbed — so you can exercise the whole
system without downloading a single weight.

```bash
# Run the full test suite:
python -m unittest discover -t . -s tests

# Drive the whole multi-model loop offline on the bundled example:
tmp=$(mktemp -d) && cp examples/trivial_bug/calc.py examples/trivial_bug/test_calc.py "$tmp"/
python -m coding_agent --repo "$tmp" --mock --verbose \
    run "Make safe_divide return 0.0 when dividing by zero." \
    --mock-script examples/trivial_bug/fix.json

# Offline evaluation harness:
python eval/run_eval.py --mock
```

## Serving the models

```bash
pip install -U "huggingface_hub[cli]"
./scripts/download_models.sh          # download GGUFs → ./models/
./scripts/serve.sh start              # one llama-server per role/port
./scripts/healthcheck.sh              # curl every endpoint
```

`./scripts/serve.sh stop` shuts everything down. Set `NGL=0` for CPU-only, or
serve a subset on GPU and the rest on CPU:

```bash
NGL=99 ./scripts/serve.sh start coder planner rag         # generators on GPU
NGL=0  ./scripts/serve.sh start dispatcher extractor embedder   # glue on CPU
```

Each role's served context window comes from its `ctx` in `config/models.yaml`.
If a draft GGUF is configured for the coder, its server auto-wires speculative
decoding.

### Serving on a different machine

The agent and the models don't have to share a machine: only model calls go over
HTTP — the sandbox, repo, retrieval index, and sessions all stay on the machine
running the agent. To serve models on one box (e.g. the one with the GPU) and
run the agent on another:

**On the serving machine** — bind to the network instead of localhost:

```bash
HOST=0.0.0.0 ./scripts/serve.sh start
```

**On the agent machine** — point the endpoints at the serving machine's IP in
`config/models.yaml`:

```yaml
defaults:
  base_url_template: "http://192.168.1.50:{port}/v1"   # serving machine's IP
```

(Individual roles can also set an explicit `base_url`, so a mixed local/remote
fleet works too.) Verify connectivity from the agent machine, then run as usual:

```bash
HOST=192.168.1.50 ./scripts/healthcheck.sh
python -m coding_agent --repo ./yourproject run "fix the failing test in foo.py"
```

**Security note:** llama-server endpoints have no authentication — anyone on the
network can use them. `HOST` defaults to `127.0.0.1` (local only) for this
reason. Only use `0.0.0.0` on a network you trust, open the ports narrowly
(e.g. `ufw allow from 192.168.1.0/24 to any port 8001:8006 proto tcp`), and give
the serving machine a static IP. On shared networks, prefer an SSH tunnel
(`ssh -L 8001:localhost:8001 -L 8002:localhost:8002 ... serving-machine`), which
needs no `HOST` change and keeps the default localhost config on both ends.

## Running the agent

With models served, drop `--mock`:

```bash
python -m coding_agent --repo ./yourproject run "fix the failing test in foo.py"
```

Add `--dispatch-all` to route *every* tool call through the LFM2 dispatcher
model; by default clean coder actions are mapped deterministically to save a hop.

### Repo-aware Q&A

```bash
python -m coding_agent --repo ./yourproject run "..." --enable-retrieval
python -m coding_agent --repo ./yourproject ask "where is auth handled?"
```

The codebase is chunked and indexed, and the index is persisted under `.index/`
(keyed by file signatures) so an unchanged repo reloads instantly. Ranking is
dependency-free TF-IDF by default; supplying a dense embedding function switches
it to dense cosine (e.g. an LFM2-ColBERT or MiniCPM-Embedding endpoint). The
`search_codebase` tool lets the agent ask questions instead of loading whole
files into context.

### Single-model mode

The coder can also run alone — no planner, extractor, or dispatcher — driving its
own read → edit → test → finish loop:

```bash
python -m coding_agent --repo ./yourproject run "fix foo.py" --single-model
```

### Evaluation

Tasks live in `eval/tasks/` as YAML files, each declaring setup files, a prompt,
an independent verification command, and (for offline runs) a scripted fix. A
task passes only if its verification command succeeds after the agent finishes —
the agent's own success claim is never trusted.

```bash
python eval/run_eval.py               # full multi-model system
python eval/run_eval.py --baseline    # coder-only mode, same tasks
```

## CLI reference

Global flags (`--config` / `--repo` / `--mock` / `--verbose`) work **before or
after** the subcommand.

```
python -m coding_agent [--repo PATH] [--config FILE] [--mock] [--verbose] <command>

  run  [task]   --mock-script FILE   --max-turns N   --backend {subprocess,docker}
                --enable-retrieval    --single-model  --dispatch-all
                --confirm             --resume latest|ID   --no-session
  ask  <question>
```

- `--confirm` prompts before each `write_file` / `patch_file` / `run_shell`
  (and never prompts for a call that would fail validation).
- `--single-model` runs the coder-only mode.
- Every `run` persists a resumable session under `.coding-agent/sessions/` after
  each turn; `--resume latest` (or `--resume <id>`) continues an interrupted run
  (the task is optional when resuming). `--no-session` disables persistence.

## Sandbox & safety

Every shell command runs through a sandbox with a best-effort denylist, a hard
timeout, and output truncation. File tools refuse paths that escape the working
directory.

- **`subprocess`** (default) pins the working directory to the repo. Fast; the
  denylist is a convenience layer, **not** a security boundary.
- **`docker`** runs commands in one long-lived `--network none` container per run
  (state persists between commands; the container is force-removed on exit).
  Because `--network none` blocks installs at run time, bake your test
  dependencies into the image first:
  ```bash
  ./scripts/build_sandbox_image.sh          # builds coding-agent-sandbox:latest
  # then set sandbox.docker_image + backend: docker in config/models.yaml
  ```

The denylist matches on command *name* (so `grep -r shutdown` is allowed while
`shutdown now` is refused) and includes a dedicated destructive-`rm` detector.

## Configuration

All behaviour is set in `config/models.yaml`:

- **Per role:** `model`, `port`, `gguf`, `hf_repo`, `temperature`, `max_tokens`,
  `ctx` (served context window), and `thinking` (sends the model's thinking mode
  for reasoning-capable models). An optional `draft_gguf` on the coder enables
  speculative decoding.
- **`sandbox`:** `backend` (`subprocess`/`docker`), `docker_image`,
  `command_timeout_s`, `denylist`.
- **`loop`:** `max_turns`, `test_command`, `require_passing_tests`,
  `dispatch_all`.
- **`retrieval`:** `chunk_lines`, `chunk_overlap`, `top_k`, `index_dir`,
  `include_globs`, `exclude_dirs`.

Set `mock: true` (or `CODING_AGENT_MOCK=1`, or `--mock`) to run fully offline.

## Expectations

Realistic scope for a 2–3B fleet: small bug fixes, writing tests, boilerplate,
single-file refactors, and repo Q&A — **not** autonomous multi-file feature
development. Quantized to Q4/Q8 the whole system is roughly 4–6 GB and can run
CPU-only or on a modest GPU. Use the evaluation harness to measure the full
multi-model system against single-model mode on your own tasks.

## Licensing

Qwen3.5 (Apache-2.0), LFM2 (free under Liquid's terms until a company passes $10M
annual revenue), and MiniCPM5-1B (Apache-2.0). Qwen2.5-Coder-3B ships under the
Qwen Research (non-commercial) license — fine for personal use. This harness code
is MIT.
