# Running locally

How to run the agent with real models on a laptop GPU: getting `llama-server`,
picking a profile, where the memory goes, and which models are worth using as of
November 2026.

## 1. Get a GPU build of llama.cpp

The agent talks to models over HTTP; `coding-agent serve` starts one
`llama-server` process per model. You need a build that matches your GPU.

**Work laptop (RTX PRO 3000, Blackwell).** The existing build at
`%USERPROFILE%\llama.cpp\build\bin\Release\llama-server.exe` is CUDA-enabled
and works (build 10558, August 2026). It was compiled with `GGML_NATIVE=ON`,
which targets only the GPU it was built on, so don't copy it to the other
laptop. Newer models (LFM3) need build b11960 or later; for a prebuilt
replacement, take the Windows **CUDA 13** zip from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), which
contains native Blackwell (sm_120) code.

**Personal laptop (GTX 1660 Ti, Turing, sm_75).** Download the Windows CUDA zip
(`llama-<build>-bin-win-cuda-<ver>-x64.zip`) plus the matching `cudart-…` zip
into one folder. Prebuilt CUDA builds include Turing as PTX, so the driver
compiles the kernels on first start (expect a slow first launch). Alternatives:
build from source with `-DCMAKE_CUDA_ARCHITECTURES=75`, or the Vulkan zip, which
works on any GPU without the CUDA runtime. Measure both with `doctor --bench`
if unsure; no published CUDA-vs-Vulkan numbers exist for this card.

Then, once per machine (PowerShell):

```powershell
$env:LLAMA_SERVER = "C:\path\to\llama-server.exe"      # or add its folder to PATH
$env:CODING_AGENT_PROFILE = "gpu-6gb"                 # personal laptop
# $env:CODING_AGENT_PROFILE = "gpu-12gb"              # work laptop
# To persist: [Environment]::SetEnvironmentVariable("CODING_AGENT_PROFILE","gpu-6gb","User")
```

Check that the GPU is visible to the build:

```bash
coding-agent doctor
#   llama-server: C:\...\llama-server.exe
#     devices : CUDA0: NVIDIA GeForce GTX 1660 Ti (6144 MiB, ...)
```

If `devices` is empty you have a CPU-only build: everything will work, slowly.

## 2. Download, start, measure

```bash
coding-agent download          # the profile's GGUF files → models/
coding-agent serve start       # refuses to start if the VRAM estimate doesn't fit
coding-agent doctor --bench    # health + tokens/s per server
```

`serve start` starts each server with the profile's `-ngl`, `-c`, `--parallel 1`,
waits until it answers `/health`, and prints the end of its log
(`runs/<role>.log`) if it dies. `serve status` and `serve stop` do what they say;
`serve start --cpu` puts every model on the CPU.

## 3. Where the memory goes

A served model needs roughly:

```
weights        ≈ size of the .gguf file
+ KV cache     = ctx × (bytes of K and V per token, summed over attention layers)
+ buffers      ≈ 0.25 GiB per server
```

The KV cache is the part people underestimate. Per token it costs
`2 (K and V) × kv_heads × head_dim × 2 bytes (f16)` for every layer that uses
attention. `gguf.py` reads those numbers from the file header:

| Model | Attention layers | KV per token | KV at 16K ctx |
|---|---|---|---|
| Qwen2.5-Coder-3B | 36 of 36 | 36 KiB | 576 MiB |
| MiniCPM5-1B | 24 of 24 | 24 KiB | 384 MiB |
| Qwen3.5-2B (hybrid) | 6 of 24 | 12 KiB | 192 MiB |
| LFM2-350M (hybrid) | 6 of 16 | 12 KiB | 192 MiB |
| LFM2-ColBERT-350M at its default 128K ctx | 6 of 16 | 12 KiB | 1.5 GiB |

Hybrid models (Qwen3.5's linear-attention layers, LFM2's convolutions) only keep
a KV cache on a few layers, which is why they are attractive on small GPUs. The
last row is the unused embedder from the original setup — 1.5 GiB of VRAM for
nothing.

`coding-agent doctor` prints this budget per server and in total, and compares it
with your GPU. Against `nvidia-smi`: the `gpu-6gb` estimate was exact (4.2 GiB),
the `gpu-12gb` one 0.8 GiB high (7.1 estimated, 6.3 measured) — it errs on the
safe side.

**Why it matters so much on Windows.** When VRAM runs out, the NVIDIA driver
doesn't fail; it quietly spills into shared system memory. The original setup
did exactly that (≈12.7 GiB on an 11.9 GiB card) and its 2B planner generated
9–20 tok/s — slower than the same model on the CPU (33 tok/s).

### Mixture-of-experts on a small GPU

A mixture-of-experts (MoE) model like Qwen3.6-35B-A3B has 35B parameters but only
about 3B active per token: each layer has many expert FFNs and a router picks a
few. The experts are most of the file but each is touched rarely, so llama.cpp
can keep them in system RAM (`--n-cpu-moe N` keeps the experts of the first N
layers there) while attention stays on the GPU. Set `cpu_moe_layers: N` in a
profile; `doctor` reads the tensor table from the GGUF to show exactly how many
GiB stay in RAM. Lower N until it no longer fits: more experts on the GPU means
faster generation.

## 4. Measured speed

RTX PRO 3000 Blackwell laptop (12 GB), llama.cpp b10558 CUDA, 128-token greedy
generation (`doctor --bench`):

| Model | Placement | tok/s |
|---|---|---|
| Qwen2.5-Coder-3B Q4_K_M | GPU (16K ctx, 0.5B draft) | 90 |
| Qwen2.5-Coder-3B Q4_K_M | CPU, 16 threads | 29 |
| Qwen3.5-2B Q5_K_M | GPU | 110 |
| Qwen3.5-2B Q5_K_M | CPU | 33 |
| MiniCPM5-1B Q8_0 | GPU | 224 |
| LFM2-350M-Extract Q8_0 | CPU | 136 |
| LFM2-1.2B-Tool Q8_0 | CPU | 50 |

The GTX 1660 Ti has less than half the memory bandwidth (288 vs 672 GB/s), and
token generation is bandwidth-bound, so expect roughly 35–45 % of the GPU numbers
there. Run `doctor --bench` on it and fill in your own table.

## 5. Profiles

| Profile | Servers | VRAM | Use when |
|---|---|---|---|
| `gpu-6gb` | coder 3B (GPU), planner 2B (GPU, also rag + dispatcher), extractor 350M (CPU) | 4.2 GiB measured | the personal laptop; also the fastest on the work laptop |
| `gpu-12gb` | the full team, 5 servers, planner thinking on | 6.3 GiB measured | experimenting with all five specialists |
| `gpu-6gb-9b` | one 9B model on the GPU for every role, extractor on CPU | ≈5.3 GiB est. | experimental upgrade for the personal laptop |
| `gpu-12gb-moe` | one 35B MoE for every role, experts in RAM | ≈10 GB reported | experimental upgrade for the work laptop (64 GB RAM) |

A role with `share: planner` sends its prompts to the planner's server but keeps
its own system prompt, temperature, token limit and thinking setting. That is
how five roles fit into three servers (or one).

## 6. Choosing models

The tested profiles use the models this project started with. A survey in
November 2026 found much stronger options that fit both laptops. Every file
below was checked to exist on Hugging Face; benchmark numbers are as reported by
their authors unless noted.

**Work laptop (12 GB VRAM, 64 GB RAM): Qwen3.6-35B-A3B** (Apache-2.0, April 2026),
`unsloth/Qwen3.6-35B-A3B-GGUF` → `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.1 GB), with
experts in RAM. Qwen reports SWE-bench Verified 73.4 and Terminal-Bench 2.0 51.5
([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)); the independent
[SWE-rebench](https://swe-rebench.com/) puts it at 24.7 %, which says more about
how much harnesses differ than about the model. A community report on a 12 GB
RTX 3060 measured 38.9 tok/s with `--n-cpu-moe 24` using 9.8 GB of VRAM
([smeltcore](https://smeltcore.com/recipes/qwen3-6-35b-a3b-on-rtx-3060-38-9-tok-s-from-a-12gb-card-via-moe-expert-cpu-offload/)).
Profile: `gpu-12gb-moe`. Strong alternatives worth an A/B test: the dense
Qwen3.8-27B at 3-bit (fits only just), Ornith-1.5-35B-A3B and
KAT-Coder-V2.5-Dev (both coding fine-tunes of the same MoE family, self-reported
scores).

**Personal laptop (6 GB VRAM, 16 GB RAM): Ornith-1.5-9B** (MIT, August 2026), a
coding fine-tune of Qwen3.5-9B: `bartowski/Ornith-1.5-9B-GGUF` →
`Ornith-1.5-9B-IQ4_XS.gguf` (5.2 GB). Self-reported SWE-bench Verified 70.6
([model card](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF)); not yet
independently verified. Only 8 of its 32 layers keep a KV cache, so with a q8_0
KV cache it fits in 6 GB at 8K context. Profile: `gpu-6gb-9b`. If it runs out of
memory use `Ornith-1.5-9B-IQ3_M.gguf` (4.7 GB). Alternatives: Qwen3.5-9B
(`unsloth/Qwen3.5-9B-GGUF`, IQ4_XS 5.2 GB) or, for speed, Qwen3.5-4B
(`unsloth/Qwen3.5-4B-GGUF`, Q5_K_M 3.1 GB). A 35B MoE doesn't make sense with
16 GB of RAM.

**Glue roles.** LFM3-1.7B (October 2026) is Liquid's successor for tool calling
and extraction (BFCL v4 71.4, IFEval 96.7 on its
[card](https://huggingface.co/LiquidAI/LFM3-1.7B)); it needs llama.cpp b11960+.
MiniCPM5-2B (`openbmb/MiniCPM5-2B-GGUF`, Apache-2.0) is the long-context choice
for repo Q&A.

**Licenses.** Apache-2.0: Qwen3.5/3.6/3.8, MiniCPM5. MIT: Ornith. Qwen Research
(non-commercial): Qwen2.5-Coder-3B. LFM Open License: LFM2 and LFM3, free for
commercial use only for organisations under $10M annual revenue — fine on a
personal laptop, not for work at a larger company. The `gpu-12gb` profile uses
two LFM2 models; `gpu-12gb-moe` uses none.

**Retire the 0.5B draft model** when you move to Qwen3.5-family models: its
vocabulary (151,936 tokens) doesn't match theirs (248,320), so it can't be used
for speculative decoding.

## 7. Thinking mode

Qwen3.5 (4B and up), Qwen3.6, Qwen3.8, Ornith and LFM3 think by default. In an
agent loop that multiplies the tokens per step several times over, so all
profiles set `thinking: false` except the `gpu-12gb` planner. The setting is sent
per request (`chat_template_kwargs.enable_thinking`), so roles sharing one model
can differ. If a thinking role returns an empty answer, it used its whole
`max_tokens` reasoning; the error says so — raise `max_tokens` or turn thinking
off.

## 8. Serving models from another machine

Only model calls go over HTTP; the sandbox, repo and traces stay on the machine
running the agent. On the serving machine:

```bash
coding-agent serve start --host 0.0.0.0
```

On the agent machine, point the profile at it:

```yaml
defaults:
  base_url_template: "http://192.168.1.50:{port}/v1"
```

llama-server has **no authentication**: anyone who can reach the port can use it.
Only do this on a network you trust and open the ports narrowly, or better, keep
the default `127.0.0.1` on both ends and use an SSH tunnel
(`ssh -L 8001:localhost:8001 -L 8002:localhost:8002 … serving-machine`).

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `serve start` refuses: "would overflow VRAM" | `doctor`; use a smaller profile, lower `ctx`, raise `cpu_moe_layers`, close other GPU apps, or `--cpu` |
| A server exits during start | the log tail is printed; full log in `runs/<role>.log` (wrong path, unsupported architecture = llama.cpp too old) |
| Generation is slow | `doctor --bench`; under 25 tok/s on GPU means spill to shared memory or a CPU-only build |
| `✗ model servers not reachable` | `serve status`; the run checks every endpoint it needs before starting |
| `ModelUnavailable … exceeds the available context size` | raise that role's `ctx` |
| Run stops with "the agent is stuck" | `coding-agent trace latest` and read the last few steps; it is usually a misunderstanding visible in the prompt |
| Tests fail with `No module named pytest` | the workspace snapshot's `python:` line shows which interpreter runs; give the project a `.venv` with its test deps |
