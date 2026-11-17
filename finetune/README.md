# Fine-tuning the glue models

**Fine-tune the weak links, not the coder.** The glue roles are the cheap,
high-leverage targets because their failures are easy to label and they run
small enough to train on a single GPU.

Your training data is already being collected: every run writes
`<repo>/.coding-agent/sessions/<id>.trace.jsonl`, and each `"kind": "model"` line
with `"role": "extractor"` holds the exact prompt (the raw tool output) and the
model's reply. Filter for the replies you would correct, and write the JSON you
wanted. (`coding-agent trace latest --full` shows them in a readable form.)

Recommended order of effort:

1. **extractor (LFM2-350M-Extract)** — the highest-leverage fix. Take real
   tool outputs (tracebacks, pytest logs, `git` output) from your traces and
   label each with the exact JSON you wish the model had produced. A few dozen
   examples measurably improves how much signal survives compression.
2. **dispatcher (LFM2-1.2B-Tool)** — label `(intent -> exact tool call)` pairs
   for any tool calls that came out malformed.

## Workflow

```bash
# 1. Collect/label pairs as JSONL: {"input": "...", "output": "..."}
# 2. Convert to chat format using the SAME system prompt as inference:
python finetune/prepare_data.py --role extractor \
    --in data/extractor_pairs.jsonl --out data/extractor_sft.jsonl

# 3. Train (needs the training stack):
pip install "unsloth" trl datasets transformers accelerate
python finetune/train_lfm2.py \
    --base LiquidAI/LFM2-350M-Extract \
    --data data/extractor_sft.jsonl \
    --out out/extractor-lora

# 4. Export to GGUF and point the role's `gguf` in your profile at it:
#    python llama.cpp/convert_hf_to_gguf.py out/extractor-lora --outfile models/extractor-ft.gguf
#    ./llama.cpp/llama-quantize models/extractor-ft.gguf models/extractor-ft-Q8_0.gguf Q8_0
```

Both LFM2 and Qwen train with Unsloth or TRL. Keep the coder (Qwen2.5-Coder-3B)
frozen unless the eval set shows it is the bottleneck — usually it is not.
