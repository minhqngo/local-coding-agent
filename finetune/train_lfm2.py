#!/usr/bin/env python3
"""LFM2 SFT training scaffold (TRL / Unsloth).

Fine-tunes one of the small glue models (extractor / dispatcher) on chat-format
JSONL produced by ``prepare_data.py``. This is a template: the heavy deps are
imported lazily and guarded so the file imports cleanly without them, and so the
rest of the repo/tests never require a training stack.

    pip install "unsloth" trl datasets transformers accelerate
    python finetune/train_lfm2.py \
        --base LiquidAI/LFM2-350M-Extract \
        --data data/extractor_sft.jsonl \
        --out out/extractor-lora

After training, export to GGUF (llama.cpp `convert_hf_to_gguf.py` + `quantize`)
and point the role's `gguf` in config/models.yaml at the result.
"""

from __future__ import annotations

import argparse


def train(args: argparse.Namespace) -> None:
    # Targets current TRL (>=0.13): TrainingArguments -> SFTConfig, and
    # dataset_text_field / max_seq_length moved onto SFTConfig; tokenizer= is now
    # processing_class=. Imports are lazy so this file loads without the stack.
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - scaffold guard
        raise SystemExit(
            "training deps missing. install with:\n"
            "  pip install trl datasets transformers accelerate  # (optionally unsloth)"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base)

    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_row(row):
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(format_row)

    sft_config = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved fine-tuned model to {args.out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="HF base model id (e.g. LiquidAI/LFM2-350M-Extract)")
    p.add_argument("--data", required=True, help="chat-format JSONL from prepare_data.py")
    p.add_argument("--out", required=True, help="output dir for the fine-tuned model")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    args = p.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
