#!/usr/bin/env python3
"""
scripts/finetune_lora.py
LoRA fine-tuning script for BioVoice using Unsloth.

Prerequisites
-------------
    pip install unsloth transformers datasets trl peft

Usage
-----
    # Fine-tune on expert feedback (default: 3 epochs, LoRA rank 16)
    python scripts/finetune_lora.py --data data/feedback/training_data.jsonl

    # Custom base model and hyperparameters
    python scripts/finetune_lora.py \\
        --data  data/feedback/training_data.jsonl \\
        --model unsloth/llama-3-8b-Instruct \\
        --rank  32 \\
        --epochs 5 \\
        --output output/biovoice-lora

    # Export to GGUF for Ollama
    python scripts/finetune_lora.py \\
        --data data/feedback/training_data.jsonl \\
        --export-gguf \\
        --gguf-quant q4_k_m

Then load in Ollama:
    ollama create biovoice-lora -f scripts/Modelfile
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for BioVoice synthesis quality"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training JSONL (ShareGPT format, from FeedbackStore.export_jsonl())",
    )
    parser.add_argument(
        "--model",
        default="unsloth/llama-3-8b-Instruct",
        help="Base model (HuggingFace repo or local path). Default: llama-3-8b-Instruct",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more parameters, more expressive). Default: 16",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling. Default: 16 (equal to rank)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs. Default: 3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size. Default: 2",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps. Default: 4",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate. Default: 2e-4",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length. Default: 2048",
    )
    parser.add_argument(
        "--output",
        default="output/biovoice-lora",
        help="Directory to save the fine-tuned model. Default: output/biovoice-lora",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export to GGUF format after training (for Ollama)",
    )
    parser.add_argument(
        "--gguf-quant",
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q5_k_m"],
        help="GGUF quantisation method. Default: q4_k_m",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print config without training",
    )
    return parser.parse_args()


def validate_data(data_path: str) -> int:
    """Validate the training JSONL and return the record count."""
    if not os.path.exists(data_path):
        print(f"ERROR: training data not found: {data_path}")
        print("  Run: python -c \"from biovoice.finetuning import FeedbackStore; "
              "FeedbackStore().export_jsonl()\"")
        sys.exit(1)

    count = 0
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: invalid JSON on line {i}: {e}")
                sys.exit(1)
            # Validate ShareGPT format
            if "conversations" not in rec:
                print(f"WARNING: line {i} missing 'conversations' key")
            count += 1

    print(f"Data: {count} training examples in {data_path}")
    return count


def train(args):
    """Run LoRA fine-tuning with Unsloth."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("ERROR: Unsloth not installed. Run:")
        print("  pip install unsloth transformers datasets trl peft")
        sys.exit(1)

    print(f"\n--- BioVoice LoRA Fine-tuning ---")
    print(f"Base model  : {args.model}")
    print(f"Training data: {args.data}")
    print(f"LoRA rank   : {args.rank} / alpha {args.alpha}")
    print(f"Epochs      : {args.epochs}")
    print(f"Output      : {args.output}")
    print()

    # 1. Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,          # auto-detect: bfloat16 on Ampere+, float16 otherwise
        load_in_4bit=True,   # QLoRA: quantise base, train adapters in fp16
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves ~30% VRAM
        random_state=42,
    )

    # 3. Load dataset
    records = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Convert ShareGPT conversations to training text
    def format_conversation(rec) -> str:
        convs = rec.get("conversations", [])
        parts = []
        for turn in convs:
            role  = turn["from"]
            value = turn["value"]
            if role == "system":
                parts.append(f"<|system|>\n{value}<|end|>")
            elif role == "human":
                parts.append(f"<|user|>\n{value}<|end|>")
            elif role == "gpt":
                parts.append(f"<|assistant|>\n{value}<|end|>")
        return "\n".join(parts) + tokenizer.eos_token

    texts = [format_conversation(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            output_dir=args.output,
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()

    # 5. Save LoRA weights
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nLoRA adapters saved: {args.output}")

    # 6. Optional GGUF export
    if args.export_gguf:
        gguf_path = os.path.join(args.output, f"biovoice-{args.gguf_quant}.gguf")
        print(f"Exporting GGUF ({args.gguf_quant}) → {gguf_path}")
        model.save_pretrained_gguf(
            args.output,
            tokenizer,
            quantization_method=args.gguf_quant,
        )
        print(f"GGUF saved: {gguf_path}")
        _write_modelfile(args.output, gguf_path)


def _write_modelfile(output_dir: str, gguf_path: str):
    """Write an Ollama Modelfile pointing at the GGUF."""
    modelfile = os.path.join(output_dir, "Modelfile")
    content = f"""\
FROM {gguf_path}

SYSTEM \"\"\"
You are BioVoice, an expert biomedical research assistant specialising in
virology, broadly neutralising antibodies, vaccine design, and related fields.
Cite every factual claim with [PMID:NNNNNNN] in parentheses.
\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
"""
    with open(modelfile, "w") as f:
        f.write(content)
    print(f"\nModelfile written: {modelfile}")
    print("To register with Ollama:")
    print(f"  ollama create biovoice-lora -f {modelfile}")
    print("  ollama run biovoice-lora")


def main():
    args = parse_args()
    count = validate_data(args.data)
    if count < 10:
        print(
            f"WARNING: only {count} training examples. "
            "Fine-tuning typically needs 50+ for meaningful improvement. "
            "Collect more feedback before training."
        )
    if args.dry_run:
        print("\nDry run complete — no training performed.")
        return
    if count < 3:
        print("ERROR: fewer than 3 examples — cannot train. Exiting.")
        sys.exit(1)
    train(args)
    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
