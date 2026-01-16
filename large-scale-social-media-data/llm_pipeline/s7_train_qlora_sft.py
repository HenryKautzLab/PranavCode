#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def format_messages(ex, tokenizer):
    """
    Convert {"messages":[{role,content},...]} into a single text string.
    Uses a simple chat template that works OK for most instruct models.
    If you want model-specific templates later, we can swap this.
    """
    msgs = ex["messages"]
    parts = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
        else:
            parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model id or local path")
    ap.add_argument("--train_jsonl", required=True, help="JSONL with {'messages':[...]}")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA load
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=True,  # A4000 doesn't do bf16 well; if crashes, set bf16=False and fp16=True in sbatch env
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
    )

    def formatting_func(ex):
        return format_messages(ex, tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA adapter + tokenizer to: {args.output_dir}")

if __name__ == "__main__":
    main()
