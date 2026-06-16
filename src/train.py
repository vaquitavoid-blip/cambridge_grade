# src/train.py
# Phase 8 — Training Pipeline
# Supports essays, mark schemes, examiner reports, model answers
# RTX 2050 (4GB VRAM) compatible — Qwen2.5-1.5B + LoRA

import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent))
from config import (
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    TRAINING_FILE, EVAL_FILE,
    TRAINING_CONFIG, LORA_CONFIG,
    GRADING_SYSTEM_PROMPT,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — DATASET VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(path: Path) -> dict:
    """
    Validates a JSONL training file before training.
    Checks format, message structure, and content quality.
    """
    if not path.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    errors   = []
    warnings = []
    samples  = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON — {e}")
                continue

            if "messages" not in sample:
                errors.append(f"Line {i}: Missing 'messages' key")
                continue

            msgs = sample["messages"]
            roles = [m.get("role") for m in msgs]
            if "user" not in roles:
                errors.append(f"Line {i}: No user message")
            if "assistant" not in roles:
                errors.append(f"Line {i}: No assistant message")

            # Check assistant response has actual content
            for m in msgs:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    if len(content) < 50:
                        warnings.append(f"Line {i}: Very short assistant response ({len(content)} chars)")
                    if "MARK:" not in content and "mark" not in content.lower():
                        warnings.append(f"Line {i}: Assistant response may not contain a mark")

            samples.append(sample)

    return {
        "valid":    len(errors) == 0,
        "samples":  len(samples),
        "errors":   errors,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    if not torch.cuda.is_available():
        console.print("[red]✗ CUDA not available.[/red]")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    console.print(f"[green]✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)[/green]")

    if vram_gb < 5:
        console.print(f"[yellow]⚠ Only {vram_gb:.1f}GB VRAM — using aggressive memory settings[/yellow]")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model():
    console.print(f"\n[cyan]Loading: {BASE_MODEL_NAME}[/cyan]")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
<<<<<<< HEAD
        llm_int8_enable_fp32_cpu_offload=True,  # allows layers to spill to CPU safely
=======
        llm_int8_enable_fp32_cpu_offload=True,
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
<<<<<<< HEAD
        device_map="auto",               # auto splits layers across GPU + CPU as needed
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
=======
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.print("[green]✓ Base model loaded[/green]")
    _vram()
    return model, tokenizer


def apply_lora(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓ LoRA: {trainable:,}/{total:,} params ({100*trainable/total:.2f}%)[/green]")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(tokenizer):
    if not TRAINING_FILE.exists():
        console.print(f"[red]No training data at {TRAINING_FILE}[/red]")
        console.print("[yellow]Run: python src/data_prep.py[/yellow]")
        sys.exit(1)

    # Validate before training
    console.print("[cyan]Validating training data...[/cyan]")
    val = validate_dataset(TRAINING_FILE)
    if not val["valid"]:
        console.print(f"[red]Dataset validation failed:[/red]")
        for e in val["errors"]:
            console.print(f"  [red]✗ {e}[/red]")
        sys.exit(1)

    if val["warnings"]:
        for w in val["warnings"][:5]:
            console.print(f"  [yellow]⚠ {w}[/yellow]")

    console.print(f"[green]✓ Dataset valid: {val['samples']} samples[/green]")

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_raw = load_jsonl(TRAINING_FILE)
    eval_raw  = load_jsonl(EVAL_FILE) if EVAL_FILE.exists() else []

    console.print(f"[cyan]Train: {len(train_raw)} · Eval: {len(eval_raw)}[/cyan]")

    def fmt(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = Dataset.from_list(train_raw).map(fmt)
    eval_ds  = Dataset.from_list(eval_raw).map(fmt) if eval_raw else None
    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, eval_ds):
    cfg = TRAINING_CONFIG.copy()

    args = TrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_DIR),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=cfg["eval_steps"] if eval_ds else None,
        load_best_model_at_end=cfg["load_best_model_at_end"] if eval_ds else False,
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        optim=cfg["optim"],
        dataloader_pin_memory=cfg["dataloader_pin_memory"],
        group_by_length=cfg["group_by_length"],
        report_to="none",
        remove_unused_columns=True,
        max_grad_norm=0.3,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
    )

    n    = len(train_ds)
    ep   = cfg["num_train_epochs"]
    est  = round(n * ep * 15 / 60)
    console.print(f"\n[bold green]Starting training...[/bold green]")
    console.print(f"[dim]~{est} minutes for {n} samples × {ep} epochs[/dim]\n")
    _vram()

    trainer.train()
    console.print("\n[bold green]✓ Training complete![/bold green]")
    _vram()
    return trainer


def save_model(trainer, tokenizer):
    FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    console.print(f"[green]✓ Saved to: {FINE_TUNED_MODEL_DIR}[/green]")


def _vram():
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    console.print(f"[dim]VRAM: {a:.2f}GB alloc / {r:.2f}GB reserved[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Cambridge Economics AI — Training        [/bold cyan]")
    console.print("[bold cyan]  Qwen2.5-1.5B + LoRA | RTX 2050 4GB      [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════[/bold cyan]\n")

    setup()
    model, tokenizer = load_base_model()
    model            = apply_lora(model)
    train_ds, eval_ds = load_training_data(tokenizer)
    trainer          = train(model, tokenizer, train_ds, eval_ds)
    save_model(trainer, tokenizer)
    console.print("\n[bold]Next:[/bold] streamlit run app/grading_app.py")


if __name__ == "__main__":
    main()