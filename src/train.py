# src/train.py
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tunes Qwen2.5-1.5B-Instruct on your Cambridge essay dataset
# Tuned for RTX 2050 (4GB VRAM) + 12GB RAM
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from rich.console import Console

sys.path.append(str(Path(__file__).parent))
from config import (
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    TRAINING_FILE, EVAL_FILE,
    TRAINING_CONFIG, LORA_CONFIG,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    """Detect GPU and set memory optimisation environment variables."""
    if not torch.cuda.is_available():
        console.print("[red]✗ CUDA not available. Run the GPU setup steps first.[/red]")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    console.print(f"[green]✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)[/green]")

    if vram_gb < 5:
        console.print(
            f"[yellow]⚠ Only {vram_gb:.1f}GB VRAM — using aggressive memory settings.[/yellow]"
        )

    # Tell PyTorch to release cached memory more aggressively
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

    # Empty cache before starting
    torch.cuda.empty_cache()

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model():
    """Load the base model with 4-bit quantization."""

    console.print(f"\n[cyan]Loading base model: {BASE_MODEL_NAME}[/cyan]")
    console.print("[dim]First run will download ~3GB — this is normal.[/dim]\n")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,     # saves ~0.4GB extra VRAM
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,            # critical for 12GB RAM systems
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.print("[green]✓ Base model loaded[/green]")
    _print_vram_usage()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LORA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model):
    """Apply LoRA adapters for memory-efficient fine-tuning."""

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,   # trades compute for VRAM savings
    )
    model.config.use_cache = False         # required when using gradient checkpointing

    lora_cfg = LoraConfig(**LORA_CONFIG)
    model    = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA applied — training {trainable:,} / {total:,} "
        f"params ({100*trainable/total:.2f}%)[/green]"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(tokenizer):
    """Load and format training + eval JSONL files."""

    if not TRAINING_FILE.exists():
        console.print(
            f"[red]Training data not found at {TRAINING_FILE}[/red]\n"
            "[yellow]Run: python src/data_prep.py  first![/yellow]"
        )
        sys.exit(1)

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_raw = load_jsonl(TRAINING_FILE)
    eval_raw  = load_jsonl(EVAL_FILE) if EVAL_FILE.exists() else []

    console.print(f"[cyan]Training samples:   {len(train_raw)}[/cyan]")
    console.print(f"[cyan]Evaluation samples: {len(eval_raw)}[/cyan]")

    if len(train_raw) == 0:
        console.print("[red]No training samples found. Run python src/data_prep.py first.[/red]")
        sys.exit(1)

    def format_sample(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = Dataset.from_list(train_raw).map(format_sample)
    eval_dataset  = Dataset.from_list(eval_raw).map(format_sample) if eval_raw else None

    return train_dataset, eval_dataset


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_dataset, eval_dataset):
    """Run the fine-tuning loop."""

    cfg = TRAINING_CONFIG.copy()

    training_args = TrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_DIR),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,       # essential for 4GB VRAM
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg["eval_steps"] if eval_dataset else None,
        load_best_model_at_end=cfg["load_best_model_at_end"] if eval_dataset else False,
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        optim=cfg["optim"],
        dataloader_pin_memory=cfg["dataloader_pin_memory"],
        group_by_length=cfg["group_by_length"],
        report_to="none",
        remove_unused_columns=True,
        max_grad_norm=0.3,                 # gradient clipping — stabilises training
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
    )

    n = len(train_dataset)
    epochs = cfg["num_train_epochs"]
    # Very rough estimate: ~15 seconds per sample per epoch on RTX 2050
    est_mins = round(n * epochs * 15 / 60)
    console.print(f"\n[bold green]Starting training...[/bold green]")
    console.print(f"[dim]Estimated time: ~{est_mins} minutes for {n} samples × {epochs} epochs[/dim]\n")

    _print_vram_usage()

    trainer.train()

    console.print("\n[bold green]✓ Training complete![/bold green]")
    _print_vram_usage()
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(trainer, tokenizer):
    """Save the LoRA adapter weights and tokenizer."""
    FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    console.print(f"\n[green]✓ Model saved to: {FINE_TUNED_MODEL_DIR}[/green]")
    console.print("[bold]Next step:[/bold] python scripts/run_grader.py")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    console.print(f"[dim]VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Cambridge Economics Grader — Training     [/bold cyan]")
    console.print("[bold cyan]  Model: Qwen2.5-1.5B  |  GPU: RTX 2050    [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")

    setup()
    model, tokenizer = load_base_model()
    model = apply_lora(model)
    train_dataset, eval_dataset = load_training_data(tokenizer)
    trainer = train(model, tokenizer, train_dataset, eval_dataset)
    save_model(trainer, tokenizer)


if __name__ == "__main__":
    main()