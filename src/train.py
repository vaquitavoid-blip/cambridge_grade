# src/train.py
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tunes Mistral-7B on your Cambridge essay dataset using LoRA (PEFT)
# LoRA means we only train a small fraction of weights → runs on 8GB VRAM
# or even CPU (very slow but possible)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from rich.console import Console
from rich.progress import track

sys.path.append(str(Path(__file__).parent))
from config import (
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    TRAINING_FILE, EVAL_FILE,
    TRAINING_CONFIG, LORA_CONFIG,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_device():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]✓ GPU detected: {gpu} ({vram:.1f} GB VRAM)[/green]")
        return "cuda", True
    else:
        console.print("[yellow]⚠ No GPU found. Training on CPU — will be slow.[/yellow]")
        console.print("[yellow]  Consider using Google Colab (free GPU) for training.[/yellow]")
        return "cpu", False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(use_gpu: bool):
    """Load the base model with 4-bit quantization to save memory."""

    console.print(f"\n[cyan]Loading base model: {BASE_MODEL_NAME}[/cyan]")
    console.print("[dim]First run will download ~4GB — this is normal.[/dim]\n")

    quantization_config = None
    if use_gpu:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if use_gpu else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.print("[green]✓ Base model loaded[/green]")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LORA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model, use_gpu: bool):
    """Wraps the model with LoRA adapters for efficient fine-tuning."""

    if use_gpu:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA applied — training {trainable_params:,} / {total_params:,} "
        f"parameters ({100 * trainable_params / total_params:.2f}%)[/green]"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(tokenizer):
    """Loads JSONL files and formats them into the chat template."""

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAINING_FILE}\n"
            "Run: python src/data_prep.py  first!"
        )

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_raw = load_jsonl(TRAINING_FILE)
    eval_raw  = load_jsonl(EVAL_FILE) if EVAL_FILE.exists() else []

    console.print(f"[cyan]Training samples:   {len(train_raw)}[/cyan]")
    console.print(f"[cyan]Evaluation samples: {len(eval_raw)}[/cyan]")

    def format_sample(sample):
        """Convert messages list → single string using the model's chat template."""
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

def train(model, tokenizer, train_dataset, eval_dataset, use_gpu: bool):
    """Runs the fine-tuning loop."""

    cfg = TRAINING_CONFIG.copy()
    if not use_gpu:
        cfg["fp16"] = False
        cfg["optim"] = "adamw_torch"
        cfg["per_device_train_batch_size"] = 1

    training_args = TrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_DIR),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg["eval_steps"] if eval_dataset else None,
        load_best_model_at_end=cfg["load_best_model_at_end"] if eval_dataset else False,
        fp16=cfg["fp16"],
        optim=cfg["optim"],
        report_to="none",   # Set to "wandb" if you want experiment tracking
        remove_unused_columns=True,
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

    console.print("\n[bold green]Starting training...[/bold green]")
    console.print(f"[dim]Estimated time: {_estimate_time(len(train_dataset), use_gpu)}[/dim]\n")

    trainer.train()

    console.print("\n[bold green]✓ Training complete![/bold green]")
    return trainer


def _estimate_time(n_samples: int, use_gpu: bool) -> str:
    """Very rough time estimate."""
    if use_gpu:
        mins = n_samples * 3 * 0.5  # ~0.5 min per sample per epoch on GPU
    else:
        mins = n_samples * 3 * 15   # ~15 min per sample per epoch on CPU
    if mins > 60:
        return f"~{mins/60:.1f} hours"
    return f"~{int(mins)} minutes"


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(trainer, tokenizer):
    """Saves the LoRA adapter weights and tokenizer."""
    FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    console.print(f"[green]✓ Model saved to: {FINE_TUNED_MODEL_DIR}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Cambridge Economics Grader — Training     [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")

    device, use_gpu = detect_device()

    model, tokenizer = load_base_model(use_gpu)
    model = apply_lora(model, use_gpu)

    train_dataset, eval_dataset = load_training_data(tokenizer)

    if len(train_dataset) == 0:
        console.print("[red]No training data found. Run python src/data_prep.py first.[/red]")
        return

    trainer = train(model, tokenizer, train_dataset, eval_dataset, use_gpu)
    save_model(trainer, tokenizer)

    console.print("\n[bold]Next step:[/bold] python scripts/run_grader.py")


if __name__ == "__main__":
    main()