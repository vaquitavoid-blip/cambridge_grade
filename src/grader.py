# src/grader.py
# ─────────────────────────────────────────────────────────────────────────────
# The core grading engine.
# Loads either the fine-tuned model (if trained) or falls back to the
# base model with a detailed system prompt for zero-shot grading.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

sys.path.append(str(Path(__file__).parent))
from config import (
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    GRADING_SYSTEM_PROMPT, GRADING_PROMPT_TEMPLATE,
)

console = Console()


class CambridgeGrader:
    """
    Loads the grading model and provides a grade() method.

    Usage:
        grader = CambridgeGrader()
        result = grader.grade(
            question="Evaluate the effectiveness of fiscal policy in reducing unemployment.",
            essay="Fiscal policy refers to...",
            level="AS",
            max_marks=12,
        )
        print(result)
    """

    def __init__(self, use_fine_tuned: bool = True):
        self.model     = None
        self.tokenizer = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(use_fine_tuned)

    def _load_model(self, use_fine_tuned: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        fine_tuned_exists = FINE_TUNED_MODEL_DIR.exists() and any(FINE_TUNED_MODEL_DIR.iterdir())

        if use_fine_tuned and fine_tuned_exists:
            console.print(f"[green]Loading fine-tuned model from {FINE_TUNED_MODEL_DIR}[/green]")
            model_path = str(FINE_TUNED_MODEL_DIR)

            # Load LoRA fine-tuned model
            from peft import PeftModel
            base_model = self._load_base(BASE_MODEL_NAME)
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        else:
            if use_fine_tuned and not fine_tuned_exists:
                console.print(
                    "[yellow]⚠ Fine-tuned model not found. Using base model.[/yellow]\n"
                    "[dim]Run python src/train.py to fine-tune on your essays.[/dim]"
                )
            self.model     = self._load_base(BASE_MODEL_NAME)
            self.tokenizer = self._load_tokenizer(BASE_MODEL_NAME)

        self.model.eval()
        console.print("[green]✓ Model ready[/green]")

    def _load_base(self, model_name: str):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quant_config = None
        if self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

    def _load_tokenizer(self, model_name: str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        return tok

    def grade(
        self,
        question:  str,
        essay:     str,
        level:     str  = "AS",       # "AS" or "IGCSE"
        max_marks: int  = 12,
        verbose:   bool = True,
    ) -> str:
        """
        Grade an essay and return the examiner-style feedback string.
        """
        level      = level.upper()
        question_type = f"{'12-mark evaluate' if max_marks == 12 else '8-mark discuss'}"

        user_prompt = GRADING_PROMPT_TEMPLATE.format(
            level=level,
            question_type=question_type,
            max_marks=max_marks,
            question=question,
            essay=essay,
        )

        messages = [
            {"role": "system",    "content": GRADING_SYSTEM_PROMPT},
            {"role": "user",      "content": user_prompt},
        ]

        if verbose:
            console.print("[cyan]Grading essay...[/cyan]")

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=900,
                temperature=0.3,     # Low temperature = more consistent grading
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def grade_and_display(
        self,
        question:  str,
        essay:     str,
        level:     str = "AS",
        max_marks: int = 12,
    ) -> str:
        """Grades and prints the result in a nicely formatted panel."""
        result = self.grade(question, essay, level, max_marks)

        console.print("\n")
        console.print(Panel(
            Markdown(result),
            title=f"[bold cyan]Cambridge {level} Economics — Examiner Feedback[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))
        return result