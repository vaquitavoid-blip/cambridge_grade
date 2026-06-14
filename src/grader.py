# src/grader.py
# ─────────────────────────────────────────────────────────────────────────────
# Core grading engine — three modes:
#   1. grade()      — full examiner grading with model evaluation point
#   2. edit_essay() — rewrites essay to perfection, shows changes
#   3. kae_analysis() — point-by-point Knowledge/Analysis/Evaluation gap analysis
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
    EDIT_PROMPT_TEMPLATE, KAE_ANALYSIS_PROMPT_TEMPLATE,
)

console = Console()

# AO mark allocations by level
AO_MARKS = {
    "AS":    {"ao1": 2, "ao2": 6, "ao3": 4, "total": 12},
    "IGCSE": {"ao1": 2, "ao2": 4, "ao3": 2, "total": 8},
}


class CambridgeGrader:
    """
    Loads the grading model and exposes three grading modes.

    Usage:
        grader = CambridgeGrader()

        # Mode 1 — Grade
        result = grader.grade(question, essay, level="AS", max_marks=12)

        # Mode 2 — Edit to perfection
        edited = grader.edit_essay(question, essay, level="AS", max_marks=12)

        # Mode 3 — KAE gap analysis
        analysis = grader.kae_analysis(
            question, level="AS", max_marks=12,
            knowledge_points="...", analysis_points="...", evaluation_points="..."
        )
    """

    def __init__(self, use_fine_tuned: bool = True):
        self.model     = None
        self.tokenizer = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(use_fine_tuned)

    def _load_model(self, use_fine_tuned: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        fine_tuned_exists = (
            FINE_TUNED_MODEL_DIR.exists() and
            any(FINE_TUNED_MODEL_DIR.iterdir())
        )

        if use_fine_tuned and fine_tuned_exists:
            console.print(f"[green]Loading fine-tuned model from {FINE_TUNED_MODEL_DIR}[/green]")
            from peft import PeftModel
            base = self._load_base(BASE_MODEL_NAME)
            self.model     = PeftModel.from_pretrained(base, str(FINE_TUNED_MODEL_DIR))
            self.tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_MODEL_DIR), trust_remote_code=True)
        else:
            if use_fine_tuned and not fine_tuned_exists:
                console.print(
                    "[yellow]⚠ Fine-tuned model not found — using base model.[/yellow]\n"
                    "[dim]Run python src/train.py to fine-tune.[/dim]"
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
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1200) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

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
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1 — GRADE
    # ─────────────────────────────────────────────────────────────────────────

    def grade(
        self,
        question:  str,
        essay:     str,
        level:     str = "AS",
        max_marks: int = 12,
        verbose:   bool = True,
    ) -> str:
        level         = level.upper()
        question_type = "12-mark evaluate" if max_marks == 12 else "8-mark discuss"

        user_prompt = GRADING_PROMPT_TEMPLATE.format(
            level=level,
            question_type=question_type,
            max_marks=max_marks,
            question=question,
            essay=essay,
        )

        if verbose:
            console.print("[cyan]Grading essay...[/cyan]")

        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=1400)

    def grade_and_display(self, question: str, essay: str, level: str = "AS", max_marks: int = 12) -> str:
        result = self.grade(question, essay, level, max_marks)
        console.print("\n")
        console.print(Panel(
            Markdown(result),
            title=f"[bold cyan]Cambridge {level} Economics — Examiner Feedback[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2 — EDIT ESSAY TO PERFECTION
    # ─────────────────────────────────────────────────────────────────────────

    def edit_essay(
        self,
        question:  str,
        essay:     str,
        level:     str = "AS",
        max_marks: int = 12,
        verbose:   bool = True,
    ) -> str:
        """
        Rewrites the student's essay to achieve the highest possible mark,
        keeping their ideas but strengthening analysis, evaluation, and terminology.
        Shows exactly what was changed and why.
        """
        if verbose:
            console.print("[cyan]Editing essay to perfection...[/cyan]")

        user_prompt = EDIT_PROMPT_TEMPLATE.format(
            question=question,
            level=level.upper(),
            max_marks=max_marks,
            essay=essay,
        )

        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=1800)

    def edit_and_display(self, question: str, essay: str, level: str = "AS", max_marks: int = 12) -> str:
        result = self.edit_essay(question, essay, level, max_marks)
        console.print("\n")
        console.print(Panel(
            Markdown(result),
            title="[bold green]Essay Edited to Perfection — Cambridge Examiner[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3 — KAE POINT-BY-POINT GAP ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def kae_analysis(
        self,
        question:          str,
        level:             str = "AS",
        max_marks:         int = 12,
        knowledge_points:  str = "",
        analysis_points:   str = "",
        evaluation_points: str = "",
        verbose:           bool = True,
    ) -> str:
        """
        Takes the student's planned K/A/E points separately and gives
        a detailed gap analysis showing exactly where marks are being lost.
        """
        if verbose:
            console.print("[cyan]Analysing Knowledge / Analysis / Evaluation points...[/cyan]")

        ao = AO_MARKS.get(level.upper(), AO_MARKS["AS"])

        user_prompt = KAE_ANALYSIS_PROMPT_TEMPLATE.format(
            question=question,
            level=level.upper(),
            max_marks=max_marks,
            knowledge_points=knowledge_points or "(none provided)",
            analysis_points=analysis_points   or "(none provided)",
            evaluation_points=evaluation_points or "(none provided)",
            ao1_max=ao["ao1"],
            ao2_max=ao["ao2"],
            ao3_max=ao["ao3"],
        )

        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=1200)

    def kae_and_display(
        self,
        question: str,
        level: str = "AS",
        max_marks: int = 12,
        knowledge_points: str = "",
        analysis_points: str = "",
        evaluation_points: str = "",
    ) -> str:
        result = self.kae_analysis(
            question, level, max_marks,
            knowledge_points, analysis_points, evaluation_points,
        )
        console.print("\n")
        console.print(Panel(
            Markdown(result),
            title="[bold yellow]Knowledge / Analysis / Evaluation — Gap Analysis[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        return result