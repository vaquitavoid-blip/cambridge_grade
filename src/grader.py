# src/grader.py
<<<<<<< HEAD
# ─────────────────────────────────────────────────────────────────────────────
# Core grading engine — three modes:
#   1. grade()      — full examiner grading with model evaluation point
#   2. edit_essay() — rewrites essay to perfection, shows changes
#   3. kae_analysis() — point-by-point Knowledge/Analysis/Evaluation gap analysis
# ─────────────────────────────────────────────────────────────────────────────
=======
# Cambridge Economics AI Examiner — Core Grading Engine
# Phases 1, 2, 3, 4: Grade / Edit / KAE Analysis / Essay Generation
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)

import re
import sys
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

sys.path.append(str(Path(__file__).parent))
from config import (
<<<<<<< HEAD
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    GRADING_SYSTEM_PROMPT, AS_GRADING_TEMPLATE, IGCSE_GRADING_TEMPLATE,
    EDIT_PROMPT_TEMPLATE, AS_EDIT_INSTRUCTIONS, IGCSE_EDIT_INSTRUCTIONS,
    KAE_ANALYSIS_PROMPT_TEMPLATE,
=======
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR, AO_MARKS,
    GRADING_SYSTEM_PROMPT,
    AS_GRADING_TEMPLATE, IGCSE_GRADING_TEMPLATE,
    EDIT_PROMPT_TEMPLATE, AS_EDIT_INSTRUCTIONS, IGCSE_EDIT_INSTRUCTIONS,
    KAE_ANALYSIS_PROMPT_TEMPLATE,
    ESSAY_GENERATOR_PROMPT, AS_GENERATOR_INSTRUCTIONS, IGCSE_GENERATOR_INSTRUCTIONS,
    EXAMINER_ASSISTANT_PROMPT,
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
)

console = Console()

# AO mark allocations by level
AO_MARKS = {
    "AS":    {"ao1": 2, "ao2": 6, "ao3": 4, "total": 12},
    "IGCSE": {"ao1": 2, "ao2": 4, "ao3": 2, "total": 8},
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PARSER — converts flat key: value format into structured dict
# ─────────────────────────────────────────────────────────────────────────────

def parse_grading_output(text: str, level: str, max_marks: int) -> dict:
    """
    Parses the flat KEY: value format output from the model into a structured dict.
    Much more robust than trying to parse markdown headings.
    """
    result = {
        "raw":        text,
        "level":      level,
        "max_marks":  max_marks,
        "mark":       None,
        "band":       "",
        "ao1_mark":   None, "ao1_max": AO_MARKS[level]["AO1"]["max"],
        "ao1_reason": "",
        "ao2_mark":   None, "ao2_max": AO_MARKS[level]["AO2"]["max"],
        "ao2_reason": "",
        "ao3_mark":   None, "ao3_max": AO_MARKS[level]["AO3"]["max"],
        "ao3_reason": "",
        "confidence":    "",
        "impression":    "",
        "strength_1":    "",
        "strength_2":    "",
        "gap_1":         "",
        "gap_2":         "",
        "eval_quality":  "",
        "model_eval":    "",
        "model_answer":  "",
        "next_band":     "",
        "point_1_status": "",
        "point_2_status": "",
    }

    def extract(key: str) -> str:
        """Extract value after KEY: up to the next KEY: or end of string."""
        pattern = rf"^{re.escape(key)}:\s*(.+?)(?=\n[A-Z][A-Z0-9_]+:|$)"
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def extract_number(key: str, max_val: int) -> int | None:
        val = extract(key)
        m = re.search(r"(\d+)", val)
        if m:
            n = int(m.group(1))
            return min(n, max_val)
        return None

    result["mark"]       = extract_number("MARK", max_marks)
    result["band"]       = extract("BAND")
    result["ao1_mark"]   = extract_number("AO1_MARK", result["ao1_max"])
    result["ao1_reason"] = extract("AO1_REASON")
    result["ao2_mark"]   = extract_number("AO2_MARK", result["ao2_max"])
    result["ao2_reason"] = extract("AO2_REASON")
    result["ao3_mark"]   = extract_number("AO3_MARK", result["ao3_max"])
    result["ao3_reason"] = extract("AO3_REASON")
    result["confidence"]    = extract("CONFIDENCE")
    result["impression"]    = extract("IMPRESSION")
    result["strength_1"]    = extract("STRENGTH_1")
    result["strength_2"]    = extract("STRENGTH_2")
    result["gap_1"]         = extract("GAP_1")
    result["gap_2"]         = extract("GAP_2")
    result["eval_quality"]  = extract("EVALUATION_QUALITY")
    result["model_eval"]    = extract("MODEL_EVAL")
    result["model_answer"]  = extract("MODEL_ANSWER")
    result["next_band"]     = extract("NEXT_BAND")
    result["point_1_status"] = extract("POINT_1_STATUS")
    result["point_2_status"] = extract("POINT_2_STATUS")

    # If mark wasn't parsed, try to infer from AO marks
    if result["mark"] is None:
        ao_sum = sum(filter(None, [result["ao1_mark"], result["ao2_mark"],
                                    result["ao3_mark"] if level == "AS" else 0]))
        if ao_sum > 0:
            result["mark"] = min(ao_sum, max_marks)

    return result


def parse_edit_output(text: str) -> dict:
    """Parse the edit essay output."""
    result = {"raw": text, "edited_essay": "", "changes": [], "predicted_mark": ""}

    essay_m = re.search(r"EDITED_ESSAY_START\s*(.*?)\s*EDITED_ESSAY_END", text, re.DOTALL)
    if essay_m:
        result["edited_essay"] = essay_m.group(1).strip()

    changes_m = re.search(r"CHANGES:\s*(.*?)(?=PREDICTED_MARK:|$)", text, re.DOTALL)
    if changes_m:
        lines = changes_m.group(1).strip().splitlines()
        result["changes"] = [l.strip("- ").strip() for l in lines if l.strip()]

    mark_m = re.search(r"PREDICTED_MARK:\s*(\d+/\d+)", text)
    if mark_m:
        result["predicted_mark"] = mark_m.group(1)

    return result


def parse_kae_output(text: str, level: str, max_marks: int) -> dict:
    """Parse KAE analysis output."""
    ao = AO_MARKS[level]
    result = {
        "raw": text,
        "ao1_score": None, "ao1_max": ao["AO1"]["max"], "ao1_feedback": "",
        "ao2_score": None, "ao2_max": ao["AO2"]["max"], "ao2_feedback": "",
        "ao3_score": None, "ao3_max": ao["AO3"]["max"], "ao3_feedback": "",
        "total_predicted": "", "fix_1": "", "fix_2": "", "fix_3": "",
        "essay_plan": "",
    }

    def exn(key, max_val):
        m = re.search(rf"^{key}:\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        if m:
            return min(int(m.group(1)), max_val)
        return None

    def ext(key):
        m = re.search(rf"^{key}:\s*(.+?)(?=\n[A-Z][A-Z0-9_]+:|$)", text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    result["ao1_score"]        = exn("AO1_SCORE", ao["AO1"]["max"])
    result["ao1_feedback"]     = ext("AO1_FEEDBACK")
    result["ao2_score"]        = exn("AO2_SCORE", ao["AO2"]["max"])
    result["ao2_feedback"]     = ext("AO2_FEEDBACK")
    result["ao3_score"]        = exn("AO3_SCORE", ao["AO3"]["max"])
    result["ao3_feedback"]     = ext("AO3_FEEDBACK")
    result["total_predicted"]  = ext("TOTAL_PREDICTED")
    result["fix_1"]            = ext("FIX_1")
    result["fix_2"]            = ext("FIX_2")
    result["fix_3"]            = ext("FIX_3")
    result["essay_plan"]       = ext("ESSAY_PLAN")
    return result


def parse_essay_output(text: str) -> dict:
    result = {"raw": text, "essay": "", "examiner_notes": ""}
    essay_m = re.search(r"ESSAY_START\s*(.*?)\s*ESSAY_END", text, re.DOTALL)
    if essay_m:
        result["essay"] = essay_m.group(1).strip()
    notes_m = re.search(r"EXAMINER_NOTES:\s*(.+?)$", text, re.DOTALL)
    if notes_m:
        result["examiner_notes"] = notes_m.group(1).strip()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRADER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CambridgeGrader:
    """
<<<<<<< HEAD
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
=======
    Cambridge Economics AI Examiner — all grading modes.

    Modes:
        grade()          → Phase 1: full AO grading with confidence scores
        edit_essay()     → Phase 2: essay improvement engine
        kae_analysis()   → Phase 3: KAE planner
        generate_essay() → Phase 4: full-mark essay generation
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
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
<<<<<<< HEAD
            base = self._load_base(BASE_MODEL_NAME)
            self.model     = PeftModel.from_pretrained(base, str(FINE_TUNED_MODEL_DIR))
            self.tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_MODEL_DIR), trust_remote_code=True)
        else:
            if use_fine_tuned and not fine_tuned_exists:
                console.print(
                    "[yellow]⚠ Fine-tuned model not found — using base model.[/yellow]\n"
                    "[dim]Run python src/train.py to fine-tune.[/dim]"
                )
=======
            base           = self._load_base(BASE_MODEL_NAME)
            self.model     = PeftModel.from_pretrained(base, str(FINE_TUNED_MODEL_DIR))
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(FINE_TUNED_MODEL_DIR), trust_remote_code=True)
        else:
            if use_fine_tuned and not fine_tuned_exists:
                console.print("[yellow]⚠ Fine-tuned model not found — using base model.[/yellow]")
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
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
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    def _load_tokenizer(self, model_name: str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────────────────────
    # SHARED INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 600) -> str:
=======
    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 700) -> str:
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
<<<<<<< HEAD

=======
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
<<<<<<< HEAD
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,   # reduced from 2048
        )

        # Move inputs to the same device as the first model parameter
=======
            input_text, return_tensors="pt", truncation=True, max_length=1024,
        )
        # Move inputs to same device as model's first parameter
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
<<<<<<< HEAD
                temperature=0.3,
=======
                temperature=0.2,          # lower = more deterministic formatting
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

<<<<<<< HEAD
        # Move outputs to CPU for decoding
=======
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
        output_ids   = outputs[0].cpu()
        input_length = inputs["input_ids"].shape[1]
        new_tokens   = output_ids[input_length:]
        result       = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

<<<<<<< HEAD
        # Fallback: if new_tokens is empty, decode the full output
        if not result:
=======
        # Fallback: if slicing failed, decode full output
        if len(result) < 10:
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
            result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return result

    # ─────────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
    # MODE 1 — GRADE
=======
    # PHASE 1 — GRADE
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
    # ─────────────────────────────────────────────────────────────────────────

    def grade(
        self,
        question:  str,
        essay:     str,
        level:     str = "AS",
        max_marks: int = 12,
<<<<<<< HEAD
        verbose:   bool = True,
    ) -> str:
        level         = level.upper()
        template      = AS_GRADING_TEMPLATE if level == "AS" else IGCSE_GRADING_TEMPLATE
        question_type = "12-mark evaluate" if max_marks == 12 else "8-mark discuss"

        user_prompt = template.format(
            question=question,
            essay=essay,
        )
=======
        rag_context: str = "",
        verbose:   bool = True,
    ) -> dict:
        """
        Returns a structured dict with mark, AO breakdown, confidence,
        strengths, gaps, model eval point etc.
        """
        level    = level.upper()
        template = AS_GRADING_TEMPLATE if level == "AS" else IGCSE_GRADING_TEMPLATE

        # Prepend RAG context if available (Phase 6)
        prefix = ""
        if rag_context:
            prefix = f"RELEVANT MARK SCHEME / EXAMINER REPORT CONTEXT:\n{rag_context}\n\n"

        user_prompt = prefix + template.format(question=question, essay=essay)
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)

        if verbose:
            console.print("[cyan]Grading essay...[/cyan]")

<<<<<<< HEAD
        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=600)

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
=======
        raw = self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=700)
        return parse_grading_output(raw, level, max_marks)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — EDIT ESSAY
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
    # ─────────────────────────────────────────────────────────────────────────

    def edit_essay(
        self,
        question:  str,
        essay:     str,
        level:     str = "AS",
        max_marks: int = 12,
        verbose:   bool = True,
<<<<<<< HEAD
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
            level_specific_instructions=AS_EDIT_INSTRUCTIONS if level.upper() == "AS" else IGCSE_EDIT_INSTRUCTIONS,
        )

        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=800)

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
=======
    ) -> dict:
        """Returns improved essay, list of changes, and predicted mark."""
        level = level.upper()
        instructions = AS_EDIT_INSTRUCTIONS if level == "AS" else IGCSE_EDIT_INSTRUCTIONS

        user_prompt = EDIT_PROMPT_TEMPLATE.format(
            question=question,
            level=level,
            max_marks=max_marks,
            essay=essay,
            level_specific_instructions=instructions,
        )

        if verbose:
            console.print("[cyan]Editing essay...[/cyan]")

        raw = self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=900)
        return parse_edit_output(raw)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — KAE ANALYSIS
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
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
<<<<<<< HEAD
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

        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=600)

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
=======
    ) -> dict:
        """Point-by-point KAE gap analysis returning structured dict."""
        level = level.upper()
        ao    = AO_MARKS[level]

        user_prompt = KAE_ANALYSIS_PROMPT_TEMPLATE.format(
            question=question,
            level=level,
            max_marks=max_marks,
            knowledge_points=knowledge_points   or "(none provided)",
            analysis_points=analysis_points     or "(none provided)",
            evaluation_points=evaluation_points or "(none provided)",
            ao1_max=ao["AO1"]["max"],
            ao2_max=ao["AO2"]["max"],
            ao3_max=ao["AO3"]["max"],
        )

        if verbose:
            console.print("[cyan]Analysing K/A/E points...[/cyan]")

        raw = self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=700)
        return parse_kae_output(raw, level, max_marks)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — ESSAY GENERATOR
    # ─────────────────────────────────────────────────────────────────────────

    def generate_essay(
        self,
        question:     str,
        level:        str = "AS",
        max_marks:    int = 12,
        from_notes:   str = "",
        from_kae:     str = "",
        verbose:      bool = True,
    ) -> dict:
        """
        Generates a full-mark Cambridge essay from:
          - question only
          - bullet-point notes (from_notes)
          - KAE plan (from_kae)
        """
        level        = level.upper()
        instructions = AS_GENERATOR_INSTRUCTIONS if level == "AS" else IGCSE_GENERATOR_INSTRUCTIONS
        mark_target  = "12/12" if level == "AS" else "8/8"

        extra_context = ""
        if from_notes:
            extra_context += f"\nSTUDENT'S NOTES/BULLET POINTS:\n{from_notes}\n"
        if from_kae:
            extra_context += f"\nSTUDENT'S KAE PLAN:\n{from_kae}\n"

        user_prompt = ESSAY_GENERATOR_PROMPT.format(
            question=question,
            level=level,
            max_marks=max_marks,
            extra_context=extra_context,
            mark_target=mark_target,
            level_instructions=instructions,
        )

        if verbose:
            console.print("[cyan]Generating full-mark essay...[/cyan]")

        raw = self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=900)
        return parse_essay_output(raw)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7 — EXAMINER ASSISTANT
    # ─────────────────────────────────────────────────────────────────────────

    def examiner_assistant(
        self,
        question:          str,
        retrieved_context: str = "",
        verbose:           bool = True,
    ) -> str:
        """Answers examiner-style questions using RAG context."""
        user_prompt = EXAMINER_ASSISTANT_PROMPT.format(
            retrieved_context=retrieved_context or "No specific resources retrieved.",
            question=question,
        )
        if verbose:
            console.print("[cyan]Consulting examiner knowledge base...[/cyan]")
        return self._generate(GRADING_SYSTEM_PROMPT, user_prompt, max_new_tokens=500)
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
