# src/data_prep.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts raw essays + grades into the JSONL format needed for fine-tuning
# Handles CSV, Excel, or individual .txt files
# ─────────────────────────────────────────────────────────────────────────────

import json
import random
import pandas as pd
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

# Import config using relative path trick so it works from any directory
import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    RAW_ESSAYS_DIR, PROCESSED_DIR, TRAINING_FILE, EVAL_FILE,
    GRADING_SYSTEM_PROMPT, GRADING_PROMPT_TEMPLATE,
    AS_MARKING_BANDS, IGCSE_MARKING_BANDS, EXAMINER_EXPECTATIONS
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# ESSAY DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EssayEntry:
    """Represents one essay with its question, grade, and feedback."""

    def __init__(
        self,
        question: str,
        essay: str,
        mark: int,
        max_marks: int,          # 12 for AS, 8 for IGCSE
        level: str,              # "AS" or "IGCSE"
        examiner_feedback: str,  # Real feedback if you have it
        question_topic: str = ""
    ):
        self.question       = question.strip()
        self.essay          = essay.strip()
        self.mark           = int(mark)
        self.max_marks      = int(max_marks)
        self.level          = level.upper()
        self.feedback       = examiner_feedback.strip()
        self.question_topic = question_topic
        self._validate()

    def _validate(self):
        if self.level not in ("AS", "IGCSE"):
            raise ValueError(f"Level must be 'AS' or 'IGCSE', got: {self.level}")
        if not (0 <= self.mark <= self.max_marks):
            raise ValueError(f"Mark {self.mark} out of range for max {self.max_marks}")
        if len(self.essay) < 50:
            raise ValueError("Essay seems too short (< 50 chars). Check your data.")

    @property
    def question_type(self):
        return f"{'12-mark evaluate' if self.max_marks == 12 else '8-mark discuss'}"

    @property
    def mark_band(self):
        bands = AS_MARKING_BANDS if self.level == "AS" else IGCSE_MARKING_BANDS
        for band_range, description in bands.items():
            if "-" in band_range:
                low, high = map(int, band_range.split("-"))
                if low <= self.mark <= high:
                    return description
            else:
                if self.mark == int(band_range):
                    return description
        return "Unknown band"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FORMAT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_sample(entry: EssayEntry) -> dict:
    """
    Converts one essay entry into a chat-format training sample.
    Format: system + user (question+essay) → assistant (grading response)
    """
    user_message = GRADING_PROMPT_TEMPLATE.format(
        level=entry.level,
        question_type=entry.question_type,
        max_marks=entry.max_marks,
        question=entry.question,
        essay=entry.essay,
    )

    # Build the target assistant response from the real grade + feedback
    # This is what the model learns to produce
    assistant_response = _build_target_response(entry)

    return {
        "messages": [
            {"role": "system",    "content": GRADING_SYSTEM_PROMPT},
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def _build_target_response(entry: EssayEntry) -> str:
    """
    Constructs the ideal grader response from known mark + feedback.
    If you have real examiner feedback, it's used directly.
    Otherwise, a structured template is generated from the mark band.
    """
    expectations = EXAMINER_EXPECTATIONS.get(
        f"{'AS_12_mark' if entry.level == 'AS' else 'IGCSE_8_mark'}", {}
    )

    response = f"""### MARK AWARDED: {entry.mark}/{entry.max_marks}

### MARK BAND: {entry.mark_band}

### WHAT THE EXAMINER SEES
{entry.feedback if entry.feedback else _auto_impression(entry)}

### MARKS BREAKDOWN
{_auto_breakdown(entry)}

### HOW TO REACH THE NEXT BAND
{_auto_improvement(entry)}

### MODEL ANSWER STRUCTURE
For this question, a top-band response would include:
{chr(10).join(f'- {s}' for s in expectations.get('must_include', []))}
"""
    return response.strip()


def _auto_impression(entry: EssayEntry) -> str:
    """Auto-generates an examiner impression based on mark."""
    ratio = entry.mark / entry.max_marks
    if ratio >= 0.85:
        return "A well-constructed response demonstrating strong analytical skills and clear evaluative judgment. The candidate shows command of economic concepts and applies them effectively to the question."
    elif ratio >= 0.65:
        return "A competent response with some good analytical development. The candidate demonstrates understanding but evaluation is underdeveloped or the argument lacks full consistency."
    elif ratio >= 0.4:
        return "A basic response showing some relevant knowledge. Analysis is limited and evaluation is mostly absent. The candidate describes rather than analyses."
    else:
        return "A weak response. Relevant points are present but undeveloped. The candidate relies on definitions and assertions without demonstrating analytical understanding."


def _auto_breakdown(entry: EssayEntry) -> str:
    """Distributes the total mark across AOs proportionally."""
    m = entry.mark
    if entry.max_marks == 12:
        # AS: AO1=2, AO2=6, AO3=4 (approx Cambridge weighting)
        ao1 = min(2, round(m * 0.17))
        ao3 = min(4, round(m * 0.33))
        ao2 = m - ao1 - ao3
        return (
            f"**Knowledge & Understanding (AO1):** {ao1}/2 — "
            f"{'Appropriate use of economic terminology and concepts.' if ao1 >= 2 else 'Limited use of economic vocabulary.'}\n"
            f"**Analysis (AO2):** {ao2}/6 — "
            f"{'Well-developed analytical chains with logical reasoning.' if ao2 >= 5 else 'Some analytical development but chains of reasoning incomplete.'}\n"
            f"**Evaluation (AO3):** {ao3}/4 — "
            f"{'Confident evaluative judgment with supporting reasoning.' if ao3 >= 3 else 'Evaluation is limited or unsupported by reasoning.'}"
        )
    else:
        # IGCSE: simpler 2+4+2 split
        ao1 = min(2, round(m * 0.25))
        ao3 = min(2, round(m * 0.25))
        ao2 = m - ao1 - ao3
        return (
            f"**Knowledge (AO1):** {ao1}/2 — "
            f"{'Economic concepts used correctly.' if ao1 >= 2 else 'Limited use of economic concepts.'}\n"
            f"**Analysis (AO2):** {ao2}/4 — "
            f"{'Good analytical development with reasoning chains.' if ao2 >= 3 else 'Analysis present but underdeveloped.'}\n"
            f"**Evaluation (AO3):** {ao3}/2 — "
            f"{'Evaluation with a supported judgment.' if ao3 >= 2 else 'Evaluation absent or unsupported.'}"
        )


def _auto_improvement(entry: EssayEntry) -> str:
    """Generates improvement advice based on current mark."""
    if entry.mark >= entry.max_marks - 1:
        return "This is already near the top band. To consistently achieve full marks, ensure your concluding judgment is explicitly tied to the context of the question, not a generic statement."
    elif entry.mark >= entry.max_marks * 0.6:
        return ("To reach the next band: (1) Develop your evaluation beyond 'it depends' — state specifically WHAT it depends on and WHY. "
                "(2) Ensure each analytical point has a full chain: cause → mechanism → consequence → real-world example. "
                "(3) Use a diagram and explicitly explain what it shows rather than just labelling it.")
    else:
        return ("To reach the next band: (1) Pick TWO points only and develop each one fully rather than listing many shallow points. "
                "(2) For each point, ask yourself: 'Why does this happen? What is the economic mechanism? What would happen next?' "
                "(3) Add at least one real-world example (a country, a policy, a statistic). "
                "(4) Finish with a sentence that directly answers the question asked.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS — Load from CSV, Excel, or folder of .txt files
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(filepath: str) -> list[EssayEntry]:
    """
    Load essays from a CSV/Excel file.

    Expected columns:
    - question       (str)
    - essay          (str)
    - mark           (int)
    - max_marks      (int)   — 12 or 8
    - level          (str)   — "AS" or "IGCSE"
    - feedback       (str)   — examiner feedback (can be empty)
    - topic          (str)   — optional topic tag
    """
    path = Path(filepath)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"question", "essay", "mark", "max_marks", "level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}\nFound: {list(df.columns)}")

    entries = []
    for i, row in df.iterrows():
        try:
            entry = EssayEntry(
                question=str(row["question"]),
                essay=str(row["essay"]),
                mark=int(row["mark"]),
                max_marks=int(row["max_marks"]),
                level=str(row["level"]),
                examiner_feedback=str(row.get("feedback", "")),
                question_topic=str(row.get("topic", "")),
            )
            entries.append(entry)
        except Exception as e:
            console.print(f"[yellow]⚠ Skipping row {i+2}: {e}[/yellow]")

    console.print(f"[green]✓ Loaded {len(entries)} essays from {path.name}[/green]")
    return entries


def load_from_txt_folder(folder: str) -> list[EssayEntry]:
    """
    Load essays from a folder of .txt files.
    Each file should follow this format:

        LEVEL: AS
        MAX_MARKS: 12
        MARK: 9
        QUESTION: Evaluate the impact of a minimum wage on employment.
        FEEDBACK: Good analysis but evaluation is weak.
        ---
        [essay text below the --- line]
    """
    folder_path = Path(folder)
    entries = []

    for txt_file in folder_path.glob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")
            meta_part, essay_text = content.split("---", 1)

            meta = {}
            for line in meta_part.strip().splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip().upper()] = val.strip()

            entry = EssayEntry(
                question=meta.get("QUESTION", ""),
                essay=essay_text.strip(),
                mark=int(meta.get("MARK", 0)),
                max_marks=int(meta.get("MAX_MARKS", 12)),
                level=meta.get("LEVEL", "AS"),
                examiner_feedback=meta.get("FEEDBACK", ""),
                question_topic=meta.get("TOPIC", ""),
            )
            entries.append(entry)
        except Exception as e:
            console.print(f"[yellow]⚠ Skipping {txt_file.name}: {e}[/yellow]")

    console.print(f"[green]✓ Loaded {len(entries)} essays from folder[/green]")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER — Converts entries to JSONL training files
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(entries: list[EssayEntry], eval_split: float = 0.15):
    """
    Converts essay entries to training + eval JSONL files.
    - 85% goes to training, 15% to evaluation by default
    """
    if len(entries) < 5:
        console.print("[red]⚠ You need at least 5 essays to build a dataset.[/red]")
        return

    random.seed(42)
    random.shuffle(entries)

    split_idx  = max(1, int(len(entries) * (1 - eval_split)))
    train_set  = entries[:split_idx]
    eval_set   = entries[split_idx:]

    _write_jsonl(train_set, TRAINING_FILE)
    _write_jsonl(eval_set,  EVAL_FILE)

    # Print summary table
    table = Table(title="Dataset Summary")
    table.add_column("Split",    style="cyan")
    table.add_column("Count",    style="magenta")
    table.add_column("AS essays",  style="green")
    table.add_column("IGCSE essays", style="green")

    for name, split in [("Training", train_set), ("Evaluation", eval_set)]:
        as_count    = sum(1 for e in split if e.level == "AS")
        igcse_count = sum(1 for e in split if e.level == "IGCSE")
        table.add_row(name, str(len(split)), str(as_count), str(igcse_count))

    console.print(table)
    console.print(f"\n[green]✓ Training data saved to:   {TRAINING_FILE}[/green]")
    console.print(f"[green]✓ Evaluation data saved to: {EVAL_FILE}[/green]")


def _write_jsonl(entries: list[EssayEntry], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            sample = build_training_sample(entry)
            f.write(json.dumps(sample) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run this file directly to process your data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("\n[bold cyan]Cambridge Economics Grader — Data Preparation[/bold cyan]\n")

    # If Google Sheets is configured, pull the latest submissions first
    try:
        import sheets_backend as backend
        status = backend.get_backend_status()
        if status["sheets_connected"]:
            n = backend.sync_sheets_to_csv()
            console.print(f"[green]✓ Synced {n} essays from Google Sheets → essays.csv[/green]")
        else:
            console.print("[dim]Google Sheets not connected — using local essays.csv[/dim]")
    except Exception as e:
        console.print(f"[dim]Sheets sync skipped: {e}[/dim]")

    all_entries = []

    # Load from CSV if it exists
    csv_path = RAW_ESSAYS_DIR / "essays.csv"
    if csv_path.exists():
        console.print(f"Found essays.csv — loading...")
        all_entries.extend(load_from_csv(str(csv_path)))

    # Load from txt folder
    txt_entries = load_from_txt_folder(str(RAW_ESSAYS_DIR))
    all_entries.extend(txt_entries)

    if not all_entries:
        console.print(
            "[yellow]No essays found. Add essays to data/raw_essays/\n"
            "Either as essays.csv (see README) or as .txt files.\n"
            "Or use: python scripts/add_essay.py to add essays one by one.[/yellow]"
        )
    else:
        console.print(f"\n[bold]Total essays loaded: {len(all_entries)}[/bold]")
        build_dataset(all_entries)