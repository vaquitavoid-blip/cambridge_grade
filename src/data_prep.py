# src/data_prep.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts raw essays + grades into the JSONL format needed for fine-tuning
# Handles CSV, Excel, or individual .txt files
#
# CRITICAL: the assistant "target" text built here MUST use the exact same
# section headings / format as get_grading_template(level) in config.py,
# which is what grader.py uses at inference time and what grading_app.py
# parses to render the UI. If these drift apart, the fine-tuned model
# learns a format the UI can't parse — which is what was causing marks
# not to display.
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
    GRADING_SYSTEM_PROMPT, get_grading_template, format_marks_breakdown,
    get_mark_band, AO_MARKS, EXAMINER_EXPECTATIONS,
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
        question_topic: str = "",
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
        return get_mark_band(self.level, self.mark)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FORMAT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_sample(entry: EssayEntry) -> dict:
    """
    Converts one essay entry into a chat-format training sample.
    Format: system + user (question+essay) → assistant (grading response)

    The user message uses get_grading_template(level), the SAME template
    grader.py uses at inference — so AS essays are trained with the AS
    template and IGCSE essays with the IGCSE template (previously both
    used the AS template, which meant the fine-tuned model never learned
    the IGCSE-specific section headings).
    """
    template = get_grading_template(entry.level)
    user_message = template.format(
        question=entry.question,
        essay=entry.essay,
    )

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

    The output sections and headings here MUST exactly match
    AS_GRADING_TEMPLATE / IGCSE_GRADING_TEMPLATE in config.py, since this
    is the text the model is trained to reproduce, and grading_app.py
    parses output using these exact heading strings.
    """
    if entry.level == "AS":
        return _build_as_target(entry)
    return _build_igcse_target(entry)


def _build_as_target(entry: EssayEntry) -> str:
    expectations = EXAMINER_EXPECTATIONS.get("AS_12_mark", {})
    ao1, ao2, ao3 = _ao_split(entry)
    ao1_note, ao2_note, ao3_note = _as_ao_notes(entry, ao1, ao2, ao3)

    strengths, weaknesses = _strengths_and_weaknesses(entry)

    return f"""### MARK AWARDED: {entry.mark}/{entry.max_marks}

### MARK BAND: {entry.mark_band}

### WHAT THE EXAMINER SEES
{entry.feedback if entry.feedback else _auto_impression(entry)}

### MARKS BREAKDOWN
{format_marks_breakdown(entry.level, ao1, ao2, ao3, ao1_note, ao2_note, ao3_note)}

### EVALUATION QUALITY
{_auto_evaluation_quality(entry, ao3)}

### MODEL EVALUATION POINT
{_auto_model_eval_point(entry)}

### STRENGTHS
{strengths}

### WHAT LOST MARKS
{weaknesses}

### HOW TO REACH THE NEXT BAND
{_auto_improvement(entry)}
""".strip()


def _build_igcse_target(entry: EssayEntry) -> str:
    ao1, ao2, ao3 = _ao_split(entry)
    ao1_note, ao2_note, ao3_note = _igcse_ao_notes(entry, ao1, ao2, ao3)

    strengths, weaknesses = _strengths_and_weaknesses(entry)

    return f"""### MARK AWARDED: {entry.mark}/{entry.max_marks}

### MARK BAND: {entry.mark_band}

### WHAT THE EXAMINER SEES
{entry.feedback if entry.feedback else _auto_impression(entry)}

### MARKS BREAKDOWN
{format_marks_breakdown(entry.level, ao1, ao2, ao3, ao1_note, ao2_note, ao3_note)}

### CONTENT ACCURACY CHECK
{_auto_content_accuracy(entry)}

### CLARITY ASSESSMENT
{_auto_clarity(entry)}

### MODEL EVALUATION POINT
{_auto_model_eval_point(entry)}

### STRENGTHS
{strengths}

### WHAT LOST MARKS
{weaknesses}

### HOW TO REACH THE NEXT BAND
{_auto_improvement(entry)}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-GENERATED CONTENT HELPERS
# Used when no real examiner feedback was provided for an essay.
# ─────────────────────────────────────────────────────────────────────────────

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


def _ao_split(entry: EssayEntry) -> tuple[int, int, int]:
    """Distributes the total mark across AO1/AO2/AO3 proportionally."""
    ao = AO_MARKS[entry.level]
    m = entry.mark
    if entry.level == "AS":
        ao1 = min(ao["ao1"], round(m * 0.17))
        ao3 = min(ao["ao3"], round(m * 0.33))
        ao2 = max(0, m - ao1 - ao3)
        # Clamp ao2 to its max in case rounding pushed it over
        if ao2 > ao["ao2"]:
            overflow = ao2 - ao["ao2"]
            ao2 = ao["ao2"]
            ao3 = min(ao["ao3"], ao3 + overflow)
    else:
        ao1 = min(ao["ao1"], round(m * 0.25))
        ao3 = min(ao["ao3"], round(m * 0.25))
        ao2 = max(0, m - ao1 - ao3)
        if ao2 > ao["ao2"]:
            overflow = ao2 - ao["ao2"]
            ao2 = ao["ao2"]
            ao1 = min(ao["ao1"], ao1 + overflow)

    return ao1, ao2, ao3


def _as_ao_notes(entry: EssayEntry, ao1: int, ao2: int, ao3: int) -> tuple[str, str, str]:
    ao = AO_MARKS["AS"]
    ao1_note = ("Appropriate use of economic terminology and concepts."
                 if ao1 >= ao["ao1"] else "Limited use of economic vocabulary.")
    ao2_note = ("Well-developed analytical chains with logical reasoning."
                 if ao2 >= 5 else "Some analytical development but chains of reasoning incomplete.")
    ao3_note = ("Confident evaluative judgment with supporting reasoning."
                 if ao3 >= 3 else "Evaluation is limited or unsupported by reasoning.")
    return ao1_note, ao2_note, ao3_note


def _igcse_ao_notes(entry: EssayEntry, ao1: int, ao2: int, ao3: int) -> tuple[str, str, str]:
    ao = AO_MARKS["IGCSE"]
    ao1_note = ("Economic concepts used correctly."
                 if ao1 >= ao["ao1"] else "Limited use of economic concepts.")
    ao2_note = ("Good analytical development with reasoning chains."
                 if ao2 >= 3 else "Analysis present but underdeveloped.")
    ao3_note = ("Evaluation with a supported judgment."
                 if ao3 >= ao["ao3"] else "Evaluation absent or unsupported.")
    return ao1_note, ao2_note, ao3_note


def _auto_evaluation_quality(entry: EssayEntry, ao3: int) -> str:
    if ao3 >= 3:
        return ("The essay engages in genuine evaluation — it weighs the question against "
                "conditions such as time horizon, magnitude, or context, and reaches a "
                "judgment that directly answers what was asked.")
    elif ao3 >= 1:
        return ("Some evaluative language is present (e.g. 'however', 'it depends'), but "
                "the judgment is not fully supported by reasoning — the essay states both "
                "sides without weighing them against each other or reaching a clear final "
                "position tied to the specific question.")
    else:
        return ("Evaluation is largely absent. The essay presents analysis without "
                "considering counter-arguments, conditions, or a final judgment that "
                "answers 'to what extent'.")


def _auto_content_accuracy(entry: EssayEntry) -> str:
    if entry.mark >= entry.max_marks - 1:
        return ("Both main points made align with standard Cambridge IGCSE mark scheme "
                "accepted answers (ACCEPTED). No off-syllabus or original points were "
                "identified that would not appear in a Cambridge mark scheme.")
    elif entry.mark >= entry.max_marks * 0.5:
        return ("At least one point made aligns with standard Cambridge IGCSE mark scheme "
                "accepted answers (ACCEPTED). One or more points may be too vague, "
                "incomplete, or borderline to be confidently marked as an accepted point — "
                "these should be made more specific and tied to standard syllabus "
                "terminology.")
    else:
        return ("The points made are too general, vague, or off-syllabus to be confidently "
                "marked as ACCEPTED against a standard Cambridge IGCSE mark scheme. Focus "
                "on stating clear, syllabus-level accepted points rather than broad or "
                "original ideas.")


def _auto_clarity(entry: EssayEntry) -> str:
    if entry.mark >= entry.max_marks * 0.75:
        return ("The explanation is clear, direct, and easy to follow — exactly what IGCSE "
                "examiners reward. Each point is explained in simple terms without "
                "unnecessary complexity.")
    elif entry.mark >= entry.max_marks * 0.4:
        return ("The explanation is mostly clear but could be more direct in places. "
                "IGCSE rewards simple, plain explanations over sophisticated language — "
                "aim for short, clear sentences that directly explain the mechanism.")
    else:
        return ("The explanation is unclear, overly brief, or overcomplicated. At IGCSE, "
                "clarity is rewarded over sophistication — state the point plainly, "
                "explain it in one or two simple sentences, then give an example.")


def _auto_model_eval_point(entry: EssayEntry) -> str:
    if entry.level == "AS":
        return (f"For this question, a top-band evaluation would reach a clear judgment "
                f"on the question asked, then immediately qualify it: 'However, the "
                f"extent to which this holds depends on [a specific condition relevant "
                f"to {entry.question_topic or 'this topic'} — e.g. the size of the "
                f"output gap, time lags, or the elasticity of the relevant curves]. In "
                f"the short run [X is more likely], whereas in the long run [Y becomes "
                f"more significant]. On balance, [final supported judgment that directly "
                f"answers the question].'")
    else:
        return ("A full-mark IGCSE response states two accepted points clearly, explains "
                "each in one or two simple sentences using correct syllabus terminology, "
                "and gives a real-world or textbook example for each — for instance, "
                "'[Accepted point 1], because [simple explanation]. For example, "
                "[example]. [Accepted point 2], because [simple explanation]. For "
                "example, [example].' A single 1-2 sentence evaluative comment (e.g. "
                "'However, this depends on...') is enough for the evaluation marks — "
                "anything longer is not rewarded at IGCSE.")


def _strengths_and_weaknesses(entry: EssayEntry) -> tuple[str, str]:
    """Generates bullet-point STRENGTHS / WHAT LOST MARKS sections."""
    ratio = entry.mark / entry.max_marks

    if ratio >= 0.75:
        strengths = (
            "- Clear command of relevant economic concepts and terminology\n"
            "- Points are developed with a logical chain of reasoning"
        )
        weaknesses = (
            "- Minor refinements to the final judgment would secure full marks — "
            "tie the conclusion even more explicitly back to the exact wording of "
            "the question"
        )
    elif ratio >= 0.5:
        strengths = (
            "- At least one point is identified and explained correctly\n"
            "- Relevant economic terminology is used"
        )
        weaknesses = (
            "- Analytical chains are incomplete — points are stated but not fully "
            "developed through to a real-world consequence or example\n"
            "- Evaluation is underdeveloped or asserted without supporting reasoning"
        )
    else:
        strengths = (
            "- Some relevant economic content is present\n"
            "- The response attempts to address the question topic"
        )
        weaknesses = (
            "- Points are largely definitional and not developed into analysis\n"
            "- No supported evaluative judgment is reached\n"
            "- Missing real-world examples or data to support the argument"
        )

    return strengths, weaknesses


def _auto_improvement(entry: EssayEntry) -> str:
    """Generates improvement advice based on current mark."""
    if entry.mark >= entry.max_marks - 1:
        return "This is already near the top band. To consistently achieve full marks, ensure your concluding judgment is explicitly tied to the context of the question, not a generic statement."
    elif entry.mark >= entry.max_marks * 0.6:
        if entry.level == "AS":
            return ("To reach the next band: (1) Develop your evaluation beyond 'it depends' — state specifically WHAT it depends on and WHY. "
                    "(2) Ensure each analytical point has a full chain: cause → mechanism → consequence → real-world example. "
                    "(3) Use a diagram and explicitly explain what it shows rather than just labelling it.")
        return ("To reach the next band: (1) Make sure both points are clearly accepted Cambridge IGCSE mark scheme answers. "
                "(2) Add a real-world or textbook example for each point. "
                "(3) Keep your evaluative comment brief — 1-2 sentences is enough.")
    else:
        if entry.level == "AS":
            return ("To reach the next band: (1) Pick TWO points only and develop each one fully rather than listing many shallow points. "
                    "(2) For each point, ask yourself: 'Why does this happen? What is the economic mechanism? What would happen next?' "
                    "(3) Add at least one real-world example (a country, a policy, a statistic). "
                    "(4) Finish with a sentence that directly answers the question asked.")
        return ("To reach the next band: (1) Make sure you state TWO accepted points — check they would appear in a Cambridge mark scheme. "
                "(2) Explain each point simply in one or two sentences — avoid AS-level complexity. "
                "(3) Add a clear example for each point. "
                "(4) Add one short evaluative sentence at the end.")


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
                examiner_feedback=str(row.get("feedback", "")) if pd.notna(row.get("feedback", "")) else "",
                question_topic=str(row.get("topic", "")) if pd.notna(row.get("topic", "")) else "",
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

    With very small datasets (e.g. 20 essays -> 3 eval samples), the eval
    split can end up with zero essays of one level, which makes
    evaluate_model.py's "extract_max_marks_from_sample" guess wrong. We
    stratify by level so both AS and IGCSE are represented in eval whenever
    enough essays of that level exist.
    """
    if len(entries) < 5:
        console.print("[red]⚠ You need at least 5 essays to build a dataset.[/red]")
        return

    random.seed(42)

    as_entries    = [e for e in entries if e.level == "AS"]
    igcse_entries = [e for e in entries if e.level == "IGCSE"]
    random.shuffle(as_entries)
    random.shuffle(igcse_entries)

    train_set: list[EssayEntry] = []
    eval_set:  list[EssayEntry] = []

    for subset in (as_entries, igcse_entries):
        if not subset:
            continue
        split_idx = max(1, int(len(subset) * (1 - eval_split))) if len(subset) > 1 else 1
        # Ensure at least 1 eval sample if we have 3+ of this level
        if len(subset) >= 3 and split_idx == len(subset):
            split_idx = len(subset) - 1
        train_set.extend(subset[:split_idx])
        eval_set.extend(subset[split_idx:])

    random.shuffle(train_set)
    random.shuffle(eval_set)

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

    if len(eval_set) == 0:
        console.print(
            "[yellow]⚠ No essays were placed in the evaluation set — "
            "evaluate_model.py will have nothing to evaluate. Add a few "
            "more essays so each level has at least 3.[/yellow]"
        )


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
