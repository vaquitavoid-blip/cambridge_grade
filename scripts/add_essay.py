# scripts/add_essay.py
# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI tool to add a new essay + grade to your training dataset.
# Run: python scripts/add_essay.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import RAW_ESSAYS_DIR, PROCESSED_DIR, TRAINING_FILE

console = Console()

ESSAYS_CSV = RAW_ESSAYS_DIR / "essays.csv"
CSV_HEADERS = ["question", "essay", "mark", "max_marks", "level", "feedback", "topic", "date_added"]


def ensure_csv_exists():
    """Create the CSV with headers if it doesn't exist yet."""
    if not ESSAYS_CSV.exists():
        ESSAYS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(ESSAYS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        console.print(f"[green]✓ Created {ESSAYS_CSV}[/green]")


def count_existing():
    """Count how many essays are already in the CSV."""
    if not ESSAYS_CSV.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(ESSAYS_CSV)
        return len(df)
    except Exception:
        return 0


def get_multiline_input(prompt: str) -> str:
    """
    Prompts for multi-line input.
    User types their text, then types END on a new line to finish.
    """
    console.print(f"\n[cyan]{prompt}[/cyan]")
    console.print("[dim](Type or paste your text. When done, type END on a new line and press Enter)[/dim]")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


def add_essay_interactive():
    """Full interactive flow to add one essay."""
    console.print("\n[bold cyan]Add Essay to Training Data[/bold cyan]")
    console.print(f"[dim]Existing essays: {count_existing()}[/dim]\n")

    # Level
    level_input = ""
    while level_input not in ("1", "2"):
        level_input = input("Level — enter 1 for AS Level, 2 for IGCSE: ").strip()
    level     = "AS" if level_input == "1" else "IGCSE"
    max_marks = 12 if level == "AS" else 8

    # Topic tag
    console.print("\n[dim]Topic tag (e.g. 'fiscal policy', 'market failure') — helps with analysis later[/dim]")
    topic = input("Topic (press Enter to skip): ").strip()

    # Question
    question = get_multiline_input("Paste the exam QUESTION:")
    if not question:
        console.print("[red]Question cannot be empty. Aborting.[/red]")
        return

    # Essay
    essay = get_multiline_input("Paste the STUDENT ESSAY:")
    if not essay or len(essay) < 50:
        console.print("[red]Essay too short (< 50 chars). Aborting.[/red]")
        return

    # Mark
    mark = -1
    while not (0 <= mark <= max_marks):
        try:
            mark = int(input(f"\nMark awarded (0–{max_marks}): ").strip())
        except ValueError:
            console.print("[red]Please enter a whole number.[/red]")

    # Feedback
    console.print("\n[dim]Examiner feedback — paste real feedback if you have it, or press Enter to skip[/dim]")
    feedback = get_multiline_input("Examiner feedback (or just type END to skip):")

    # Confirm
    console.print("\n[bold]Summary of what you're adding:[/bold]")
    console.print(f"  Level:     {level} ({max_marks} marks)")
    console.print(f"  Topic:     {topic or '(none)'}")
    console.print(f"  Mark:      {mark}/{max_marks}")
    console.print(f"  Question:  {question[:80]}{'...' if len(question) > 80 else ''}")
    console.print(f"  Essay:     {len(essay.split())} words")
    console.print(f"  Feedback:  {'Yes' if feedback else 'None'}")

    confirm = input("\nSave this essay? (y/n): ").strip().lower()
    if confirm != "y":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Write to CSV
    ensure_csv_exists()
    with open(ESSAYS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow({
            "question":   question,
            "essay":      essay,
            "mark":       mark,
            "max_marks":  max_marks,
            "level":      level,
            "feedback":   feedback,
            "topic":      topic,
            "date_added": datetime.now().strftime("%Y-%m-%d"),
        })

    total = count_existing()
    console.print(f"\n[bold green]✓ Essay saved! Total in dataset: {total}[/bold green]")

    if total >= 20:
        console.print(
            f"\n[cyan]You now have {total} essays — enough to train the model![/cyan]\n"
            "[bold]Next:[/bold] python src/data_prep.py   (prepare training data)\n"
            "       python src/train.py          (fine-tune the model)"
        )
    else:
        needed = 20 - total
        console.print(f"[yellow]Add {needed} more essays before training (minimum 20 recommended).[/yellow]")


def add_from_file(filepath: str):
    """
    Bulk-add essays from a CSV/Excel file.
    Merges into the main essays.csv.
    Expects columns: question, essay, mark, max_marks, level, feedback (optional), topic (optional)
    """
    import pandas as pd

    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"question", "essay", "mark", "max_marks", "level"}
    missing  = required - set(df.columns)
    if missing:
        console.print(f"[red]File is missing columns: {missing}[/red]")
        return

    # Fill optional columns
    for col in ["feedback", "topic"]:
        if col not in df.columns:
            df[col] = ""
    df["date_added"] = datetime.now().strftime("%Y-%m-%d")

    ensure_csv_exists()
    df[CSV_HEADERS].to_csv(
        ESSAYS_CSV,
        mode="a",
        index=False,
        header=not ESSAYS_CSV.exists(),
        encoding="utf-8",
    )

    console.print(f"[green]✓ Added {len(df)} essays from {path.name}[/green]")
    console.print(f"[green]  Total in dataset: {count_existing()}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Bulk import: python scripts/add_essay.py path/to/file.csv
        add_from_file(sys.argv[1])
    else:
        # Interactive single-essay entry
        add_essay_interactive()

        # Ask if they want to add another
        while True:
            again = input("\nAdd another essay? (y/n): ").strip().lower()
            if again == "y":
                add_essay_interactive()
            else:
                break

        console.print("\n[dim]To prepare training data: python src/data_prep.py[/dim]")