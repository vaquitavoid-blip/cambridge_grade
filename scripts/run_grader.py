# scripts/run_grader.py
# ─────────────────────────────────────────────────────────────────────────────
# Main interactive CLI — grade an essay, get examiner feedback, see analysis
# Run: python scripts/run_grader.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

sys.path.append(str(Path(__file__).parent.parent / "src"))
from feedback_engine import (
    show_examiner_expectations,
    show_marking_bands,
    show_essay_analysis,
    show_model_answer_guide,
)

console = Console()
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_multiline_input(prompt: str) -> str:
    console.print(f"\n[cyan]{prompt}[/cyan]")
    console.print("[dim](Paste your text. Type END on a new line and press Enter when done)[/dim]")
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


def select_option(prompt: str, options: list[str]) -> str:
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}.[/cyan] {opt}")
    while True:
        choice = input("\nEnter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        console.print("[red]Invalid choice.[/red]")


def save_result(question: str, essay: str, level: str, max_marks: int, result: str):
    """Save the grading result to a JSON file for future reference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = RESULTS_DIR / f"result_{timestamp}.json"
    data = {
        "timestamp": timestamp,
        "level":     level,
        "max_marks": max_marks,
        "question":  question,
        "essay":     essay,
        "result":    result,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Result saved to: {filename}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# GRADING FLOW
# ─────────────────────────────────────────────────────────────────────────────

def run_grading_session():
    """Full grading flow: collect inputs → run analysis → run ML grader → display."""

    console.print(Panel(
        "[bold white]Cambridge Economics AI Grader[/bold white]\n"
        "[dim]AS Level (12-mark) & IGCSE (8-mark) • Examiner-style feedback[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Step 1 — Level
    level_choice = select_option("Select exam level:", ["AS Level (12-mark)", "IGCSE (8-mark)"])
    level     = "AS"    if "AS" in level_choice    else "IGCSE"
    max_marks = 12      if level == "AS"            else 8

    # Step 2 — Optional: show what examiner expects first
    if Confirm.ask("\nWould you like to see what the examiner expects before grading?", default=False):
        show_examiner_expectations(level, max_marks)
        input("\nPress Enter to continue to grading...")

    # Step 3 — Question
    question = get_multiline_input("Paste the EXAM QUESTION:")
    if not question:
        console.print("[red]No question entered. Exiting.[/red]")
        return

    # Step 4 — Essay
    essay = get_multiline_input("Paste the STUDENT ESSAY:")
    if not essay or len(essay) < 30:
        console.print("[red]Essay too short. Exiting.[/red]")
        return

    # Step 5 — Rule-based analysis (instant, no model needed)
    console.print("\n[bold cyan]Step 1 of 2: Quick Structural Analysis[/bold cyan]")
    show_essay_analysis(essay, level, max_marks)

    # Step 6 — Full ML grading
    console.print("\n[bold cyan]Step 2 of 2: AI Examiner Grading[/bold cyan]")
    use_model = Confirm.ask("Run the full AI grader for examiner-level feedback?", default=True)

    if use_model:
        try:
            from grader import CambridgeGrader
            grader = CambridgeGrader(use_fine_tuned=True)
            result = grader.grade_and_display(question, essay, level, max_marks)
            save_result(question, essay, level, max_marks, result)
        except ImportError as e:
            console.print(f"[red]Could not load grader: {e}[/red]")
            console.print("[yellow]Make sure you've installed requirements: pip install -r requirements.txt[/yellow]")
        except Exception as e:
            console.print(f"[red]Grading error: {e}[/red]")
            console.print("[yellow]If model isn't trained yet, run: python src/train.py[/yellow]")
    else:
        console.print("[dim]Skipped full AI grading.[/dim]")

    # Step 7 — Post-grading options
    while True:
        action = select_option(
            "What would you like to do next?",
            [
                "See model answer structure for this question",
                "See full marking band table",
                "Grade another essay",
                "Exit",
            ],
        )

        if action.startswith("See model answer"):
            show_model_answer_guide(question, level, max_marks)
        elif action.startswith("See full marking"):
            show_marking_bands(level)
        elif action.startswith("Grade another"):
            run_grading_session()
            return
        else:
            console.print("\n[dim]Goodbye! Keep practising those chains of analysis.[/dim]\n")
            return


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE — grade multiple essays from a CSV
# Usage: python scripts/run_grader.py --batch path/to/essays.csv
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(csv_path: str):
    """Grade all essays in a CSV file and save results."""
    import pandas as pd
    from grader import CambridgeGrader

    console.print(f"\n[cyan]Batch grading: {csv_path}[/cyan]")
    df = pd.read_csv(csv_path) if csv_path.endswith(".csv") else pd.read_excel(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    grader  = CambridgeGrader(use_fine_tuned=True)
    results = []

    for i, row in df.iterrows():
        console.print(f"\n[dim]Grading essay {i+1}/{len(df)}...[/dim]")
        try:
            result = grader.grade(
                question  = str(row.get("question", "")),
                essay     = str(row.get("essay", "")),
                level     = str(row.get("level", "AS")),
                max_marks = int(row.get("max_marks", 12)),
                verbose   = False,
            )
            results.append({**row.to_dict(), "ai_feedback": result})
        except Exception as e:
            console.print(f"[red]Error on row {i+1}: {e}[/red]")
            results.append({**row.to_dict(), "ai_feedback": f"ERROR: {e}"})

    # Save results
    out_path = RESULTS_DIR / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    console.print(f"\n[green]✓ Batch results saved to: {out_path}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            console.print("[red]Usage: python scripts/run_grader.py --batch path/to/essays.csv[/red]")
        else:
            run_batch(sys.argv[2])
    else:
        run_grading_session()