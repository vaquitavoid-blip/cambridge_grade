# scripts/run_grader.py
# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI — three grading modes:
#   1. Grade essay      — full examiner feedback + model evaluation point
#   2. Edit to perfect  — rewrites essay with changes explained
#   3. KAE analysis     — enter K/A/E points separately, get gap analysis
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
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
# INPUT HELPERS
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


def get_level_and_marks() -> tuple[str, int]:
    choice = select_option("Select exam level:", ["AS Level (12-mark)", "IGCSE (8-mark)"])
    level     = "AS"   if "AS" in choice else "IGCSE"
    max_marks = 12     if level == "AS"  else 8
    return level, max_marks


def save_result(mode: str, data: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = RESULTS_DIR / f"{mode}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Result saved to: {filename}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — GRADE ESSAY
# ─────────────────────────────────────────────────────────────────────────────

def run_grading():
    console.print(Panel(
        "[bold]Mode 1 — Grade Essay[/bold]\n"
        "[dim]Full examiner feedback, mark breakdown, and model evaluation point[/dim]",
        border_style="cyan",
    ))

    level, max_marks = get_level_and_marks()
    question = get_multiline_input("Paste the EXAM QUESTION:")
    if not question:
        console.print("[red]No question entered.[/red]")
        return

    show_preview = input("\nShow quick structural analysis before AI grading? (y/n): ").strip().lower()
    if show_preview == "y":
        essay_preview = get_multiline_input("Paste the essay (for structural check):")
        show_essay_analysis(essay_preview, level, max_marks)
        use_same = input("\nUse this same essay for AI grading? (y/n): ").strip().lower()
        essay = essay_preview if use_same == "y" else get_multiline_input("Paste the essay for AI grading:")
    else:
        essay = get_multiline_input("Paste the STUDENT ESSAY:")

    if not essay:
        console.print("[red]No essay entered.[/red]")
        return

    try:
        from grader import CambridgeGrader
        grader = CambridgeGrader(use_fine_tuned=True)
        result = grader.grade_and_display(question, essay, level, max_marks)
        save_result("grade", {"question": question, "essay": essay, "level": level, "result": result})
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — EDIT ESSAY TO PERFECTION
# ─────────────────────────────────────────────────────────────────────────────

def run_edit():
    console.print(Panel(
        "[bold]Mode 2 — Edit Essay to Perfection[/bold]\n"
        "[dim]Rewrites your essay for the highest possible mark, with changes explained[/dim]",
        border_style="green",
    ))

    level, max_marks = get_level_and_marks()
    question = get_multiline_input("Paste the EXAM QUESTION:")
    essay    = get_multiline_input("Paste the STUDENT ESSAY to improve:")

    if not question or not essay:
        console.print("[red]Question and essay are both required.[/red]")
        return

    try:
        from grader import CambridgeGrader
        grader = CambridgeGrader(use_fine_tuned=True)
        result = grader.edit_and_display(question, essay, level, max_marks)
        save_result("edit", {"question": question, "essay": essay, "level": level, "result": result})
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — KAE POINT-BY-POINT GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_kae():
    console.print(Panel(
        "[bold]Mode 3 — Knowledge / Analysis / Evaluation Gap Analysis[/bold]\n"
        "[dim]Enter your planned points separately — see exactly where you're losing marks[/dim]",
        border_style="yellow",
    ))

    level, max_marks = get_level_and_marks()
    ao = {"AS": {"ao1": 2, "ao2": 6, "ao3": 4}, "IGCSE": {"ao1": 2, "ao2": 4, "ao3": 2}}[level]

    question = get_multiline_input("Paste the EXAM QUESTION:")
    if not question:
        console.print("[red]No question entered.[/red]")
        return

    console.print(f"\n[dim]AO mark split for {level}: Knowledge={ao['ao1']}, Analysis={ao['ao2']}, Evaluation={ao['ao3']}[/dim]")

    console.print(Panel(
        "Enter your points for each section below.\n"
        "• [bold]Knowledge[/bold]: definitions, concepts, what you know\n"
        "• [bold]Analysis[/bold]: how/why — chains of reasoning, mechanisms\n"
        "• [bold]Evaluation[/bold]: judgments, conditions, 'it depends on...', final answer",
        title="Instructions",
        border_style="dim",
    ))

    knowledge_points  = get_multiline_input(f"KNOWLEDGE points (AO1 — {ao['ao1']} marks available):")
    analysis_points   = get_multiline_input(f"ANALYSIS points (AO2 — {ao['ao2']} marks available):")
    evaluation_points = get_multiline_input(f"EVALUATION points (AO3 — {ao['ao3']} marks available):")

    try:
        from grader import CambridgeGrader
        grader = CambridgeGrader(use_fine_tuned=True)
        result = grader.kae_and_display(
            question, level, max_marks,
            knowledge_points, analysis_points, evaluation_points,
        )
        save_result("kae", {
            "question": question, "level": level,
            "knowledge": knowledge_points, "analysis": analysis_points,
            "evaluation": evaluation_points, "result": result,
        })
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold white]Cambridge Economics AI Grader[/bold white]\n"
        "[dim]AS Level (12-mark) & IGCSE (8-mark)[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    while True:
        mode = select_option(
            "Select mode:",
            [
                "📝  Grade an essay  — full feedback + model evaluation point",
                "✏️   Edit to perfection — rewrite essay for highest mark",
                "🔬  KAE analysis — enter K/A/E points, see gap analysis",
                "📚  What examiners expect",
                "📈  View marking bands",
                "🚪  Exit",
            ],
        )

        if mode.startswith("📝"):
            run_grading()
        elif mode.startswith("✏️"):
            run_edit()
        elif mode.startswith("🔬"):
            run_kae()
        elif mode.startswith("📚"):
            level = select_option("Level?", ["AS Level", "IGCSE"])
            lc    = "AS" if "AS" in level else "IGCSE"
            show_examiner_expectations(lc, 12 if lc == "AS" else 8)
        elif mode.startswith("📈"):
            level = select_option("Level?", ["AS Level", "IGCSE"])
            show_marking_bands("AS" if "AS" in level else "IGCSE")
        else:
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()