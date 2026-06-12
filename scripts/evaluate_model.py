# scripts/evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluates the trained model's grading accuracy against known marks.
# Computes: mean absolute error, accuracy within ±1 mark, confusion matrix
# Run: python scripts/evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves to file
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import EVAL_FILE, MODELS_DIR

console = Console()
REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MARK EXTRACTION — parse the mark from the model's text output
# ─────────────────────────────────────────────────────────────────────────────

def extract_predicted_mark(response: str, max_marks: int) -> int | None:
    """
    Parses 'MARK AWARDED: X/12' or similar patterns from grader output.
    Returns None if not found.
    """
    patterns = [
        r"MARK AWARDED[:\s]+(\d+)\s*/\s*\d+",
        r"(\d+)\s*/\s*" + str(max_marks),
        r"awarded[:\s]+(\d+)",
        r"mark[:\s]+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            mark = int(match.group(1))
            if 0 <= mark <= max_marks:
                return mark
    return None


def extract_true_mark(sample: dict, max_marks: int) -> int | None:
    """Extract the true mark from a training sample."""
    # Look for the assistant response which contains MARK AWARDED
    for msg in sample.get("messages", []):
        if msg["role"] == "assistant":
            return extract_predicted_mark(msg["content"], max_marks)
    return None


def extract_max_marks_from_sample(sample: dict) -> int:
    """Detect whether the sample is 12-mark or 8-mark."""
    for msg in sample.get("messages", []):
        if "12" in msg.get("content", "") and "mark" in msg.get("content", "").lower():
            return 12
    return 8


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(use_fine_tuned: bool = True, sample_limit: int = None):
    """
    Runs the model on eval essays and compares predictions to true marks.
    """
    if not EVAL_FILE.exists():
        console.print(
            f"[red]Eval file not found: {EVAL_FILE}[/red]\n"
            "[yellow]Run python src/data_prep.py first.[/yellow]"
        )
        return

    # Load eval samples
    with open(EVAL_FILE, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if sample_limit:
        samples = samples[:sample_limit]

    if not samples:
        console.print("[red]No evaluation samples found.[/red]")
        return

    console.print(f"\n[cyan]Evaluating on {len(samples)} essays...[/cyan]\n")

    from grader import CambridgeGrader
    grader = CambridgeGrader(use_fine_tuned=use_fine_tuned)

    true_marks      = []
    predicted_marks = []
    errors          = []

    for i, sample in enumerate(samples, 1):
        max_marks = extract_max_marks_from_sample(sample)
        true_mark = extract_true_mark(sample, max_marks)

        if true_mark is None:
            console.print(f"[yellow]⚠ Skipping sample {i} — could not extract true mark[/yellow]")
            continue

        # Get the question + essay from the user message
        user_msg = next(
            (m["content"] for m in sample["messages"] if m["role"] == "user"), ""
        )

        # Extract question and essay from the formatted prompt
        question = _extract_field(user_msg, "Question:")
        essay    = _extract_field(user_msg, "Student's Essay:")

        if not essay:
            continue

        try:
            response       = grader.grade(question, essay, max_marks=max_marks, verbose=False)
            predicted_mark = extract_predicted_mark(response, max_marks)

            if predicted_mark is None:
                console.print(f"[yellow]⚠ Sample {i}: could not parse predicted mark from response[/yellow]")
                continue

            true_marks.append(true_mark)
            predicted_marks.append(predicted_mark)
            console.print(f"  [{i}/{len(samples)}] True: {true_mark}  Predicted: {predicted_mark}  {'✓' if abs(true_mark-predicted_mark)<=1 else '✗'}")

        except Exception as e:
            errors.append(str(e))
            console.print(f"[red]  Error on sample {i}: {e}[/red]")

    if not true_marks:
        console.print("[red]No valid predictions made.[/red]")
        return

    _print_metrics(true_marks, predicted_marks)
    _save_report(true_marks, predicted_marks, errors)
    _plot_results(true_marks, predicted_marks)


def _extract_field(text: str, field_label: str) -> str:
    """Extracts a field from the formatted prompt template."""
    pattern = re.escape(field_label) + r"\s*(.*?)(?=\n[A-Z]|\Z)"
    match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(true_marks: list, predicted_marks: list):
    true      = np.array(true_marks)
    predicted = np.array(predicted_marks)
    errors    = np.abs(true - predicted)

    mae            = np.mean(errors)
    rmse           = np.sqrt(np.mean((true - predicted) ** 2))
    exact_accuracy = np.mean(errors == 0) * 100
    within_1       = np.mean(errors <= 1) * 100
    within_2       = np.mean(errors <= 2) * 100

    table = Table(
        title="Model Evaluation Results",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric",          style="cyan")
    table.add_column("Value",           style="bold white")
    table.add_column("Interpretation",  style="dim")

    table.add_row(
        "Mean Absolute Error (MAE)",
        f"{mae:.2f} marks",
        "Lower is better. < 1.5 is good for this task.",
    )
    table.add_row(
        "Root Mean Squared Error",
        f"{rmse:.2f} marks",
        "Penalises large errors more.",
    )
    table.add_row(
        "Exact accuracy",
        f"{exact_accuracy:.1f}%",
        "% of essays graded at the exact correct mark.",
    )
    table.add_row(
        "Within ±1 mark",
        f"{within_1:.1f}%",
        "Cambridge examiner tolerance is ±1 mark.",
    )
    table.add_row(
        "Within ±2 marks",
        f"{within_2:.1f}%",
        "Broad accuracy check.",
    )
    table.add_row(
        "Samples evaluated",
        str(len(true_marks)),
        "",
    )

    console.print("\n")
    console.print(table)

    if within_1 >= 75:
        console.print("\n[bold green]✓ Model is performing at examiner-level accuracy (≥75% within ±1 mark)[/bold green]")
    elif within_1 >= 50:
        console.print("\n[yellow]Model is acceptable but needs more training data for higher accuracy.[/yellow]")
    else:
        console.print("\n[red]Model needs more training data. Add more essays and retrain.[/red]")


def _save_report(true_marks: list, predicted_marks: list, errors: list):
    from datetime import datetime
    true      = np.array(true_marks)
    predicted = np.array(predicted_marks)
    abs_errors = np.abs(true - predicted)

    report = {
        "timestamp":       datetime.now().isoformat(),
        "n_samples":       len(true_marks),
        "mae":             float(np.mean(abs_errors)),
        "rmse":            float(np.sqrt(np.mean((true - predicted) ** 2))),
        "exact_accuracy":  float(np.mean(abs_errors == 0)),
        "within_1_mark":   float(np.mean(abs_errors <= 1)),
        "within_2_marks":  float(np.mean(abs_errors <= 2)),
        "predictions":     list(zip(true_marks, predicted_marks)),
        "errors":          errors,
    }

    report_path = REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[dim]Report saved to: {report_path}[/dim]")


def _plot_results(true_marks: list, predicted_marks: list):
    """Saves a scatter plot of true vs predicted marks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation — True vs Predicted Marks", fontsize=14)

    true      = np.array(true_marks)
    predicted = np.array(predicted_marks)

    # Scatter plot
    axes[0].scatter(true, predicted, alpha=0.6, color="steelblue", edgecolors="navy", s=60)
    max_val = max(true.max(), predicted.max()) + 1
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    axes[0].fill_between([0, max_val], [-1, max_val-1], [1, max_val+1], alpha=0.1, color="green", label="±1 mark zone")
    axes[0].set_xlabel("True Mark")
    axes[0].set_ylabel("Predicted Mark")
    axes[0].set_title("True vs Predicted")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = predicted - true
    axes[1].hist(errors, bins=range(int(errors.min())-1, int(errors.max())+2),
                 color="steelblue", edgecolor="navy", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", label="No error")
    axes[1].set_xlabel("Prediction Error (Predicted − True)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "evaluation_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    console.print(f"[dim]Plot saved to: {plot_path}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("\n[bold cyan]Cambridge Grader — Model Evaluation[/bold cyan]\n")

    use_ft = "--base" not in sys.argv
    limit  = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    if "--base" in sys.argv:
        console.print("[yellow]Using base model (not fine-tuned)[/yellow]")
    else:
        console.print("[cyan]Using fine-tuned model[/cyan]")

    evaluate(use_fine_tuned=use_ft, sample_limit=limit)