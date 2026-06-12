# src/feedback_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Feedback Engine — teaches students WHAT Cambridge examiners expect
# Works independently of the grader (no model needed)
# Can be used as a teaching tool or combined with grader output
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich import box

sys.path.append(str(Path(__file__).parent))
from config import EXAMINER_EXPECTATIONS, AS_MARKING_BANDS, IGCSE_MARKING_BANDS

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD ANALYSIS
# These are the words/phrases Cambridge examiners reward
# ─────────────────────────────────────────────────────────────────────────────

EVALUATION_KEYWORDS = [
    "however", "in contrast", "on the other hand", "it depends",
    "in the long run", "in the short run", "it is important to note",
    "nevertheless", "despite this", "although", "conversely",
    "the extent to which", "this is more likely when", "assuming",
    "the significance of", "to conclude", "overall", "therefore",
    "in conclusion", "the most significant", "more/less effective",
    "compared to", "in reality", "in practice",
]

ANALYSIS_PHRASES = [
    "this leads to", "as a result", "therefore", "consequently",
    "this means that", "which causes", "this results in",
    "due to", "because", "since", "thus", "hence",
    "this will cause", "leading to", "this increases", "this decreases",
    "the effect is", "this shifts", "rising", "falling",
]

ECONOMIC_CONCEPTS_AS = [
    "aggregate demand", "aggregate supply", "price level", "real gdp",
    "multiplier", "accelerator", "fiscal policy", "monetary policy",
    "supply-side policy", "trade-off", "opportunity cost",
    "price elasticity", "income elasticity", "externality",
    "market failure", "government failure", "asymmetric information",
    "monopoly", "oligopoly", "perfect competition", "monopolistic competition",
    "economies of scale", "diseconomies", "long run average cost",
    "marginal cost", "marginal revenue", "allocative efficiency",
    "productive efficiency", "dynamic efficiency", "x-inefficiency",
    "balance of payments", "current account", "exchange rate",
    "inflation", "deflation", "unemployment", "economic growth",
]

ECONOMIC_CONCEPTS_IGCSE = [
    "demand", "supply", "price", "equilibrium", "elasticity",
    "consumer", "producer", "market", "opportunity cost",
    "scarcity", "factors of production", "specialisation",
    "inflation", "unemployment", "economic growth",
    "government spending", "taxation", "imports", "exports",
    "subsidy", "minimum price", "maximum price",
]


# ─────────────────────────────────────────────────────────────────────────────
# ESSAY ANALYSER — rule-based analysis without needing a model
# ─────────────────────────────────────────────────────────────────────────────

class EssayAnalyser:
    """
    Analyses an essay for Cambridge examiner signal words and structures.
    No ML needed — pure rule-based analysis.
    """

    def __init__(self, essay: str, level: str = "AS", max_marks: int = 12):
        self.essay      = essay.lower()
        self.raw_essay  = essay
        self.level      = level.upper()
        self.max_marks  = max_marks
        self.word_count = len(essay.split())

    def count_evaluation_words(self) -> list[str]:
        return [kw for kw in EVALUATION_KEYWORDS if kw in self.essay]

    def count_analysis_phrases(self) -> list[str]:
        return [ph for ph in ANALYSIS_PHRASES if ph in self.essay]

    def count_economic_concepts(self) -> list[str]:
        concepts = ECONOMIC_CONCEPTS_AS if self.level == "AS" else ECONOMIC_CONCEPTS_IGCSE
        return [c for c in concepts if c in self.essay]

    def has_diagram_reference(self) -> bool:
        diagram_words = ["diagram", "figure", "graph", "curve", "shifts", "shown below",
                         "ad curve", "as curve", "supply curve", "demand curve"]
        return any(w in self.essay for w in diagram_words)

    def has_real_world_example(self) -> bool:
        example_signals = [
            "for example", "for instance", "such as", "in the uk", "in the us",
            "in china", "in india", "as seen in", "evidenced by", "data shows",
            "according to", "recently", "in 20",  # catches "in 2023" etc
        ]
        return any(s in self.essay for s in example_signals)

    def has_conclusion(self) -> bool:
        conclusion_signals = ["in conclusion", "to conclude", "overall", "therefore",
                               "in summary", "on balance", "ultimately"]
        return any(s in self.essay for s in conclusion_signals)

    def paragraph_count(self) -> int:
        return len([p for p in self.raw_essay.split("\n\n") if p.strip()])

    def get_checklist(self) -> dict:
        eval_words    = self.count_evaluation_words()
        analysis_words = self.count_analysis_phrases()
        concepts      = self.count_economic_concepts()

        return {
            "word_count":          self.word_count,
            "paragraph_count":     self.paragraph_count(),
            "evaluation_words":    eval_words,
            "analysis_phrases":    analysis_words,
            "economic_concepts":   concepts,
            "has_diagram":         self.has_diagram_reference(),
            "has_real_example":    self.has_real_world_example(),
            "has_conclusion":      self.has_conclusion(),
            "eval_word_count":     len(eval_words),
            "analysis_count":      len(analysis_words),
            "concept_count":       len(concepts),
        }

    def quick_score(self) -> tuple[int, str]:
        """
        Returns a predicted score range and confidence note.
        Rule-based heuristic — NOT a replacement for the full ML grader.
        """
        c = self.get_checklist()
        score = 0

        # Word count baseline
        if self.max_marks == 12:
            if c["word_count"] >= 400: score += 2
            elif c["word_count"] >= 250: score += 1
        else:
            if c["word_count"] >= 250: score += 2
            elif c["word_count"] >= 150: score += 1

        # Economic concepts
        if c["concept_count"] >= 5:   score += 2
        elif c["concept_count"] >= 3: score += 1

        # Analysis phrases
        if c["analysis_count"] >= 4:  score += 2
        elif c["analysis_count"] >= 2: score += 1

        # Evaluation words
        if c["eval_word_count"] >= 4:  score += 2
        elif c["eval_word_count"] >= 2: score += 1

        # Bonuses
        if c["has_diagram"]:       score += 1
        if c["has_real_example"]:  score += 1
        if c["has_conclusion"]:    score += 1

        # Scale to max_marks
        predicted = round(score / 11 * self.max_marks)
        predicted = max(1, min(predicted, self.max_marks))
        return predicted, "Heuristic estimate — use full grader for accuracy"


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def show_examiner_expectations(level: str = "AS", max_marks: int = 12):
    """Prints what Cambridge examiners expect for this question type."""
    key = "AS_12_mark" if level == "AS" else "IGCSE_8_mark"
    exp = EXAMINER_EXPECTATIONS.get(key, {})

    console.print(f"\n[bold cyan]What Cambridge Examiners Expect — {level} {max_marks}-Mark Questions[/bold cyan]\n")

    # Structure
    console.print(Panel(
        "\n".join(f"  [bold]{i+1}.[/bold] {s}" for i, s in enumerate(exp.get("structure", []))),
        title="[yellow]Ideal Structure[/yellow]",
        border_style="yellow",
    ))

    # Must include
    console.print(Panel(
        "\n".join(f"  [green]✓[/green] {m}" for m in exp.get("must_include", [])),
        title="[green]Must Include for Top Marks[/green]",
        border_style="green",
    ))

    # Common mistakes
    console.print(Panel(
        "\n".join(f"  [red]✗[/red] {m}" for m in exp.get("common_mistakes", [])),
        title="[red]Common Mistakes That Lose Marks[/red]",
        border_style="red",
    ))


def show_marking_bands(level: str = "AS"):
    """Displays the full marking band table."""
    bands = AS_MARKING_BANDS if level == "AS" else IGCSE_MARKING_BANDS
    max_m = 12 if level == "AS" else 8

    table = Table(
        title=f"Cambridge {level} Economics — Marking Bands ({max_m} marks)",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Marks", style="bold cyan", width=8)
    table.add_column("What the Examiner Needs to See", style="white")

    colors = ["green", "green", "yellow", "yellow", "orange3", "red", "red"]
    for i, (band, desc) in enumerate(bands.items()):
        color = colors[min(i, len(colors)-1)]
        table.add_row(f"[{color}]{band}[/{color}]", desc)

    console.print("\n")
    console.print(table)


def show_essay_analysis(essay: str, level: str = "AS", max_marks: int = 12):
    """
    Runs the rule-based analysis on an essay and displays a detailed
    checklist of what's present and what's missing.
    """
    analyser  = EssayAnalyser(essay, level, max_marks)
    checklist = analyser.get_checklist()
    predicted, note = analyser.quick_score()

    console.print(f"\n[bold cyan]Essay Analysis Report — {level} {max_marks}-Mark[/bold cyan]\n")

    # Stats row
    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value",  style="bold")
    stats_table.add_row("Word count",      str(checklist["word_count"]))
    stats_table.add_row("Paragraphs",      str(checklist["paragraph_count"]))
    stats_table.add_row("Economic concepts found", str(checklist["concept_count"]))
    stats_table.add_row("Analysis phrases", str(checklist["analysis_count"]))
    stats_table.add_row("Evaluation words", str(checklist["eval_word_count"]))
    stats_table.add_row("Predicted mark (heuristic)", f"~{predicted}/{max_marks}")
    console.print(stats_table)

    # Checklist
    checklist_items = [
        ("Has diagram reference",       checklist["has_diagram"],
         "Add a diagram (e.g. AD/AS) and explain what it shows"),
        ("Has real-world example",       checklist["has_real_example"],
         "Add a real country/policy/statistic as evidence"),
        ("Has conclusion/judgment",      checklist["has_conclusion"],
         "End with a supported final judgment answering the question"),
        ("Uses evaluation language",     checklist["eval_word_count"] >= 2,
         "Use: 'However', 'It depends on', 'In the long run'..."),
        ("Uses analysis chains",         checklist["analysis_count"] >= 2,
         "Use: 'This leads to', 'As a result', 'Therefore'..."),
        ("Uses economic terminology",    checklist["concept_count"] >= 3,
         "Use specific economic concepts relevant to the question"),
    ]

    check_table = Table(title="Examiner Checklist", box=box.ROUNDED, show_lines=True)
    check_table.add_column("Criterion",   style="white")
    check_table.add_column("Present?",    width=10)
    check_table.add_column("If missing",  style="dim")

    for label, present, advice in checklist_items:
        status = "[green]✓ Yes[/green]" if present else "[red]✗ No[/red]"
        check_table.add_row(label, status, "" if present else advice)

    console.print(check_table)

    # Keywords found
    if checklist["evaluation_words"]:
        console.print(
            f"\n[green]Evaluation words found:[/green] "
            + ", ".join(f"[italic]{w}[/italic]" for w in checklist["evaluation_words"][:8])
        )
    if checklist["economic_concepts"]:
        console.print(
            f"[green]Economic concepts found:[/green] "
            + ", ".join(f"[italic]{c}[/italic]" for c in checklist["economic_concepts"][:8])
        )


def show_model_answer_guide(question: str, level: str = "AS", max_marks: int = 12):
    """
    Prints a structural guide for how to answer a specific question.
    Does NOT write the answer — teaches the student how to approach it.
    """
    console.print(f"\n[bold cyan]How to Approach This Question[/bold cyan]")
    console.print(f"[dim]{question}[/dim]\n")

    key = "AS_12_mark" if level == "AS" else "IGCSE_8_mark"
    exp = EXAMINER_EXPECTATIONS.get(key, {})

    steps = exp.get("structure", [])
    for i, step in enumerate(steps, 1):
        console.print(f"  [bold cyan]{i}.[/bold cyan] {step}")

    console.print(
        f"\n[yellow]Tip:[/yellow] Cambridge examiners mark on DEPTH not breadth. "
        f"{'Two well-developed points beat five shallow ones.' if level == 'IGCSE' else 'Two fully developed analytical chains with evaluation beat three underdeveloped ones.'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — standalone demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import questionary

    console.print("\n[bold cyan]Feedback Engine — Cambridge Economics[/bold cyan]\n")

    mode = questionary.select(
        "What would you like to do?",
        choices=[
            "See what examiners expect (AS 12-mark)",
            "See what examiners expect (IGCSE 8-mark)",
            "View marking bands (AS)",
            "View marking bands (IGCSE)",
            "Analyse my essay",
        ],
    ).ask()

    if "AS 12-mark" in mode:
        show_examiner_expectations("AS", 12)
    elif "IGCSE 8-mark" in mode:
        show_examiner_expectations("IGCSE", 8)
    elif "AS)" in mode:
        show_marking_bands("AS")
    elif "IGCSE)" in mode:
        show_marking_bands("IGCSE")
    elif "Analyse" in mode:
        level     = questionary.select("Level?", choices=["AS", "IGCSE"]).ask()
        max_marks = 12 if level == "AS" else 8
        console.print("Paste your essay below. Press Enter twice then type END and press Enter:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        essay = "\n".join(lines)
        show_essay_analysis(essay, level, max_marks)