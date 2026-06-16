# src/config.py
# Cambridge Economics AI Examiner & Learning Platform
# Central configuration — all prompts, marking criteria, paths, model settings

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent
DATA_DIR         = BASE_DIR / "data"
RAW_ESSAYS_DIR   = DATA_DIR / "raw_essays"
PROCESSED_DIR    = DATA_DIR / "processed"
MODELS_DIR       = BASE_DIR / "models"
KNOWLEDGE_DIR    = DATA_DIR / "knowledge_base"   # Phase 5/6 — uploaded docs
ANALYTICS_DIR    = DATA_DIR / "analytics"        # Phase 9 — student tracking
TRAINING_FILE    = PROCESSED_DIR / "training_data.jsonl"
EVAL_FILE        = PROCESSED_DIR / "eval_data.jsonl"

for d in [RAW_ESSAYS_DIR, PROCESSED_DIR, MODELS_DIR, KNOWLEDGE_DIR, ANALYTICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
FINE_TUNED_MODEL_DIR = MODELS_DIR / "cambridge_grader_v1"

# ── Training — RTX 2050 4GB VRAM ─────────────────────────────────────────────
TRAINING_CONFIG = {
    "num_train_epochs":            3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate":               2e-4,
    "warmup_ratio":                0.05,
    "lr_scheduler_type":           "cosine",
    "max_seq_length":              1024,
    "save_steps":                  50,
    "logging_steps":               5,
    "eval_steps":                  50,
    "load_best_model_at_end":      True,
    "fp16":                        False,
    "bf16":                        True,
    "optim":                       "paged_adamw_8bit",
    "dataloader_pin_memory":       False,
    "group_by_length":             True,
}

LORA_CONFIG = {
    "r":              8,
    "lora_alpha":     16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout":   0.05,
    "bias":           "none",
    "task_type":      "CAUSAL_LM",
}

# ── AO Mark Allocations ───────────────────────────────────────────────────────
AO_MARKS = {
    "AS": {
        "AO1": {"name": "Knowledge & Understanding", "max": 2},
        "AO2": {"name": "Analysis",                  "max": 6},
        "AO3": {"name": "Evaluation",                "max": 4},
        "total": 12,
    },
    "IGCSE": {
        "AO1": {"name": "Knowledge",                      "max": 2},
        "AO2": {"name": "Development & Content",          "max": 6},
        "AO3": {"name": "Evaluation (brief, optional)",   "max": 0},
        "total": 8,
    },
}

# ── Marking Bands ─────────────────────────────────────────────────────────────
AS_MARKING_BANDS = {
    "12":   "Outstanding: Clear sustained argument, two+ fully developed SEDE chains, strong supported evaluative judgment with conditions/context. Top AO2 and AO3.",
    "10-11":"Strong: Two well-developed points, clear analysis chains, some evaluation with reasoning. Minor gap in judgment or development.",
    "8-9":  "Good: Two points, at least one well developed. Some analysis. Evaluation present but limited or unsupported.",
    "6-7":  "Adequate: One well-developed point or two partially developed. Some analysis, little or no evaluation.",
    "4-5":  "Weak: Basic points, mostly definitions, limited development, no real evaluation.",
    "2-3":  "Very weak: Simple statements, one or two relevant points, no development.",
    "1":    "Minimal: Single relevant point or definition only.",
    "0":    "No relevant economic content.",
}

IGCSE_MARKING_BANDS = {
    "8":   "Full marks: Two well-developed accepted points, clearly explained with examples. Brief evaluative comment present. All content matches Cambridge MS accepted answers.",
    "7":   "Strong: Two accepted points, both developed. Minor gap in example or evaluative comment.",
    "5-6": "Good: Two accepted points, at least one developed with explanation and example.",
    "3-4": "Adequate: One well-developed accepted point OR two basic accepted points.",
    "1-2": "Weak: One basic accepted point, definition-level only.",
    "0":   "No relevant content or all points outside accepted MS answers.",
}

# ── Examiner Expectations ─────────────────────────────────────────────────────
EXAMINER_EXPECTATIONS = {
    "AS_12_mark": {
        "structure": [
            "Introduction with clear thesis (1-2 sentences only)",
            "Point 1: State → Explain → Develop → Example (SEDE chain)",
            "Point 2: State → Explain → Develop → Example (SEDE chain)",
            "Evaluation: Weigh both sides, consider context/conditions/time horizons",
            "Conclusion: Supported final judgment answering 'to what extent'",
        ],
        "must_include": [
            "Economic terminology used correctly throughout",
            "At least one real-world example or data reference",
            "Evaluation language: 'However', 'It depends on', 'In the long run', 'To what extent'",
            "A final judgment that directly answers the question",
            "Diagram where it adds analytical value (not compulsory but rewarded)",
        ],
        "common_mistakes": [
            "Listing points without developing them (breadth not depth)",
            "Evaluation that just says 'on the other hand' without a reasoned judgment",
            "Not answering the specific context of the question",
            "Spending too long on definitions",
            "Weak evaluation — stating both sides without a supported final judgment",
        ],
    },
    "IGCSE_8_mark": {
        "structure": [
            "Point 1: State a clear accepted point (must match Cambridge MS)",
            "Explain: Simple direct explanation of why/how",
            "Example: Textbook or real-world example",
            "Point 2: Repeat for a second accepted point",
            "Optional: 1-2 sentence evaluative comment (minor marks only)",
        ],
        "must_include": [
            "Two points matching Cambridge IGCSE MS accepted answers",
            "Clear simple explanations — clarity rewarded over sophistication",
            "At least one example per point",
            "Syllabus-level economic vocabulary only",
        ],
        "common_mistakes": [
            "Writing points NOT in the Cambridge MS — original ideas score zero",
            "AS-level style evaluation — barely rewarded at IGCSE",
            "Three or four shallow points instead of two developed ones",
            "Over-complicating explanations",
            "Using concepts beyond the IGCSE syllabus",
        ],
    },
}

# ── System Prompt ─────────────────────────────────────────────────────────────
GRADING_SYSTEM_PROMPT = """You are an experienced Cambridge International Examinations (CIE) economics examiner.

CRITICAL RULES:
1. Always output EXACTLY the format requested — every ### heading, every field
2. Never skip sections or merge them
3. Always award a specific numeric mark, never a range
4. Never leave placeholders like [X] — always fill them in with real values
5. Base your mark on the essay content actually provided

FOR AS LEVEL: AO1=/2, AO2=/6, AO3=/4. Evaluation is critical and must be supported.
FOR IGCSE: AO1=/2, AO2=/6. Two accepted mark-scheme points + clarity = near full marks. Evaluation barely weighted."""


# ── Grading Prompts (strict format to fix output parsing) ────────────────────

AS_GRADING_TEMPLATE = """Grade this AS Level Economics essay as a Cambridge examiner.

QUESTION: {question}

ESSAY:
{essay}

OUTPUT EXACTLY THIS FORMAT — fill every field with real content, no placeholders:

MARK: [number]/12
BAND: [one sentence band description]
AO1_MARK: [number]/2
AO1_REASON: [one sentence explaining exactly why this AO1 mark was awarded]
AO2_MARK: [number]/6
AO2_REASON: [one sentence explaining exactly why this AO2 mark was awarded — are chains complete?]
AO3_MARK: [number]/4
AO3_REASON: [one sentence explaining exactly why this AO3 mark was awarded — is the judgment supported?]
CONFIDENCE: [High/Medium/Low]
IMPRESSION: [2-3 sentences overall examiner impression of this specific essay]
STRENGTH_1: [specific strength from the essay]
STRENGTH_2: [specific strength from the essay]
GAP_1: [specific thing that lost marks and why]
GAP_2: [specific thing that lost marks and why]
EVALUATION_QUALITY: [paragraph assessing evaluation specifically — was it supported? Did it answer to what extent?]
MODEL_EVAL: [Write a complete top-band evaluation paragraph for THIS specific question]
NEXT_BAND: [Specific actionable advice to reach the next mark band]"""


IGCSE_GRADING_TEMPLATE = """Grade this IGCSE Economics essay as a Cambridge examiner.

QUESTION: {question}

ESSAY:
{essay}

IGCSE RULES: Award marks for accepted Cambridge MS points only. Clarity rewarded. Evaluation barely weighted.

OUTPUT EXACTLY THIS FORMAT — fill every field with real content, no placeholders:

MARK: [number]/8
BAND: [one sentence band description]
AO1_MARK: [number]/2
AO1_REASON: [one sentence — are the points accepted MS answers? Clearly stated?]
AO2_MARK: [number]/6
AO2_REASON: [one sentence — are points explained clearly with examples? Simple and direct?]
AO3_MARK: 0/0
AO3_REASON: Evaluation not required at IGCSE — any brief comment noted but not penalised.
CONFIDENCE: [High/Medium/Low]
IMPRESSION: [2-3 sentences overall examiner impression]
POINT_1_STATUS: [ACCEPTED or NOT_ACCEPTED — state the point and why it is/isn't in the MS]
POINT_2_STATUS: [ACCEPTED or NOT_ACCEPTED — state the point and why it is/isn't in the MS]
STRENGTH_1: [specific strength]
STRENGTH_2: [specific strength]
GAP_1: [specific thing that lost marks]
GAP_2: [specific thing that lost marks]
MODEL_ANSWER: [Write what a full-mark 8/8 IGCSE response to this question looks like — two accepted points, clear explanation, example]
NEXT_BAND: [Simple direct advice to reach the next mark band]"""


EDIT_PROMPT_TEMPLATE = """You are a Cambridge economics examiner editing a student essay for maximum marks.

QUESTION: {question}
LEVEL: {level} ({max_marks} marks)

ORIGINAL ESSAY:
{essay}

{level_specific_instructions}

OUTPUT EXACTLY:

EDITED_ESSAY_START
[The improved essay — full text]
EDITED_ESSAY_END

CHANGES:
- [Change 1]: [Why this improves the mark]
- [Change 2]: [Why this improves the mark]
- [Change 3]: [Why this improves the mark]

PREDICTED_MARK: [number]/{max_marks}"""

AS_EDIT_INSTRUCTIONS = """AS LEVEL FOCUS:
- Complete every analytical chain: State → Explain → Develop → Example
- Strengthen evaluation with conditions, magnitude, time horizons and a clear judgment
- Add economic terminology where missing
- Diagrams optional — only add reference if it genuinely helps"""

IGCSE_EDIT_INSTRUCTIONS = """IGCSE FOCUS:
- Ensure both points match standard Cambridge mark scheme accepted answers
- Remove any original/creative points not in the MS — they score zero
- Make explanations simple and direct
- Add a clear example for each point
- Keep evaluation to 1-2 sentences maximum
- Remove AS-level concepts"""

KAE_ANALYSIS_PROMPT_TEMPLATE = """You are a Cambridge economics examiner analysing a student's essay plan.

QUESTION: {question}
LEVEL: {level} ({max_marks} marks)

STUDENT'S KNOWLEDGE POINTS (AO1 — {ao1_max} marks):
{knowledge_points}

STUDENT'S ANALYSIS POINTS (AO2 — {ao2_max} marks):
{analysis_points}

STUDENT'S EVALUATION POINTS (AO3 — {ao3_max} marks):
{evaluation_points}

OUTPUT EXACTLY:

AO1_SCORE: [number]/{ao1_max}
AO1_FEEDBACK: [What's strong, what terminology is missing]
AO2_SCORE: [number]/{ao2_max}
AO2_FEEDBACK: [Are chains complete? What needs developing?]
AO3_SCORE: [number]/{ao3_max}
AO3_FEEDBACK: [Are judgments supported? What's missing?]
TOTAL_PREDICTED: [number]/{max_marks}
FIX_1: [Most important fix for marks]
FIX_2: [Second most important fix]
FIX_3: [Third most important fix]
ESSAY_PLAN: [A top-band essay plan for this question based on the student's points]"""

ESSAY_GENERATOR_PROMPT = """You are a Cambridge economics examiner writing a model essay.

QUESTION: {question}
LEVEL: {level} ({max_marks} marks)
{extra_context}

Write a complete {mark_target}-mark Cambridge Economics essay following these rules:
{level_instructions}

OUTPUT EXACTLY:

ESSAY_START
[Full model essay]
ESSAY_END

EXAMINER_NOTES: [Brief notes on why this essay would achieve full marks]"""

AS_GENERATOR_INSTRUCTIONS = """- Two fully developed SEDE chains (State→Explain→Develop→Example)
- Strong supported evaluation with conditions and a clear final judgment answering 'to what extent'
- Correct economic terminology throughout
- Real-world example in each point
- Brief introduction with thesis, brief conclusion with judgment"""

IGCSE_GENERATOR_INSTRUCTIONS = """- Two accepted Cambridge mark scheme points
- Simple clear explanations — clarity over sophistication
- One example per point
- 1-2 sentence evaluative comment at the end
- Syllabus-level vocabulary only"""

EXAMINER_ASSISTANT_PROMPT = """You are a Cambridge Economics Examiner Assistant with deep knowledge of Cambridge International Economics marking standards.

CONTEXT FROM KNOWLEDGE BASE:
{retrieved_context}

STUDENT QUESTION: {question}

Answer clearly and specifically, citing the retrieved context where relevant. Focus on practical, actionable guidance."""