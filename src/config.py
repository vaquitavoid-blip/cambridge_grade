# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Cambridge Economics Grader
# Tuned for: RTX 2050 (4GB VRAM) + 12GB RAM
# Base model: Qwen2.5-1.5B-Instruct — best quality that fits in 4GB VRAM
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
RAW_ESSAYS_DIR  = DATA_DIR / "raw_essays"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = BASE_DIR / "models"
TRAINING_FILE   = PROCESSED_DIR / "training_data.jsonl"
EVAL_FILE       = PROCESSED_DIR / "eval_data.jsonl"

for d in [RAW_ESSAYS_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Base Model ────────────────────────────────────────────────────────────────
# Qwen2.5-1.5B-Instruct:
#   - 1.5B parameters → fits in 4GB VRAM with 4-bit quantization
#   - Strong instruction-following for structured grading output
#   - Fast download (~3GB), fast training on RTX 2050
#   - Much better than Phi/TinyLlama for long-form structured text
#
# Upgrade path (if you get access to more VRAM later):
#   4-8GB  → "Qwen/Qwen2.5-3B-Instruct"
#   8-16GB → "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
FINE_TUNED_MODEL_DIR = MODELS_DIR / "cambridge_grader_v1"

# ── Training Hyperparameters — tuned for RTX 2050 4GB VRAM ──────────────────
TRAINING_CONFIG = {
    "num_train_epochs":              3,
    "per_device_train_batch_size":   1,      # 4GB VRAM — keep at 1
    "gradient_accumulation_steps":   8,      # effective batch size = 8
    "learning_rate":                 2e-4,
    "warmup_ratio":                  0.05,
    "lr_scheduler_type":             "cosine",
    "max_seq_length":                1024,   # reduced from 2048 to fit VRAM
    "save_steps":                    50,
    "logging_steps":                 5,
    "eval_steps":                    50,
    "load_best_model_at_end":        True,
    "fp16":                          False,  # RTX 2050 supports bf16 — more stable
    "bf16":                          True,
    "optim":                         "paged_adamw_8bit",
    "dataloader_pin_memory":         False,  # saves RAM on 12GB system
    "group_by_length":               True,   # speeds up training by grouping similar lengths
}

# ── LoRA Config — tuned for 4GB VRAM ─────────────────────────────────────────
LORA_CONFIG = {
    "r":              8,        # reduced from 16 — saves VRAM, enough for this task
    "lora_alpha":     16,       # typically 2x rank
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],  # Qwen uses MLP projections too
    "lora_dropout":   0.05,
    "bias":           "none",
    "task_type":      "CAUSAL_LM",
}

# ── Cambridge Marking Criteria ────────────────────────────────────────────────
AS_MARKING_BANDS = {
    "12":   "Outstanding: Clear, sustained argument. Two+ well-developed analytical points with strong chain of reasoning. Evaluates with a supported final judgment. Uses relevant economic concepts correctly. Diagrams used effectively where appropriate.",
    "10-11": "Strong: Good analytical development. Two points clearly developed. Some evaluation present. Minor gaps in the chain of reasoning or judgment.",
    "8-9":  "Good: Two points attempted. At least one well developed with analysis. Limited or weak evaluation.",
    "6-7":  "Adequate: Some analytical development. May have one well-developed point. Evaluation is superficial or missing.",
    "4-5":  "Weak: Basic analytical points. Mostly definitions and assertions. Little development. No real evaluation.",
    "2-3":  "Very weak: Simple statements. One or two relevant points but no development. Possible definition.",
    "1":    "Minimal: A single relevant point or definition. No development.",
    "0":    "No relevant economic content.",
}

IGCSE_MARKING_BANDS = {
    "8":   "Full marks: Two well-developed points with economic reasoning and real-world examples. Clear evaluation with supported judgment.",
    "7":   "Strong: Two well-developed analytical points. Evaluation present but judgment may lack support.",
    "5-6": "Good: Two points, at least one with development. Limited evaluation.",
    "3-4": "Adequate: One well-developed point OR two basic points. Minimal evaluation.",
    "1-2": "Weak: One basic point. Mostly definition-level. No analysis.",
    "0":   "No relevant content.",
}

# ── Examiner Expectations ─────────────────────────────────────────────────────
EXAMINER_EXPECTATIONS = {
    "AS_12_mark": {
        "structure": [
            "Introduction with clear thesis (1-2 sentences only)",
            "Point 1: State → Explain → Develop → Example (SEDE chain)",
            "Point 2: State → Explain → Develop → Example (SEDE chain)",
            "Evaluation: Weigh both sides, consider context/assumptions",
            "Conclusion: Supported final judgment (not a summary)",
        ],
        "must_include": [
            "Economic terminology used correctly throughout",
            "At least one real-world example or data reference",
            "Explicit evaluation language: 'However', 'It depends on', 'In the long run', 'To what extent'",
            "A judgment that directly answers the exact question asked",
            "A relevant diagram where it adds analytical value (e.g. AD/AS, supply/demand) — not compulsory but rewarded",
        ],
        "common_mistakes": [
            "Listing points without developing them (breadth not depth)",
            "Evaluation that just says 'on the other hand' without a reasoned judgment",
            "Describing a diagram instead of using it to explain a mechanism",
            "Not answering the specific context of the question",
            "Spending too long on definitions",
            "Weak evaluation — stating both sides without making a supported final judgment",
            "Ignoring the specific word in the question: 'evaluate', 'assess', 'discuss' each need different approaches",
        ],
    },
    "IGCSE_8_mark": {
        "structure": [
            "Point 1: State the point clearly",
            "Explain: Why/how does this happen? (economic logic)",
            "Develop: What is the consequence? (chain of reasoning)",
            "Example: Real-world evidence",
            "Point 2: Repeat the above",
            "Evaluation: Brief judgment on which effect is more significant and why",
        ],
        "must_include": [
            "Two distinct analytical points",
            "Economic vocabulary (elasticity, incentive, opportunity cost, etc.)",
            "At least one real-world example",
            "A brief evaluative comment at the end",
        ],
        "common_mistakes": [
            "Writing three or four shallow points instead of two deep ones",
            "Forgetting the evaluation at the end",
            "Using examples without linking them to economic theory",
            "Repeating the question or writing a long introduction",
        ],
    },
}

# ── Prompt Templates ──────────────────────────────────────────────────────────
GRADING_SYSTEM_PROMPT = """You are an experienced Cambridge International Examinations (CIE) economics examiner with 15+ years of experience marking AS Level and IGCSE Economics papers.

You grade essays with precision, consistency, and detailed explanatory feedback. You know exactly what Cambridge examiners look for: analytical depth, evaluative judgment, correct economic terminology, and the ability to apply theory to real-world contexts.

Key principles you always apply:
- Diagrams are rewarded where relevant but are NOT compulsory — an essay with strong analysis and evaluation but no diagram can still score full marks
- Evaluation (AO3) is the hardest mark to earn and must be genuinely evaluative — not just "on the other hand" but a supported judgment considering magnitude, context, assumptions, or time horizons
- A top-band evaluation explicitly answers "to what extent" or "in what circumstances" the argument holds
- You always model what a perfect evaluation point looks like for the specific question asked"""

GRADING_PROMPT_TEMPLATE = """## Essay to Grade

**Exam Level:** {level}
**Question Type:** {question_type} ({max_marks} marks)
**Question:** {question}

**Student's Essay:**
{essay}

---

## Your Task

Grade this essay as a Cambridge examiner. Use this EXACT format:

### MARK AWARDED: [X]/{max_marks}

### MARK BAND: [Band description]

### WHAT THE EXAMINER SEES
[2-3 sentences describing the overall impression — be specific to this essay]

### MARKS BREAKDOWN
**Knowledge & Understanding (AO1):** [X marks] — [specific reason referencing the essay]
**Analysis (AO2):** [X marks] — [specific reason — is the chain of reasoning complete? Are mechanisms explained?]
**Evaluation (AO3):** [X marks] — [specific reason — does the student make a supported judgment? Do they consider conditions/context/magnitude?]

### EVALUATION QUALITY
[Detailed assessment of the evaluation specifically:
- What evaluative points were made?
- Were they supported with reasoning?
- Did the student reach a clear final judgment?
- What's missing to reach the top evaluation band?]

### ✳ MODEL EVALUATION POINT
[Write a complete, top-band evaluation paragraph for THIS specific question — show exactly what Cambridge examiners want to see. Include: a clear judgment, the conditions it depends on, a reason why, and a link back to the question. This is what the student should aim to write.]

### STRENGTHS
- [Specific strength 1 — quote or reference from the essay]
- [Specific strength 2]

### WHAT LOST MARKS
- [Specific gap 1 — what was missing and exactly why it cost marks]
- [Specific gap 2]

### HOW TO REACH THE NEXT BAND
[Concrete, actionable advice — what exact changes would push this to the next mark band]
"""

EDIT_PROMPT_TEMPLATE = """You are a Cambridge economics examiner and expert essay editor.

A student has written the following essay for this question:

**Question:** {question}
**Level:** {level} ({max_marks} marks)

**Original Essay:**
{essay}

Your task: Rewrite this essay to achieve the highest possible mark band while:
1. Keeping the student's original ideas and structure where strong
2. Strengthening every analytical chain (State → Explain → Develop → Example)
3. Making evaluation genuinely evaluative — add conditions, magnitude, context, and a clear final judgment
4. Adding economic terminology where missing
5. Making diagrams optional — only reference one if it genuinely strengthens the argument
6. NOT changing the student's voice completely — improve, don't replace

Show your edits clearly. Format:

### EDITED ESSAY
[The improved essay — full text]

### WHAT WAS CHANGED AND WHY
- [Change 1]: [Why this improves the mark]
- [Change 2]: [Why this improves the mark]
[Continue for all significant changes]

### PREDICTED MARK AFTER EDITING: [X]/{max_marks}
"""

KAE_ANALYSIS_PROMPT_TEMPLATE = """You are a Cambridge economics examiner.

A student is planning an essay response for this question:
**Question:** {question}
**Level:** {level} ({max_marks} marks)

They have entered their planned points below:

**KNOWLEDGE points (what they know / definitions / concepts):**
{knowledge_points}

**ANALYSIS points (how/why — chains of reasoning):**
{analysis_points}

**EVALUATION points (judgments, conditions, final answer):**
{evaluation_points}

Your task: Analyse each section and give a detailed gap analysis.

### KNOWLEDGE (AO1) ASSESSMENT
[Score: X/{ao1_max}]
[What's strong, what's missing, what terminology should be added]

### ANALYSIS (AO2) ASSESSMENT  
[Score: X/{ao2_max}]
[Are the chains complete? State→Explain→Develop→Example? Which points need developing further?]

### EVALUATION (AO3) ASSESSMENT
[Score: X/{ao3_max}]
[Are the judgments supported? Do they answer "to what extent"? What's missing?]

### OVERALL PREDICTED MARK: [X]/{max_marks}

### PRIORITY FIXES (in order of mark impact)
1. [Most important fix]
2. [Second most important]
3. [Third]

### WHAT A COMPLETE ANSWER LOOKS LIKE
[Brief outline of what a top-band answer to this question would include across all three AOs]
"""