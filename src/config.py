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
    "8":   "Full marks: Two well-developed accepted points, each clearly explained with a relevant example. A brief evaluative comment present (1-2 sentences only). All content aligns with standard mark scheme accepted answers.",
    "7":   "Strong: Two developed accepted points with explanation. Minor gap in example or evaluative comment.",
    "5-6": "Good: Two accepted points, at least one developed with explanation and example. No evaluation required for this band.",
    "3-4": "Adequate: One well-developed accepted point OR two basic accepted points. Minimal development.",
    "1-2": "Weak: One basic accepted point. Definition-level only.",
    "0":   "No relevant content, or all points fall outside accepted mark scheme answers.",
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
            "Weak evaluation — stating both sides without a supported final judgment",
        ],
    },
    "IGCSE_8_mark": {
        "structure": [
            "Point 1: State a clear accepted point (must align with mark scheme)",
            "Explain: Simple, direct explanation of why/how — clarity rewarded over complexity",
            "Example: A textbook or real-world example to illustrate the point",
            "Point 2: Repeat for a second accepted point",
            "Optional: 1-2 sentence evaluative comment (minor marks only)",
        ],
        "must_include": [
            "Two distinct points that match standard Cambridge IGCSE mark scheme accepted answers",
            "Clear, simple explanation — examiners reward directness over sophistication",
            "At least one example per point",
            "Syllabus-level economic vocabulary (not AS-level concepts)",
        ],
        "common_mistakes": [
            "Writing points NOT in the Cambridge mark scheme — original ideas score zero at IGCSE",
            "Writing AS-level style deep evaluation — IGCSE barely rewards this",
            "Three or four shallow points instead of two well-developed ones",
            "Over-complicating explanations — keep it clear and direct",
            "Using concepts beyond the IGCSE syllabus",
            "Spending time on evaluation when content points are incomplete",
        ],
    },
}

# ── Prompt Templates ──────────────────────────────────────────────────────────
GRADING_SYSTEM_PROMPT = """You are an experienced Cambridge International Examinations (CIE) economics examiner with 15+ years of experience marking both AS Level and IGCSE Economics papers.

You apply DIFFERENT standards depending on the level:

FOR AS LEVEL (12-mark questions):
- Evaluation (AO3) is critical and heavily weighted — must show supported judgment, consider conditions/magnitude/time horizons
- "To what extent" and "it depends on..." with reasoning is expected
- Analysis chains must be fully developed: State → Explain → Develop → Example
- Original thought and independent reasoning is rewarded
- Diagrams are useful but not compulsory

FOR IGCSE (8-mark questions):
- Content accuracy is paramount — points must match Cambridge mark scheme accepted answers
- Original ideas NOT in the mark scheme score ZERO — creativity is penalised
- Evaluation is barely weighted — 1-2 sentences at the end is sufficient, more is a waste of time
- Clarity beats complexity — simple, direct explanations score better than sophisticated ones
- Two well-developed accepted points with examples = near full marks
- Syllabus-level vocabulary only — AS-level concepts are out of scope

You always model what a perfect answer looks like for the specific question asked."""


AS_GRADING_TEMPLATE = """## AS Level Essay to Grade

**Question (12 marks):** {question}

**Student's Essay:**
{essay}

---

Grade this AS Level essay. Use this EXACT format:

### MARK AWARDED: [X]/12

### MARK BAND: [Band description]

### WHAT THE EXAMINER SEES
[2-3 sentences — overall impression, specific to this essay]

### MARKS BREAKDOWN
**Knowledge & Understanding (AO1):** [X]/2 — [specific reason]
**Analysis (AO2):** [X]/6 — [is the chain complete? State→Explain→Develop→Example? Are mechanisms clear?]
**Evaluation (AO3):** [X]/4 — [does the student make a supported judgment? Do they consider conditions/context/magnitude/time horizons?]

### EVALUATION QUALITY
[Assess the evaluation specifically:
- What evaluative points were made?
- Were they supported with reasoning or just asserted?
- Did the student reach a clear final judgment that answers "to what extent"?
- What is missing for top AO3 marks?]

### ✳ MODEL EVALUATION POINT
[Write a complete, top-band evaluation paragraph for THIS specific question. Include: a clear judgment, the conditions it depends on, the reasoning, and a direct link back to the question. Show exactly what Cambridge AS examiners want to see.]

### STRENGTHS
- [Specific strength from the essay]
- [Specific strength 2]

### WHAT LOST MARKS
- [Specific gap with reason]
- [Specific gap 2]

### HOW TO REACH THE NEXT BAND
[Exact, actionable advice]
"""

IGCSE_GRADING_TEMPLATE = """## IGCSE Essay to Grade

**Question (8 marks):** {question}

**Student's Essay:**
{essay}

---

Grade this IGCSE essay. Remember: IGCSE rewards mark-scheme accuracy and clarity, NOT original thinking or deep evaluation.

Use this EXACT format:

### MARK AWARDED: [X]/8

### MARK BAND: [Band description]

### WHAT THE EXAMINER SEES
[2-3 sentences — overall impression, specific to this essay]

### MARKS BREAKDOWN
**Content & Knowledge (AO1):** [X]/2 — [are the points accepted IGCSE mark scheme answers? Are they clearly stated?]
**Development & Explanation (AO2):** [X]/4 — [are points explained clearly and simply? Is there an example? Is the explanation direct?]
**Evaluative Comment (AO3):** [X]/2 — [brief evaluative comment present? 1-2 sentences is all that's needed at IGCSE]

### CONTENT ACCURACY CHECK
[Critical for IGCSE — state whether each point the student made is:
✓ ACCEPTED — matches standard Cambridge IGCSE mark scheme answers
✗ NOT ACCEPTED — original or off-syllabus point that scores zero
Flag any points that would not appear in a Cambridge MS]

### CLARITY ASSESSMENT
[IGCSE rewards simple, direct explanations. Is the student's explanation clear and easy to follow? Or is it unnecessarily complex?]

### ✳ WHAT A FULL-MARK ANSWER LOOKS LIKE
[Write two accepted points with clear explanation and example, showing exactly what an 8/8 IGCSE response to this question would look like. Keep it simple and direct.]

### WHAT LOST MARKS
- [Specific gap — was it an off-MS point, unclear explanation, missing example?]
- [Specific gap 2]

### HOW TO REACH THE NEXT BAND
[Simple, direct advice for IGCSE students — focus on accepted points and clarity]
"""

# Keep a single template reference for backwards compat — dynamically chosen in grader.py
GRADING_PROMPT_TEMPLATE = AS_GRADING_TEMPLATE  # default; grader.py picks the right one

EDIT_PROMPT_TEMPLATE = """You are a Cambridge economics examiner and expert essay editor.

A student has written the following essay for this question:

**Question:** {question}
**Level:** {level} ({max_marks} marks)

**Original Essay:**
{essay}

Your task: Rewrite this essay to achieve the highest possible mark band.

{level_specific_instructions}

Format your response as:

### EDITED ESSAY
[The improved essay — full text]

### WHAT WAS CHANGED AND WHY
- [Change 1]: [Why this improves the mark]
- [Change 2]: [Why this improves the mark]

### PREDICTED MARK AFTER EDITING: [X]/{max_marks}
"""

AS_EDIT_INSTRUCTIONS = """For AS Level, focus on:
1. Keeping the student's original ideas where strong
2. Completing every analytical chain: State → Explain → Develop → Example
3. Making evaluation genuinely evaluative — add conditions, magnitude, time horizons, and a clear final judgment
4. Strengthening economic terminology
5. Diagrams optional — only add reference if it genuinely strengthens the argument
6. NOT replacing the student's voice entirely — improve, don't rewrite from scratch"""

IGCSE_EDIT_INSTRUCTIONS = """For IGCSE, focus on:
1. Ensuring both points match standard Cambridge mark scheme accepted answers — remove any original/off-syllabus points
2. Making explanations simple and direct — clarity is rewarded over sophistication
3. Adding a clear example for each point
4. Keeping evaluation to 1-2 sentences only — do not add AS-level style evaluation
5. Removing any concepts beyond the IGCSE syllabus
6. Keeping language at IGCSE level throughout"""

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