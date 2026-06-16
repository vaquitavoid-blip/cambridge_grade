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

# ── Sequence length ───────────────────────────────────────────────────────────
# IMPORTANT: a full AS essay (400-600 words ~ 550-800 tokens) + the grading
# template + system prompt easily exceeds 1024 tokens. If the prompt is
# truncated to 1024 at inference, the END of the essay (often the
# conclusion/evaluation) gets cut off before the model ever sees it, which
# leads to incomplete/garbled grading output that the UI then fails to parse.
#
# 2048 fits comfortably on 4GB VRAM with 4-bit quantization + LoRA r=8 for
# this 1.5B model. This value is used for BOTH training (max_seq_length) and
# inference (tokenizer truncation) so the model is never asked to handle
# inputs longer than it was trained on.
MAX_SEQ_LENGTH = 2048

# ── Training Hyperparameters — tuned for RTX 2050 4GB VRAM ──────────────────
TRAINING_CONFIG = {
    "num_train_epochs":              3,
    "per_device_train_batch_size":   1,      # 4GB VRAM — keep at 1
    "gradient_accumulation_steps":   8,      # effective batch size = 8
    "learning_rate":                 2e-4,
    "warmup_ratio":                  0.05,
    "lr_scheduler_type":             "cosine",
    "max_seq_length":                MAX_SEQ_LENGTH,
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

# ── AO mark allocations by level ──────────────────────────────────────────────
# Single source of truth — used by grader.py, data_prep.py, grading_app.py,
# evaluate_model.py and run_grader.py so the AO split can never drift between
# training data, prompts, and the UI.
AO_MARKS = {
    "AS":    {"ao1": 2, "ao2": 6, "ao3": 4, "total": 12},
    "IGCSE": {"ao1": 2, "ao2": 4, "ao3": 2, "total": 8},
}

# Display labels for each AO, per level — also the single source of truth
# for the headings the model is trained to produce AND the headings
# grading_app.py searches for. Keeping these identical end-to-end is what
# makes the marks breakdown reliably parseable.
AO_LABELS = {
    "AS": {
        "ao1": "Knowledge & Understanding (AO1)",
        "ao2": "Analysis (AO2)",
        "ao3": "Evaluation (AO3)",
    },
    "IGCSE": {
        "ao1": "Content & Knowledge (AO1)",
        "ao2": "Development & Explanation (AO2)",
        "ao3": "Evaluative Comment (AO3)",
    },
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
<<<<<<< HEAD
    "8":   "Full marks: Two well-developed accepted points, each clearly explained with a relevant example. A brief evaluative comment present (1-2 sentences only). All content aligns with standard mark scheme accepted answers.",
    "7":   "Strong: Two developed accepted points with explanation. Minor gap in example or evaluative comment.",
    "5-6": "Good: Two accepted points, at least one developed with explanation and example. No evaluation required for this band.",
    "3-4": "Adequate: One well-developed accepted point OR two basic accepted points. Minimal development.",
    "1-2": "Weak: One basic accepted point. Definition-level only.",
    "0":   "No relevant content, or all points fall outside accepted mark scheme answers.",
=======
    "8":   "Full marks: Two well-developed accepted points with clear economic reasoning and relevant examples. Only brief evaluation is required.",
    "7":   "Strong: Two developed points with clear explanation and examples. Minor weakness in development or evaluative comment.",
    "5-6": "Good: Two relevant points, at least one developed with explanation and example. Evaluation may be limited or absent.",
    "3-4": "Adequate: One developed point OR two basic points. Limited explanation.",
    "1-2": "Weak: One basic relevant point. Mostly definition-level response.",
    "0":   "No relevant economic content.",
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
}

# ── IGCSE Mark Scheme Bank ─────────────────────────────────────────────────────
# Cambridge IGCSE Economics rewards points that match standard mark-scheme
# "accepted answers" almost exclusively. This bank gives the grader (and the
# content validator / feedback engine) a structured reference of commonly
# accepted points by topic, so IGCSE feedback can say WHICH accepted points
# were hit and which were missed — not just "looks economics-y".
#
# This is a starting reference set covering core IGCSE syllabus topics.
# It is intentionally generic (not transcribed from any specific past paper)
# — extend it with topic/question-specific accepted points as real mark
# schemes are added (see scripts/add_essay.py "topic" field).
IGCSE_MARK_SCHEME_BANK = {
    "Price System": {
        "accepted_points": [
            "Price acts as a signal/incentive to producers and consumers",
            "Price mechanism allocates scarce resources between competing uses",
            "Changes in demand/supply cause price to adjust towards equilibrium",
            "Shortages cause prices to rise; surpluses cause prices to fall",
        ],
        "common_examples": ["food prices", "fuel/oil prices", "housing market"],
    },
    "Market Failure": {
        "accepted_points": [
            "Externalities — third parties affected without compensation (positive/negative)",
            "Public goods are non-excludable and non-rivalrous, leading to under-provision",
            "Merit goods are under-consumed due to imperfect information",
            "Demerit goods are over-consumed due to imperfect information",
            "Government intervention: taxation, subsidies, regulation, provision of public goods",
        ],
        "common_examples": ["pollution from factories", "education", "healthcare", "smoking/alcohol taxes"],
    },
    "Microeconomics": {
        "accepted_points": [
            "Opportunity cost — the next best alternative forgone",
            "Division of labour increases productivity through specialisation",
            "Economies of scale lower average costs of production as output rises",
            "Firms may aim for profit maximisation, growth, or market share",
        ],
        "common_examples": ["car manufacturing", "small vs large firms", "supermarkets"],
    },
    "Macroeconomics": {
        "accepted_points": [
            "Inflation reduces purchasing power and erodes savings",
            "Unemployment leads to lost output and lower government tax revenue",
            "Economic growth increases living standards via higher real GDP",
            "Fiscal policy: government spending and taxation to manage the economy",
            "Monetary policy: changing interest rates to influence borrowing/spending",
        ],
        "common_examples": ["interest rate changes", "government spending on infrastructure", "income tax changes"],
    },
    "International Economics": {
        "accepted_points": [
            "Specialisation and trade allow countries to consume beyond domestic production possibilities",
            "Tariffs/quotas protect domestic industries but raise prices for consumers",
            "A weaker exchange rate makes exports cheaper and imports more expensive",
            "A current account deficit means imports of goods/services exceed exports",
        ],
        "common_examples": ["import tariffs on steel", "exchange rate depreciation", "trade between developed/developing countries"],
    },
    "Development Economics": {
        "accepted_points": [
            "Developing countries often have lower GDP per capita and living standards",
            "Indicators of development: GDP per capita, literacy rate, life expectancy, HDI",
            "Barriers to development: lack of infrastructure, education, capital, debt",
            "Foreign direct investment can boost growth but profits may leave the country",
        ],
        "common_examples": ["sub-Saharan African economies", "remittances", "aid programmes"],
    },
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
<<<<<<< HEAD
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
=======
   "IGCSE_8_mark": {
    "structure": [
        "Point 1: State the point clearly",
        "Explain: Why/how does this happen?",
        "Develop: Consequence or impact",
        "Example: Real-world or textbook example",
        "Point 2: Repeat the above",
        "Optional brief judgment at the end",
    ],
    "must_include": [
        "Two distinct economic points",
        "Clear economic vocabulary",
        "At least one relevant example",
        "Simple and accurate explanation",
    ],
    "common_mistakes": [
        "Writing several shallow points instead of two developed points",
        "Missing examples",
        "Repeating the question",
        "Weak explanation of economic reasoning",
        "Overly complex AS-level evaluation",
    ],
  },
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)
}

# ── Prompt Templates ──────────────────────────────────────────────────────────
GRADING_SYSTEM_PROMPT = """You are an experienced Cambridge International Examinations (CIE) economics examiner with 15+ years of experience marking both AS Level and IGCSE Economics papers.

<<<<<<< HEAD
You apply DIFFERENT standards depending on the level:

FOR AS LEVEL (12-mark questions):
- Evaluation (AO3) is critical and heavily weighted — must show supported judgment, consider conditions/magnitude/time horizons
- "To what extent" and "it depends on..." with reasoning is expected
- Analysis chains must be fully developed: State → Explain → Develop → Example
- Original thought and independent reasoning is rewarded
- Diagrams are useful but not compulsory
=======
IMPORTANT DIFFERENCE BETWEEN LEVELS:

FOR AS LEVEL:
- Evaluation (AO3) carries significant weight.
- Strong supported judgments are expected.
- Students should consider conditions, assumptions, short-run vs long-run effects and magnitude.
- Weak evaluation should be penalised.

FOR IGCSE:
- Content and explanation are far more important than evaluation.
- A brief evaluative comment is sufficient.
- Full marks can be awarded with minimal evaluation if content and development are excellent.
- Do NOT apply AS Level evaluation standards to IGCSE essays.
- Reward clear explanations, accurate economics and relevant examples.

You always:
- Award marks based on Cambridge standards.
- Explain specifically why marks were awarded or deducted.
- Give actionable examiner-style feedback.
- Identify strengths and weaknesses.
- Suggest how to improve the answer."""
>>>>>>> 37ceb5e6 (Add Phase 1-10: RAG, analytics, grading platform)

FOR IGCSE (8-mark questions):
- Content accuracy is paramount — points must match Cambridge mark scheme accepted answers
- Original ideas NOT in the mark scheme score ZERO — creativity is penalised
- Evaluation is barely weighted — 1-2 sentences at the end is sufficient, more is a waste of time
- Clarity beats complexity — simple, direct explanations score better than sophisticated ones
- Two well-developed accepted points with examples = near full marks
- Syllabus-level vocabulary only — AS-level concepts are out of scope

You ALWAYS award a specific numeric mark — never a placeholder, range, or "X". You always model what a perfect answer looks like for the specific question asked."""


AS_GRADING_TEMPLATE = """## AS Level Essay to Grade

**Question (12 marks):** {question}

**Student's Essay:**
{essay}

---

Grade this AS Level essay. Use this EXACT format. Replace every bracketed
placeholder with real content — marks must be actual numbers, never "X".

### MARK AWARDED: <number>/12

### MARK BAND: <band description>

### WHAT THE EXAMINER SEES
<2-3 sentences — overall impression, specific to this essay>

### MARKS BREAKDOWN
**Knowledge & Understanding (AO1):** <number>/2 — <specific reason>
**Analysis (AO2):** <number>/6 — <is the chain complete? State→Explain→Develop→Example? Are mechanisms clear?>
**Evaluation (AO3):** <number>/4 — <does the student make a supported judgment? Do they consider conditions/context/magnitude/time horizons?>

### EVALUATION QUALITY
<Assess the evaluation specifically:
- What evaluative points were made?
- Were they supported with reasoning or just asserted?
- Did the student reach a clear final judgment that answers "to what extent"?
- What is missing for top AO3 marks?>

### MODEL EVALUATION POINT
<Write a complete, top-band evaluation paragraph for THIS specific question. Include: a clear judgment, the conditions it depends on, the reasoning, and a direct link back to the question. Show exactly what Cambridge AS examiners want to see.>

### STRENGTHS
- <specific strength from the essay>
- <specific strength 2>

### WHAT LOST MARKS
- <specific gap with reason>
- <specific gap 2>

### HOW TO REACH THE NEXT BAND
<exact, actionable advice>
"""

IGCSE_GRADING_TEMPLATE = """## IGCSE Essay to Grade

**Question (8 marks):** {question}

**Student's Essay:**
{essay}

---

Grade this IGCSE essay. Remember: IGCSE rewards mark-scheme accuracy and clarity, NOT original thinking or deep evaluation.

Use this EXACT format. Replace every bracketed placeholder with real content
— marks must be actual numbers, never "X".

### MARK AWARDED: <number>/8

### MARK BAND: <band description>

### WHAT THE EXAMINER SEES
<2-3 sentences — overall impression, specific to this essay>

### MARKS BREAKDOWN
**Content & Knowledge (AO1):** <number>/2 — <are the points accepted IGCSE mark scheme answers? Are they clearly stated?>
**Development & Explanation (AO2):** <number>/4 — <are points explained clearly and simply? Is there an example? Is the explanation direct?>
**Evaluative Comment (AO3):** <number>/2 — <brief evaluative comment present? 1-2 sentences is all that's needed at IGCSE>

### CONTENT ACCURACY CHECK
<Critical for IGCSE — state whether each point the student made is:
ACCEPTED — matches standard Cambridge IGCSE mark scheme answers
NOT ACCEPTED — original or off-syllabus point that scores zero
Flag any points that would not appear in a Cambridge MS>

### CLARITY ASSESSMENT
<IGCSE rewards simple, direct explanations. Is the student's explanation clear and easy to follow? Or is it unnecessarily complex?>

### MODEL EVALUATION POINT
<Write two accepted points with clear explanation and example, showing exactly what an 8/8 IGCSE response to this question would look like. Keep it simple and direct.>

### STRENGTHS
- <specific strength from the essay>
- <specific strength 2>

### WHAT LOST MARKS
- <specific gap — was it an off-MS point, unclear explanation, missing example?>
- <specific gap 2>

### HOW TO REACH THE NEXT BAND
<simple, direct advice for IGCSE students — focus on accepted points and clarity>
"""

GRADING_TEMPLATES = {
    "AS":    AS_GRADING_TEMPLATE,
    "IGCSE": IGCSE_GRADING_TEMPLATE,
}


def get_grading_template(level: str) -> str:
    """Returns the correct grading template for a level ('AS' or 'IGCSE').

    This is the SINGLE place that decides which template is used — both
    data_prep.py (when building training samples) and grader.py (at
    inference) call this, so the model is always trained and queried with
    the exact same prompt structure per level.
    """
    return GRADING_TEMPLATES.get(level.upper(), AS_GRADING_TEMPLATE)


# Backwards-compat alias (older scripts referenced this name)
GRADING_PROMPT_TEMPLATE = AS_GRADING_TEMPLATE

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
<the improved essay — full text>

### WHAT WAS CHANGED AND WHY
- <change 1>: <why this improves the mark>
- <change 2>: <why this improves the mark>

### PREDICTED MARK AFTER EDITING: <number>/{max_marks}
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

Your task: Analyse each section and give a detailed gap analysis. Replace
every bracketed placeholder with a real number — never "X".

### KNOWLEDGE (AO1) ASSESSMENT
Score: <number>/{ao1_max}
<What's strong, what's missing, what terminology should be added>

### ANALYSIS (AO2) ASSESSMENT
Score: <number>/{ao2_max}
<Are the chains complete? State→Explain→Develop→Example? Which points need developing further?>

### EVALUATION (AO3) ASSESSMENT
Score: <number>/{ao3_max}
<Are the judgments supported? Do they answer "to what extent"? What's missing?>

### OVERALL PREDICTED MARK: <number>/{max_marks}

### PRIORITY FIXES (in order of mark impact)
1. <most important fix>
2. <second most important>
3. <third>

### WHAT A COMPLETE ANSWER LOOKS LIKE
<Brief outline of what a top-band answer to this question would include across all three AOs>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FORMATTING HELPERS
# Used by BOTH data_prep.py (to build training targets) and grading_app.py
# (to render the UI). Keeping the exact wording/heading text identical
# between training and inference is what makes the model's output reliably
# parseable by the UI.
# ─────────────────────────────────────────────────────────────────────────────

def format_marks_breakdown(level: str, ao1: int, ao2: int, ao3: int,
                            ao1_note: str = "", ao2_note: str = "", ao3_note: str = "") -> str:
    """Builds the '### MARKS BREAKDOWN' body text for a given level and AO marks.

    Produces lines in the exact form:
        **<AO Label>:** <mark>/<max> — <note>
    which is the same form the model is trained to produce, and the same
    form grading_app.py's regex looks for.
    """
    level = level.upper()
    ao = AO_MARKS.get(level, AO_MARKS["AS"])
    labels = AO_LABELS.get(level, AO_LABELS["AS"])

    lines = []
    for key, value, note in [("ao1", ao1, ao1_note), ("ao2", ao2, ao2_note), ("ao3", ao3, ao3_note)]:
        line = f"**{labels[key]}:** {value}/{ao[key]}"
        if note:
            line += f" — {note}"
        lines.append(line)
    return "\n".join(lines)


def get_mark_band(level: str, mark: int) -> str:
    """Returns the marking-band description for a given level and mark."""
    bands = AS_MARKING_BANDS if level.upper() == "AS" else IGCSE_MARKING_BANDS
    for band_range, description in bands.items():
        if "-" in band_range:
            low, high = map(int, band_range.split("-"))
            if low <= mark <= high:
                return description
        else:
            if mark == int(band_range):
                return description
    return "Unknown band"