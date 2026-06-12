# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Cambridge Economics Grader
# Edit this file to change model, paths, or marking criteria
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

# Create dirs if they don't exist
for d in [RAW_ESSAYS_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Base Model ────────────────────────────────────────────────────────────────
# We use Mistral-7B-Instruct — strong reasoning, runs on most laptops (with 4-bit)
# Alternatives if you have more RAM/GPU:
#   "meta-llama/Llama-3-8B-Instruct"  (needs HuggingFace token)
#   "google/flan-t5-large"             (smaller, less capable)
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
FINE_TUNED_MODEL_DIR = MODELS_DIR / "cambridge_grader_v1"

# ── Training Hyperparameters ──────────────────────────────────────────────────
TRAINING_CONFIG = {
    "num_train_epochs":       3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,   # Effective batch size = 8
    "learning_rate":          2e-4,
    "warmup_ratio":           0.05,
    "lr_scheduler_type":      "cosine",
    "max_seq_length":         2048,
    "save_steps":             50,
    "logging_steps":          10,
    "eval_steps":             50,
    "load_best_model_at_end": True,
    "fp16":                   True,     # Set False if no GPU
    "optim":                  "paged_adamw_8bit",
}

# ── LoRA Config (memory-efficient fine-tuning) ────────────────────────────────
LORA_CONFIG = {
    "r":             16,       # Rank — higher = more capacity, more memory
    "lora_alpha":    32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout":  0.05,
    "bias":          "none",
    "task_type":     "CAUSAL_LM",
}

# ── Cambridge Marking Criteria ────────────────────────────────────────────────
AS_MARKING_BANDS = {
    "12": "Outstanding: Clear, sustained argument. Two+ well-developed analytical points with strong chain of reasoning. Evaluates with a supported final judgment. Uses relevant economic concepts and diagrams correctly.",
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

# ── Examiner Expectations (for feedback engine) ───────────────────────────────
EXAMINER_EXPECTATIONS = {
    "AS_12_mark": {
        "structure": [
            "Introduction with clear thesis (1–2 sentences only)",
            "Point 1: State → Explain → Develop → Example (SEDE chain)",
            "Point 2: State → Explain → Develop → Example (SEDE chain)",
            "Evaluation: Weigh both sides, consider context/assumptions",
            "Conclusion: Supported final judgment (not a summary)",
        ],
        "must_include": [
            "At least one relevant diagram (AD/AS, supply/demand, PPF, etc.)",
            "Economic terminology used correctly",
            "At least one real-world example or data reference",
            "Explicit evaluation words: 'However', 'It depends on', 'In the long run'",
            "A judgment that answers the exact question asked",
        ],
        "common_mistakes": [
            "Listing points without developing them (breadth not depth)",
            "Evaluation that just says 'on the other hand' without a reasoned judgment",
            "Describing a diagram instead of using it to explain",
            "Not answering the specific context of the question",
            "Spending too long on definitions",
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

You always:
- Award marks based on the official Cambridge marking band descriptors
- Explain specifically why marks were awarded or deducted
- Give actionable, examiner-level feedback
- Identify the strongest and weakest parts of each essay
- Suggest exactly how to improve to reach the next mark band"""

GRADING_PROMPT_TEMPLATE = """## Essay to Grade

**Exam Level:** {level}
**Question Type:** {question_type} ({max_marks} marks)
**Question:** {question}

**Student's Essay:**
{essay}

---

## Your Task

Grade this essay as a Cambridge examiner. Provide your response in this exact format:

### MARK AWARDED: [X]/{max_marks}

### MARK BAND: [Band description]

### WHAT THE EXAMINER SEES
[2–3 sentences describing overall impression]

### MARKS BREAKDOWN
**Knowledge & Understanding (AO1):** [X marks] — [reason]
**Analysis (AO2):** [X marks] — [reason]  
**Evaluation (AO3):** [X marks] — [reason]

### STRENGTHS
- [Specific strength 1 with quote or reference from essay]
- [Specific strength 2]

### WHAT LOST MARKS
- [Specific gap 1 — what was missing and why it cost marks]
- [Specific gap 2]

### HOW TO REACH THE NEXT BAND
[Concrete, specific advice — what exact sentences/arguments would push this to a higher mark]

### MODEL ANSWER STRUCTURE
[Brief outline of what a top-band answer to THIS specific question would include]
"""