# src/content_validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Content Validator — checks that submitted text is actually an economics
# essay before allowing it into the dataset. Blocks spam, random text,
# essays from other subjects, etc.
# ─────────────────────────────────────────────────────────────────────────────

import re

# ─────────────────────────────────────────────────────────────────────────────
# ECONOMICS VOCABULARY
# A broad list covering AS Level and IGCSE Economics syllabus terms.
# Used to score how "economics-like" a piece of text is.
# ─────────────────────────────────────────────────────────────────────────────

ECONOMICS_KEYWORDS = [
    # Core micro
    "demand", "supply", "equilibrium", "price mechanism", "elasticity",
    "elastic", "inelastic", "consumer surplus", "producer surplus",
    "opportunity cost", "scarcity", "factors of production", "marginal",
    "marginal cost", "marginal revenue", "marginal utility",
    "market failure", "externality", "externalities", "public good",
    "merit good", "demerit good", "subsidy", "subsidies", "tax", "taxation",
    "indirect tax", "price ceiling", "price floor", "maximum price",
    "minimum price", "monopoly", "oligopoly", "perfect competition",
    "monopolistic competition", "competitive market", "market structure",
    "economies of scale", "diseconomies of scale", "barriers to entry",
    "profit maximisation", "profit maximization", "allocative efficiency",
    "productive efficiency", "x-inefficiency", "asymmetric information",

    # Core macro
    "gdp", "gross domestic product", "inflation", "deflation",
    "unemployment", "economic growth", "recession", "fiscal policy",
    "monetary policy", "interest rate", "interest rates", "exchange rate",
    "central bank", "aggregate demand", "aggregate supply",
    "government spending", "budget deficit", "budget surplus",
    "national debt", "money supply", "quantitative easing",
    "multiplier effect", "accelerator", "phillips curve",
    "supply-side policy", "supply side policy", "income tax",
    "balance of payments", "current account", "trade deficit",
    "trade surplus", "exports", "imports", "tariff", "tariffs", "quota",
    "free trade", "protectionism", "comparative advantage",
    "exchange rates", "devaluation", "appreciation", "depreciation",

    # Development / other
    "developing economy", "developing country", "developed economy",
    "income inequality", "poverty", "standard of living", "human development",
    "globalisation", "globalization", "multinational corporation",
    "labour market", "wage", "wages", "minimum wage", "trade union",

    # General economic actors / framing
    "consumers", "producers", "households", "firms", "government intervention",
    "market economy", "command economy", "mixed economy", "private sector",
    "public sector", "economic agents", "resource allocation",
]

# Keywords that strongly suggest this is NOT an economics essay
OFF_TOPIC_SIGNALS = {
    "psychology": ["cognitive dissonance", "classical conditioning", "neurotransmitter",
                   "operant conditioning", "psychoanalysis", "freud", "piaget"],
    "english_lit": ["protagonist", "antagonist", "metaphor", "iambic pentameter",
                    "stanza", "soliloquy", "characterisation", "narrative voice"],
    "biology":     ["mitochondria", "photosynthesis", "chromosome", "enzyme",
                    "cell membrane", "dna replication", "osmosis"],
    "history":     ["treaty of", "world war", "revolution of", "dictatorship",
                    "colonialism", "annexation"],
    "chemistry":   ["covalent bond", "molar mass", "exothermic", "ionic bond",
                    "periodic table", "chemical equation"],
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_economics_essay(text: str, question: str = "") -> dict:
    """
    Checks whether the given text looks like a genuine economics essay.

    Returns a dict:
    {
        "is_valid":     bool,
        "confidence":   "high" | "medium" | "low",
        "score":        float (0-1, proportion of econ signal),
        "keywords_found": list[str],
        "warnings":     list[str],
        "reason":       str  (human-readable summary)
    }

    This is a heuristic check, not perfect — designed to catch obvious
    spam/off-topic submissions, not to be a strict gatekeeper. The UI
    should warn the user and let them confirm/override if they believe
    it's a genuine economics essay that just uses unusual phrasing.
    """
    combined_text = f"{question}\n{text}".lower()
    warnings = []

    # ── Check 1: minimum length ────────────────────────────────────────────
    word_count = len(text.split())
    if word_count < 50:
        return {
            "is_valid":       False,
            "confidence":     "high",
            "score":          0.0,
            "keywords_found": [],
            "warnings":       ["Text is too short to be a real essay (under 50 words)."],
            "reason":         "Too short — likely not a complete essay.",
        }

    # ── Check 2: gibberish detection ───────────────────────────────────────
    # Real essays have a reasonable ratio of common English words
    common_words = {"the", "a", "an", "is", "are", "of", "to", "in", "and",
                     "this", "that", "for", "on", "with", "as", "it", "be",
                     "by", "or", "from", "which", "will", "can", "would"}
    words = re.findall(r"[a-z']+", combined_text)
    if words:
        common_ratio = sum(1 for w in words if w in common_words) / len(words)
        if common_ratio < 0.08:
            warnings.append(
                "Text doesn't read like normal English prose — check the OCR extraction is correct."
            )

    # ── Check 3: economics keyword density ─────────────────────────────────
    keywords_found = [kw for kw in ECONOMICS_KEYWORDS if kw in combined_text]
    econ_score = min(len(keywords_found) / 5, 1.0)  # 5+ keywords = full score

    # ── Check 4: off-topic subject detection ───────────────────────────────
    off_topic_matches = {}
    for subject, signals in OFF_TOPIC_SIGNALS.items():
        matches = [s for s in signals if s in combined_text]
        if matches:
            off_topic_matches[subject] = matches

    # ── Decision logic ───────────────────────────────────────────────────────
    if off_topic_matches and not keywords_found:
        subjects = ", ".join(off_topic_matches.keys())
        return {
            "is_valid":       False,
            "confidence":     "high",
            "score":          0.0,
            "keywords_found": [],
            "warnings":       [f"This text appears to be about {subjects}, not economics."],
            "reason":         f"Detected {subjects} content with no economics terminology.",
        }

    if len(keywords_found) == 0:
        return {
            "is_valid":       False,
            "confidence":     "medium",
            "score":          0.0,
            "keywords_found": [],
            "warnings":       ["No economics terminology detected in the question or essay."],
            "reason":         "Could not find any recognisable economics vocabulary.",
        }

    if len(keywords_found) < 3:
        return {
            "is_valid":       False,  # blocked — requires explicit override
            "confidence":     "low",
            "score":          econ_score,
            "keywords_found": keywords_found,
            "warnings":       [
                f"Only found {len(keywords_found)} economics term(s)"
                + (f": {', '.join(keywords_found)}" if keywords_found else "")
                + ". This doesn't look like a substantial economics essay."
            ],
            "reason":         "Low economics vocabulary density — likely not a genuine economics essay.",
        }

    # 3+ keywords found — confident this is economics content
    return {
        "is_valid":       True,
        "confidence":     "high",
        "score":          econ_score,
        "keywords_found": keywords_found,
        "warnings":       warnings,
        "reason":         f"Found {len(keywords_found)} economics terms — looks like a genuine economics essay.",
    }


def get_validation_badge(result: dict) -> tuple[str, str]:
    """
    Returns (emoji, label) for displaying validation status in the UI.
    """
    if not result["is_valid"]:
        return "🚫", "Not recognised as an economics essay"
    if result["confidence"] == "high":
        return "✅", "Looks like a genuine economics essay"
    if result["confidence"] == "medium":
        return "⚠️", "Possibly economics — please verify"
    return "⚠️", "Low confidence — please verify"