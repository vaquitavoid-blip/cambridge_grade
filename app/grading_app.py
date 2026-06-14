# app/grading_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Professional Streamlit Grading UI
# Three modes: Grade, Edit to Perfection, KAE Analysis
#
# Run: streamlit run app/grading_app.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from config import (
    AS_MARKING_BANDS, IGCSE_MARKING_BANDS,
    EXAMINER_EXPECTATIONS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cambridge Economics Grader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* General */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem; font-weight: 700; }
    h2 { font-size: 1.2rem; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }

    /* Mark badge */
    .mark-badge {
        display: inline-block;
        font-size: 2.4rem;
        font-weight: 800;
        color: white;
        padding: 10px 28px;
        border-radius: 12px;
        margin: 8px 0 16px 0;
    }
    .mark-high   { background: linear-gradient(135deg, #22c55e, #16a34a); }
    .mark-mid    { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .mark-low    { background: linear-gradient(135deg, #ef4444, #dc2626); }

    /* Section cards */
    .feedback-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .feedback-card h4 { margin: 0 0 8px 0; color: #1e293b; font-size: 0.95rem; }

    /* AO bars */
    .ao-row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
    .ao-label { width: 160px; font-size: 0.85rem; color: #475569; }
    .ao-bar-bg { flex: 1; background: #e2e8f0; border-radius: 6px; height: 10px; }
    .ao-bar-fill { height: 10px; border-radius: 6px; }
    .ao-score { font-weight: 700; font-size: 0.9rem; color: #1e293b; width: 50px; text-align: right; }

    /* Model eval box */
    .model-eval {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-left: 5px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* IGCSE content check */
    .accepted   { color: #16a34a; font-weight: 600; }
    .rejected   { color: #dc2626; font-weight: 600; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Sidebar */
    .sidebar-metric { text-align: center; padding: 8px; }
    .sidebar-metric .value { font-size: 2rem; font-weight: 800; color: #3b82f6; }
    .sidebar-metric .label { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI grader model...")
def load_grader():
    from grader import CambridgeGrader
    return CambridgeGrader(use_fine_tuned=True)


def mark_color_class(mark: int, max_marks: int) -> str:
    ratio = mark / max_marks
    if ratio >= 0.75:
        return "mark-high"
    elif ratio >= 0.5:
        return "mark-mid"
    return "mark-low"


def parse_mark(result_text: str, max_marks: int) -> int | None:
    import re
    m = re.search(r"MARK AWARDED[:\s]+(\d+)\s*/\s*\d+", result_text, re.IGNORECASE)
    if m:
        mark = int(m.group(1))
        if 0 <= mark <= max_marks:
            return mark
    return None


def save_result(data: dict):
    results_dir = ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = results_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def render_grading_result(result_text: str, level: str, max_marks: int):
    """Renders the raw grading text into a clean structured UI."""
    import re

    # Extract mark
    mark = parse_mark(result_text, max_marks)

    # ── Mark badge ────────────────────────────────────────────────────────────
    if mark is not None:
        color_class = mark_color_class(mark, max_marks)
        st.markdown(
            f'<div class="mark-badge {color_class}">{mark} / {max_marks}</div>',
            unsafe_allow_html=True,
        )

        # Visual mark band bar
        pct = int(mark / max_marks * 100)
        bar_color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
        st.markdown(
            f'<div style="background:#e2e8f0;border-radius:8px;height:12px;margin-bottom:20px">'
            f'<div style="width:{pct}%;background:{bar_color};height:12px;border-radius:8px;'
            f'transition:width 0.5s"></div></div>',
            unsafe_allow_html=True,
        )

    # ── Render sections as tabs ───────────────────────────────────────────────
    tab_feedback, tab_model, tab_improve = st.tabs([
        "📋 Examiner Feedback",
        "✳ Model Answer / Eval Point",
        "🚀 How to Improve",
    ])

    with tab_feedback:
        _render_section(result_text, "WHAT THE EXAMINER SEES", "Examiner's Impression")
        _render_marks_breakdown(result_text, level, max_marks)

        if level == "AS":
            _render_section(result_text, "EVALUATION QUALITY", "Evaluation Quality")
        else:
            _render_section(result_text, "CONTENT ACCURACY CHECK", "Content Accuracy Check")
            _render_section(result_text, "CLARITY ASSESSMENT", "Clarity Assessment")

        _render_strengths_weaknesses(result_text)

    with tab_model:
        if level == "AS":
            section = _extract_section(result_text, "✳ MODEL EVALUATION POINT")
            if section:
                st.markdown('<div class="model-eval">', unsafe_allow_html=True)
                st.markdown("**✳ What a top-band evaluation looks like for this question:**")
                st.markdown(section)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            section = _extract_section(result_text, "✳ WHAT A FULL-MARK ANSWER LOOKS LIKE")
            if section:
                st.markdown('<div class="model-eval">', unsafe_allow_html=True)
                st.markdown("**✳ What a full-mark IGCSE answer looks like:**")
                st.markdown(section)
                st.markdown('</div>', unsafe_allow_html=True)

        if not section:
            st.info("Model answer section not found in response — try grading again.")

    with tab_improve:
        _render_section(result_text, "HOW TO REACH THE NEXT BAND", "How to Reach the Next Band")
        _render_section(result_text, "WHAT LOST MARKS", "What Lost Marks")

    # ── Raw text expander ─────────────────────────────────────────────────────
    with st.expander("View full raw examiner response"):
        st.text(result_text)


def _extract_section(text: str, heading: str) -> str:
    """Extracts the content under a ### heading."""
    import re
    pattern = rf"###\s*{re.escape(heading)}\s*\n(.*?)(?=\n###|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _render_section(text: str, heading: str, display_title: str):
    content = _extract_section(text, heading)
    if content:
        st.markdown(f"**{display_title}**")
        st.markdown(f'<div class="feedback-card">{content}</div>', unsafe_allow_html=True)


def _render_marks_breakdown(text: str, level: str, max_marks: int):
    """Renders AO marks as visual bars."""
    import re

    content = _extract_section(text, "MARKS BREAKDOWN")
    if not content:
        return

    st.markdown("**Marks Breakdown**")

    # AS AOs
    ao_configs = {
        "AS": [
            ("Knowledge & Understanding (AO1)", 2),
            ("Analysis (AO2)", 6),
            ("Evaluation (AO3)", 4),
        ],
        "IGCSE": [
            ("Content & Knowledge (AO1)", 2),
            ("Development & Explanation (AO2)", 4),
            ("Evaluative Comment (AO3)", 2),
        ],
    }

    aos = ao_configs.get(level, ao_configs["AS"])
    colors = ["#6366f1", "#3b82f6", "#10b981"]

    st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
    for (ao_name, ao_max), color in zip(aos, colors):
        # Try to extract the mark for this AO
        short_name = ao_name.split("(")[0].strip()
        mark_match = re.search(
            rf"{re.escape(ao_name)}[^0-9]*(\d+)[^0-9]*{ao_max}",
            content, re.IGNORECASE
        )
        ao_mark = int(mark_match.group(1)) if mark_match else None
        pct = int(ao_mark / ao_max * 100) if ao_mark is not None else 0
        mark_str = f"{ao_mark}/{ao_max}" if ao_mark is not None else f"?/{ao_max}"

        st.markdown(
            f'<div class="ao-row">'
            f'<div class="ao-label">{short_name}</div>'
            f'<div class="ao-bar-bg"><div class="ao-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<div class="ao-score">{mark_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Also show the raw text
    st.markdown("---")
    st.markdown(content)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_strengths_weaknesses(text: str):
    col1, col2 = st.columns(2)
    with col1:
        strengths = _extract_section(text, "STRENGTHS")
        if strengths:
            st.markdown("**✅ Strengths**")
            st.markdown(f'<div class="feedback-card" style="border-left:4px solid #22c55e">{strengths}</div>',
                        unsafe_allow_html=True)
    with col2:
        gaps = _extract_section(text, "WHAT LOST MARKS")
        if gaps:
            st.markdown("**❌ What Lost Marks**")
            st.markdown(f'<div class="feedback-card" style="border-left:4px solid #ef4444">{gaps}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Cambridge Economics Grader")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["📝 Grade Essay", "✏️ Edit to Perfection", "🔬 KAE Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    level = st.selectbox("Exam Level", ["AS Level (12 marks)", "IGCSE (8 marks)"])
    level_code = "AS" if "AS" in level else "IGCSE"
    max_marks  = 12 if level_code == "AS" else 8

    st.markdown("---")

    # Quick marking band reference
    with st.expander("📈 Mark Band Reference"):
        bands = AS_MARKING_BANDS if level_code == "AS" else IGCSE_MARKING_BANDS
        for band, desc in bands.items():
            ratio = int(band.split("-")[0]) / max_marks if "-" in band else int(band) / max_marks
            color = "#22c55e" if ratio >= 0.75 else "#f59e0b" if ratio >= 0.5 else "#ef4444"
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:6px 10px;margin:4px 0;'
                f'background:#f8fafc;border-radius:0 6px 6px 0;font-size:0.82rem">'
                f'<strong>{band}/{max_marks}</strong> — {desc[:80]}...</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    with st.expander("📚 Examiner Expects"):
        key = "AS_12_mark" if level_code == "AS" else "IGCSE_8_mark"
        exp = EXAMINER_EXPECTATIONS.get(key, {})
        for item in exp.get("must_include", []):
            st.markdown(f"✅ {item}")
        st.markdown("**Common mistakes:**")
        for item in exp.get("common_mistakes", []):
            st.markdown(f"❌ {item}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# ── MODE 1 — GRADE ────────────────────────────────────────────────────────────
if mode == "📝 Grade Essay":
    st.title("📝 Grade Essay")
    st.caption(f"Cambridge {level} · AI Examiner Feedback")

    question = st.text_area(
        "Exam Question",
        placeholder="e.g. Evaluate the effectiveness of fiscal policy in reducing unemployment. [12]",
        height=80,
    )

    essay = st.text_area(
        "Student Essay",
        placeholder="Paste or type the full essay here...",
        height=340,
    )

    if essay:
        wc = len(essay.split())
        color = "green" if wc >= 200 else "orange" if wc >= 100 else "red"
        st.caption(f":{color}[{wc} words]")

    col_btn, col_tip = st.columns([1, 3])
    with col_btn:
        grade_btn = st.button("🎯 Grade Essay", type="primary", use_container_width=True,
                              disabled=not (question.strip() and essay.strip()))
    with col_tip:
        if not question.strip():
            st.caption("⚠️ Enter the exam question to enable grading")
        elif not essay.strip():
            st.caption("⚠️ Enter the essay to enable grading")

    if grade_btn:
        with st.spinner("Grading... this takes 30–60 seconds"):
            try:
                grader = load_grader()
                result = grader.grade(question, essay, level_code, max_marks, verbose=False)
                st.markdown("---")
                render_grading_result(result, level_code, max_marks)
                save_result({"mode": "grade", "level": level_code, "question": question,
                             "essay": essay, "result": result})
            except Exception as e:
                st.error(f"Grading failed: {e}")
                st.info("Make sure the model is trained: `python src/train.py`")


# ── MODE 2 — EDIT ─────────────────────────────────────────────────────────────
elif mode == "✏️ Edit to Perfection":
    st.title("✏️ Edit Essay to Perfection")
    st.caption(f"Cambridge {level} · Rewrites your essay for the highest possible mark")

    if level_code == "IGCSE":
        st.info(
            "**IGCSE mode:** The editor will ensure both points match accepted mark scheme answers, "
            "simplify explanations, and keep evaluation to 1-2 sentences only."
        )
    else:
        st.info(
            "**AS mode:** The editor will strengthen analytical chains, deepen evaluation with "
            "conditions and judgment, and add economic terminology where missing."
        )

    question = st.text_area("Exam Question", height=80,
                             placeholder="e.g. Discuss the likely effects of a rise in interest rates. [12]")
    essay    = st.text_area("Your Essay (to improve)", height=300,
                             placeholder="Paste your essay here — it will be edited for maximum marks...")

    if st.button("✏️ Edit to Perfection", type="primary",
                 disabled=not (question.strip() and essay.strip())):
        with st.spinner("Editing essay... this takes 45–90 seconds"):
            try:
                grader = load_grader()
                result = grader.edit_essay(question, essay, level_code, max_marks, verbose=False)

                tab_edited, tab_changes = st.tabs(["📄 Edited Essay", "🔍 Changes & Why"])

                import re
                edited_section = ""
                changes_section = ""
                predicted_mark = ""

                m = re.search(r"### EDITED ESSAY\n(.*?)(?=\n###|\Z)", result, re.DOTALL)
                if m:
                    edited_section = m.group(1).strip()

                m = re.search(r"### WHAT WAS CHANGED AND WHY\n(.*?)(?=\n###|\Z)", result, re.DOTALL)
                if m:
                    changes_section = m.group(1).strip()

                m = re.search(r"PREDICTED MARK AFTER EDITING[:\s]+(\d+/\d+)", result, re.IGNORECASE)
                if m:
                    predicted_mark = m.group(1)

                with tab_edited:
                    if predicted_mark:
                        st.success(f"🎯 Predicted mark after editing: **{predicted_mark}**")
                    if edited_section:
                        st.markdown(edited_section)
                    else:
                        st.text(result)

                with tab_changes:
                    if changes_section:
                        st.markdown(changes_section)
                    else:
                        st.text(result)

                save_result({"mode": "edit", "level": level_code, "question": question,
                             "essay": essay, "result": result})
            except Exception as e:
                st.error(f"Edit failed: {e}")


# ── MODE 3 — KAE ──────────────────────────────────────────────────────────────
elif mode == "🔬 KAE Analysis":
    st.title("🔬 KAE Point-by-Point Analysis")
    st.caption(f"Cambridge {level} · Enter your points — see exactly where marks are lost")

    ao = {"AS": {"ao1": 2, "ao2": 6, "ao3": 4}, "IGCSE": {"ao1": 2, "ao2": 4, "ao3": 2}}[level_code]

    st.markdown(
        f"Enter your planned points for each Assessment Objective. "
        f"**AO split for {level_code}:** "
        f"Knowledge={ao['ao1']} marks · "
        f"Analysis={ao['ao2']} marks · "
        f"Evaluation={ao['ao3']} marks"
    )

    question = st.text_area("Exam Question", height=75,
                             placeholder="Paste the exam question here...")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**📖 Knowledge (AO1) — {ao['ao1']} marks**")
        st.caption("Definitions, concepts, what you know about the topic")
        knowledge = st.text_area("Knowledge points", height=200, label_visibility="collapsed",
                                  placeholder="e.g.\n- Fiscal policy = government use of taxation and spending\n- Aggregate demand = C + I + G + (X-M)")

    with col2:
        st.markdown(f"**🔗 Analysis (AO2) — {ao['ao2']} marks**")
        st.caption("How/why — chains of reasoning and economic mechanisms")
        analysis = st.text_area("Analysis points", height=200, label_visibility="collapsed",
                                 placeholder="e.g.\n- Increase in G → AD shifts right → real GDP rises\n- Multiplier effect amplifies the initial injection")

    with col3:
        st.markdown(f"**⚖️ Evaluation (AO3) — {ao['ao3']} marks**")
        if level_code == "IGCSE":
            st.caption("Brief comment only needed — 1-2 sentences max for IGCSE")
        else:
            st.caption("Judgments, conditions, 'it depends on...', final answer")
        evaluation = st.text_area("Evaluation points", height=200, label_visibility="collapsed",
                                   placeholder="e.g.\n- Depends on the size of multiplier\n- Less effective if economy is at full employment\n- Short run vs long run distinction")

    if st.button("🔬 Analyse My Points", type="primary",
                 disabled=not question.strip()):
        with st.spinner("Analysing your points... 30–60 seconds"):
            try:
                grader = load_grader()
                result = grader.kae_analysis(
                    question, level_code, max_marks,
                    knowledge, analysis, evaluation,
                    verbose=False,
                )

                st.markdown("---")

                # Render AO sections in columns
                import re

                def get_section(heading):
                    m = re.search(rf"###\s*{re.escape(heading)}\s*\n(.*?)(?=\n###|\Z)",
                                  result, re.DOTALL | re.IGNORECASE)
                    return m.group(1).strip() if m else ""

                col_k, col_a, col_e = st.columns(3)
                ao_names = {
                    "AS":    ["KNOWLEDGE (AO1) ASSESSMENT", "ANALYSIS (AO2) ASSESSMENT", "EVALUATION (AO3) ASSESSMENT"],
                    "IGCSE": ["KNOWLEDGE (AO1) ASSESSMENT", "ANALYSIS (AO2) ASSESSMENT", "EVALUATION (AO3) ASSESSMENT"],
                }

                colors = ["#6366f1", "#3b82f6", "#10b981"]
                for col, heading, color in zip([col_k, col_a, col_e], ao_names[level_code], colors):
                    with col:
                        content = get_section(heading)
                        short = heading.split("(")[0].strip().title()
                        st.markdown(
                            f'<div style="border-left:4px solid {color};padding:10px 14px;'
                            f'background:#f8fafc;border-radius:0 8px 8px 0">'
                            f'<strong>{short}</strong><br><small>{content or "Not found"}</small></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # Overall mark
                overall = get_section("OVERALL PREDICTED MARK")
                if overall:
                    st.success(f"🎯 **Predicted Mark: {overall}**")

                # Priority fixes
                fixes = get_section("PRIORITY FIXES")
                if fixes:
                    st.markdown("### 🛠️ Priority Fixes")
                    st.markdown(fixes)

                # Complete answer guide
                complete = get_section("WHAT A COMPLETE ANSWER LOOKS LIKE")
                if complete:
                    st.markdown("### ✳ What a Complete Answer Looks Like")
                    st.markdown(
                        f'<div class="model-eval">{complete}</div>',
                        unsafe_allow_html=True,
                    )

                with st.expander("View full raw response"):
                    st.text(result)

                save_result({"mode": "kae", "level": level_code, "question": question,
                             "knowledge": knowledge, "analysis": analysis,
                             "evaluation": evaluation, "result": result})

            except Exception as e:
                st.error(f"Analysis failed: {e}")