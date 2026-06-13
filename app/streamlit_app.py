# app/streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit web app — essay submission portal
# Anyone with the link can upload essays (PDF/image/typed) + grades
# All submissions are saved to the shared dataset automatically
#
# Run locally:   streamlit run app/streamlit_app.py
# Share online:  deploy to Streamlit Cloud (free) — see README for steps
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from ocr_pipeline import extract_text_from_bytes, extract_text_from_multiple_files, ocr_quality_report
from content_validator import validate_economics_essay, get_validation_badge
from config import RAW_ESSAYS_DIR, EXAMINER_EXPECTATIONS, AS_MARKING_BANDS, IGCSE_MARKING_BANDS
import sheets_backend as backend

CSV_HEADERS = backend.CSV_HEADERS


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cambridge Economics Grader — Essay Submission",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; }
    .stTextArea textarea { font-size: 15px; line-height: 1.6; }
    .success-box {
        background: #d4edda; border: 1px solid #c3e6cb;
        border-radius: 8px; padding: 16px; margin: 12px 0;
    }
    .warning-box {
        background: #fff3cd; border: 1px solid #ffeeba;
        border-radius: 8px; padding: 12px; margin: 8px 0;
    }
    .info-box {
        background: #d1ecf1; border: 1px solid #bee5eb;
        border-radius: 8px; padding: 12px; margin: 8px 0;
    }
    .mark-band {
        background: #f8f9fa; border-left: 4px solid #007bff;
        padding: 10px 14px; margin: 6px 0; border-radius: 0 6px 6px 0;
    }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_essays():
    try:
        return backend.count_essays()
    except Exception:
        return 0


def save_essay(data: dict):
    success, used_backend = backend.save_essay(data)
    return success, used_backend


def get_mark_band_description(mark: int, level: str) -> str:
    bands = AS_MARKING_BANDS if level == "AS" else IGCSE_MARKING_BANDS
    for band_range, desc in bands.items():
        if "-" in band_range:
            low, high = map(int, band_range.split("-"))
            if low <= mark <= high:
                return desc
        elif mark == int(band_range):
            return desc
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Cambridge_Assessment_International_Education_logo.svg/320px-Cambridge_Assessment_International_Education_logo.svg.png", width=200)

    # Backend status
    status = backend.get_backend_status()
    if status["active_backend"] == "Google Sheets":
        st.success("🟢 Connected to Google Sheets")
    else:
        st.info("🔵 Using local storage (CSV)")
        if status["error"]:
            st.caption(f"⚠ {status['error']}")

    st.markdown("## 📊 Dataset Stats")
    total = count_essays()
    st.metric("Essays submitted", total)
    st.metric("Essays needed for good model", "100+")
    st.progress(min(total / 100, 1.0))

    if total < 20:
        st.warning(f"Need {20 - total} more to start training")
    elif total < 60:
        st.info(f"{60 - total} more for strong accuracy")
    else:
        st.success("Great dataset size!")

    st.markdown("---")
    st.markdown("### 📖 What this is")
    st.markdown(
        "This tool collects real Cambridge Economics essays + examiner marks "
        "to train an AI that grades like a real examiner. "
        "Every essay you submit makes the model smarter."
    )

    st.markdown("---")
    st.markdown("### 🔒 Privacy")
    st.markdown("Essays are stored securely and only used for training the grading model.")

    page = st.radio(
        "Navigate",
        ["📝 Submit Essay", "📚 What Examiners Expect", "📈 Mark Bands", "📊 Dataset Explorer"],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SUBMIT ESSAY
# ─────────────────────────────────────────────────────────────────────────────

if page == "📝 Submit Essay":
    st.title("📝 Submit an Essay")
    st.markdown(
        "Help train the Cambridge Economics AI grader by submitting a real essay "
        "with its mark. The more essays we collect, the more accurately the model grades."
    )

    # Show success banner from a previous submission (after rerun, count is fresh)
    if st.session_state.get("just_submitted"):
        info = st.session_state.pop("just_submitted")
        st.balloons()
        st.markdown(
            f'<div class="success-box">'
            f'<h3>✅ Essay submitted successfully!</h3>'
            f'<p>Saved to {info["backend_label"]}. Dataset now has <strong>{info["total"]} essays</strong>. '
            f'Thank you for contributing!</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.info("You can submit another essay below — the more, the better!")

    st.markdown("---")

    # ── Step 1: Level & Question Type ─────────────────────────────────────────
    st.markdown("### Step 1 — Exam Details")
    col1, col2 = st.columns(2)

    with col1:
        level = st.selectbox("Exam Level", ["AS Level", "IGCSE"])
        level_code = "AS" if "AS" in level else "IGCSE"

    with col2:
        max_marks = 12 if level_code == "AS" else 8
        st.metric("Maximum marks", f"{max_marks} marks")

    topic = st.selectbox(
        "Topic area",
        ["-- Select topic --",
         "Macroeconomics", "Microeconomics", "Market Failure",
         "International Economics", "Development Economics",
         "Price System", "Business Economics", "Other"],
    )

    question = st.text_area(
        "Exam Question",
        placeholder="e.g. Evaluate the effectiveness of fiscal policy in reducing unemployment. [12]",
        height=100,
    )

    # ── Step 2: Essay Input ───────────────────────────────────────────────────
    st.markdown("### Step 2 — Essay")
    st.markdown("Choose how to provide the essay:")

    input_method = st.radio(
        "Input method",
        ["✏️ Type / Paste essay", "📄 Upload PDF", "🖼️ Upload Image (photo of handwritten essay)"],
        horizontal=True,
    )

    essay_text = ""
    ocr_method = None

    if input_method == "✏️ Type / Paste essay":
        essay_text = st.text_area(
            "Essay text",
            placeholder="Paste or type the student's essay here...",
            height=350,
        )

    elif input_method == "📄 Upload PDF":
        uploaded_files = st.file_uploader(
            "Upload PDF(s) — multiple files allowed (e.g. one per page)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            with st.spinner(f"Extracting text from {len(uploaded_files)} file(s)..."):
                files_data = [(f.read(), f.name) for f in uploaded_files]
                essay_text, ocr_method, page_reports = extract_text_from_multiple_files(files_data)
                report = ocr_quality_report(essay_text, ocr_method)

            if report["quality"] == "good":
                st.success(f"✓ Extracted {report['word_count']} words from {len(uploaded_files)} file(s)")
            elif report["quality"] == "fair":
                st.warning(f"⚠ Extracted {report['word_count']} words — please check the text below looks correct.")
            else:
                st.error("Could not extract text well. Try clearer files or type the essay manually.")

            if len(uploaded_files) > 1:
                with st.expander("Per-file extraction details"):
                    for r in page_reports:
                        st.caption(f"📄 {r['filename']}: {r['word_count']} words ({r['method']})")

            for warning in report["warnings"]:
                st.warning(warning)

            if essay_text:
                with st.expander("Preview extracted text (click to expand and verify)"):
                    st.text(essay_text[:2000] + ("..." if len(essay_text) > 2000 else ""))
                    if st.button("Text looks wrong — clear it"):
                        essay_text = ""

    elif input_method == "🖼️ Upload Image (photo of handwritten essay)":
        st.info(
            "💡 Tips for best OCR results: good lighting, flat surface, all text in frame, "
            "no shadows. Upload pages in order if your essay spans multiple pages.\n\n"
            "⚠️ **Note:** Automatic handwriting recognition is imperfect — always check and "
            "correct the extracted text below before submitting."
        )
        uploaded_files = st.file_uploader(
            "Upload image(s) — multiple pages allowed, in order",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            # Show thumbnails of all uploaded pages
            cols = st.columns(min(len(uploaded_files), 4))
            for i, f in enumerate(uploaded_files):
                with cols[i % len(cols)]:
                    st.image(f, caption=f"Page {i+1}: {f.name}", use_column_width=True)

            with st.spinner(f"Running OCR on {len(uploaded_files)} image(s)..."):
                files_data = [(f.read(), f.name) for f in uploaded_files]
                essay_text, ocr_method, page_reports = extract_text_from_multiple_files(files_data)
                report = ocr_quality_report(essay_text, ocr_method)

            if report["quality"] != "poor":
                st.success(f"✓ Extracted {report['word_count']} words from {len(uploaded_files)} page(s)")
            else:
                st.error("Poor extraction quality — please type the essay manually instead.")

            if len(uploaded_files) > 1:
                with st.expander("Per-page extraction details"):
                    for r in page_reports:
                        st.caption(f"🖼️ {r['filename']}: {r['word_count']} words ({r['method']})")

            for warning in report["warnings"]:
                st.warning(warning)

            if essay_text:
                st.markdown("**Extracted text — please review and correct any OCR errors:**")
                essay_text = st.text_area("Edit extracted text", value=essay_text, height=300)

    # Show live word count
    if essay_text:
        wc = len(essay_text.split())
        col_wc, _ = st.columns([1, 3])
        with col_wc:
            color = "green" if wc >= 150 else "orange" if wc >= 80 else "red"
            st.markdown(f"**Word count:** :{color}[{wc} words]")

    # ── Step 3: Mark & Feedback ───────────────────────────────────────────────
    st.markdown("### Step 3 — Mark & Feedback")

    mark = st.slider(
        f"Mark awarded (out of {max_marks})",
        min_value=0,
        max_value=max_marks,
        value=max_marks // 2,
    )

    # Show what that mark means
    band_desc = get_mark_band_description(mark, level_code)
    if band_desc:
        st.markdown(
            f'<div class="mark-band">📋 <strong>Mark band for {mark}/{max_marks}:</strong> {band_desc}</div>',
            unsafe_allow_html=True,
        )

    feedback = st.text_area(
        "Examiner feedback (optional but very valuable)",
        placeholder="Paste any real examiner comments here if you have them...",
        height=120,
    )

    # ── Step 3.5: Economics Content Check ─────────────────────────────────────
    validation_result = None
    override = False

    if essay_text and len(essay_text.strip()) >= 50:
        validation_result = validate_economics_essay(essay_text, question)
        emoji, label = get_validation_badge(validation_result)

        if validation_result["is_valid"]:
            st.success(f"{emoji} {label}")
            if validation_result["keywords_found"]:
                st.caption("Detected: " + ", ".join(validation_result["keywords_found"][:8]))
            override = True
        else:
            st.error(f"{emoji} **{label}**")
            st.markdown(f"_{validation_result['reason']}_")
            for w in validation_result["warnings"]:
                st.warning(w)
            st.markdown(
                "**Submission is blocked.** If you're confident this IS a genuine, "
                "complete economics essay (e.g. unusual phrasing, or OCR missed "
                "some economics terms), tick the box below to override:"
            )
            override = st.checkbox("⚠️ This IS a genuine economics essay — submit anyway")

    # ── Step 4: Submit ────────────────────────────────────────────────────────
    st.markdown("---")

    # Validation
    ready    = True
    messages = []

    if not question.strip():
        messages.append("⚠ Please enter the exam question.")
        ready = False
    if not essay_text or len(essay_text.strip()) < 50:
        messages.append("⚠ Essay is too short or empty.")
        ready = False
    if topic == "-- Select topic --":
        messages.append("⚠ Please select a topic area.")
        ready = False
    if validation_result is not None and not validation_result["is_valid"] and not override:
        messages.append("⚠ This doesn't look like an economics essay. Tick the override box above if you're sure.")
        ready = False

    for msg in messages:
        st.warning(msg)

    if st.button("🚀 Submit Essay to Dataset", disabled=not ready, type="primary", use_container_width=True):
        success, used_backend = save_essay({
            "question":   question.strip(),
            "essay":      essay_text.strip(),
            "mark":       mark,
            "max_marks":  max_marks,
            "level":      level_code,
            "feedback":   feedback.strip(),
            "topic":      topic,
            "date_added": datetime.now().strftime("%Y-%m-%d"),
            "source":     ocr_method or "typed",
        })

        if success:
            new_total = count_essays()
            backend_label = "Google Sheets" if used_backend == "sheets" else "local storage"
            # Store the success info and rerun so the sidebar count refreshes
            # immediately and the banner shows on the fresh page load.
            st.session_state["just_submitted"] = {
                "backend_label": backend_label,
                "total":         new_total,
            }
            st.rerun()
        else:
            st.error("Something went wrong saving your essay. Please try again.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: WHAT EXAMINERS EXPECT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📚 What Examiners Expect":
    st.title("📚 What Cambridge Examiners Expect")

    tab_as, tab_igcse = st.tabs(["AS Level — 12 mark", "IGCSE — 8 mark"])

    for tab, level_key, level_label in [
        (tab_as,    "AS_12_mark",   "AS Level 12-mark"),
        (tab_igcse, "IGCSE_8_mark", "IGCSE 8-mark"),
    ]:
        with tab:
            exp = EXAMINER_EXPECTATIONS.get(level_key, {})

            st.markdown(f"### 📐 Ideal Structure")
            for i, step in enumerate(exp.get("structure", []), 1):
                st.markdown(f"**{i}.** {step}")

            st.markdown("---")
            st.markdown("### ✅ Must Include for Top Marks")
            for item in exp.get("must_include", []):
                st.markdown(f"✅ {item}")

            st.markdown("---")
            st.markdown("### ❌ Common Mistakes That Lose Marks")
            for item in exp.get("common_mistakes", []):
                st.markdown(f"❌ {item}")

            st.markdown("---")
            st.markdown("### 💡 The SEDE Chain")
            st.info(
                "Every analytical point should follow this chain:\n\n"
                "**S**tate → **E**xplain → **D**evelop → **E**xample\n\n"
                "Cambridge examiners reward depth, not breadth. "
                "Two fully developed points beat five shallow ones every time."
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MARK BANDS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Mark Bands":
    st.title("📈 Cambridge Marking Bands")

    tab_as, tab_igcse = st.tabs(["AS Level (12 marks)", "IGCSE (8 marks)"])

    for tab, bands, max_m, level in [
        (tab_as,    AS_MARKING_BANDS,    12, "AS"),
        (tab_igcse, IGCSE_MARKING_BANDS, 8,  "IGCSE"),
    ]:
        with tab:
            colors = {
                0.85: "#28a745",  # green  — top band
                0.65: "#5cb85c",  # light green
                0.45: "#ffc107",  # amber
                0.25: "#fd7e14",  # orange
                0.0:  "#dc3545",  # red
            }

            for band_range, desc in bands.items():
                if "-" in band_range:
                    low, high = map(int, band_range.split("-"))
                    mid = (low + high) / 2
                else:
                    mid = int(band_range)

                ratio = mid / max_m
                color = "#dc3545"
                for threshold, c in sorted(colors.items(), reverse=True):
                    if ratio >= threshold:
                        color = c
                        break

                st.markdown(
                    f'<div style="border-left: 5px solid {color}; padding: 10px 14px; '
                    f'margin: 8px 0; background: #f8f9fa; border-radius: 0 8px 8px 0;">'
                    f'<strong style="color:{color}; font-size:18px">{band_range}/{max_m}</strong>'
                    f'<br><span style="color:#333">{desc}</span></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Overview of all submitted essays.")

    df = backend.load_all_essays()

    if df.empty:
        st.info("No essays submitted yet. Be the first!")
    else:
        df["word_count"] = df["essay"].astype(str).apply(lambda x: len(x.split()))
        df["mark"]      = pd.to_numeric(df["mark"], errors="coerce")
        df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce")
        df = df.dropna(subset=["mark"])

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Essays",   len(df))
        col2.metric("AS Level",       len(df[df.level == "AS"]))
        col3.metric("IGCSE",          len(df[df.level == "IGCSE"]))
        col4.metric("Avg Word Count", int(df["word_count"].mean()))

        st.markdown("---")

        # Mark distribution chart
        st.markdown("### Mark Distribution")
        try:
            import plotly.express as px

            fig = px.histogram(
                df, x="mark", color="level",
                barmode="overlay",
                nbins=13,
                labels={"mark": "Mark", "count": "Essays"},
                color_discrete_map={"AS": "#4e8cff", "IGCSE": "#ff7f50"},
                opacity=0.8,
            )
            fig.update_layout(bargap=0.1, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(df.groupby("mark").size())

        # Topic breakdown
        if "topic" in df.columns:
            st.markdown("### Essays by Topic")
            topic_counts = df["topic"].value_counts()
            st.bar_chart(topic_counts)

        # Recent submissions
        st.markdown("### Recent Submissions")
        display_cols = ["date_added", "level", "mark", "max_marks", "topic", "word_count"]
        available    = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[available].sort_values("date_added", ascending=False).head(20),
            use_container_width=True,
        )