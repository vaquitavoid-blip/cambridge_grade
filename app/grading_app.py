# app/grading_app.py
# Cambridge Economics AI Examiner & Learning Platform
# Complete unified dashboard — all 10 phases

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from config import AS_MARKING_BANDS, IGCSE_MARKING_BANDS, EXAMINER_EXPECTATIONS, AO_MARKS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cambridge Economics AI Examiner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem}
h1{font-size:1.7rem;font-weight:800}
h2{font-size:1.15rem;font-weight:700;border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin-top:1.2rem}
.mark-badge{display:inline-block;font-size:2.5rem;font-weight:900;color:white;
            padding:10px 30px;border-radius:14px;margin:8px 0 14px 0;letter-spacing:1px}
.mark-high{background:linear-gradient(135deg,#22c55e,#15803d)}
.mark-mid{background:linear-gradient(135deg,#f59e0b,#b45309)}
.mark-low{background:linear-gradient(135deg,#ef4444,#b91c1c)}
.card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
      padding:14px 18px;margin:8px 0}
.card-green{border-left:4px solid #22c55e}
.card-red{border-left:4px solid #ef4444}
.card-blue{border-left:4px solid #3b82f6}
.card-yellow{border-left:4px solid #f59e0b}
.ao-row{display:flex;align-items:center;gap:10px;margin:5px 0}
.ao-label{width:200px;font-size:.83rem;color:#475569}
.ao-bar{flex:1;background:#e2e8f0;border-radius:6px;height:10px}
.ao-fill{height:10px;border-radius:6px}
.ao-score{font-weight:700;font-size:.88rem;width:55px;text-align:right}
.rag-badge{background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;
           padding:4px 10px;font-size:.8rem;color:#1d4ed8;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_grader():
    from grader import CambridgeGrader
    return CambridgeGrader(use_fine_tuned=True)


@st.cache_resource(show_spinner=False)
def load_rag():
    try:
        from rag_engine import get_rag_status, retrieve_context, add_document, list_documents, delete_document, is_rag_available
        return {
            "available": is_rag_available(),
            "get_status": get_rag_status,
            "retrieve": retrieve_context,
            "add": add_document,
            "list": list_documents,
            "delete": delete_document,
        }
    except Exception:
        return {"available": False}


# ── Helpers ───────────────────────────────────────────────────────────────────
def mark_class(mark, max_marks):
    r = mark / max_marks if max_marks else 0
    return "mark-high" if r >= 0.75 else "mark-mid" if r >= 0.5 else "mark-low"


def ao_bar(label, score, max_score, color):
    pct = int(score / max_score * 100) if max_score and score is not None else 0
    score_str = f"{score}/{max_score}" if score is not None else f"?/{max_score}"
    st.markdown(
        f'<div class="ao-row">'
        f'<div class="ao-label">{label}</div>'
        f'<div class="ao-bar"><div class="ao-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<div class="ao-score">{score_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def save_result(data: dict):
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def render_grade_result(r: dict, level: str, max_marks: int):
    """Renders a structured grade result dict into the UI."""
    if r.get("mark") is None and not r.get("impression") and not r.get("raw"):
        st.error("Model returned empty output. See raw response below.")
        st.code(r.get("raw", "(nothing)"))
        return

    # ── Mark badge ────────────────────────────────────────────────────────────
    if r.get("mark") is not None:
        cls = mark_class(r["mark"], max_marks)
        st.markdown(f'<div class="mark-badge {cls}">{r["mark"]} / {max_marks}</div>', unsafe_allow_html=True)
        pct = int(r["mark"] / max_marks * 100)
        bar_color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
        st.markdown(
            f'<div style="background:#e2e8f0;border-radius:8px;height:12px;margin-bottom:1rem">'
            f'<div style="width:{pct}%;background:{bar_color};height:12px;border-radius:8px"></div></div>',
            unsafe_allow_html=True,
        )
        if r.get("confidence"):
            conf_color = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}.get(r["confidence"], "#64748b")
            st.markdown(
                f'<span style="background:{conf_color}20;border:1px solid {conf_color}60;'
                f'border-radius:6px;padding:3px 10px;font-size:.8rem;color:{conf_color}">'
                f'Confidence: {r["confidence"]}</span>',
                unsafe_allow_html=True,
            )
        if r.get("band"):
            st.caption(f"**Band:** {r['band']}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["📋 Feedback", "📊 AO Breakdown", "✳ Model Answer", "🚀 Improve"])

    with t1:
        if r.get("impression"):
            st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
            st.markdown("**Examiner's Impression**")
            st.write(r["impression"])
            st.markdown('</div>', unsafe_allow_html=True)

        if level == "IGCSE":
            if r.get("point_1_status") or r.get("point_2_status"):
                st.markdown("**Content Accuracy Check**")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                for label, val in [("Point 1", r.get("point_1_status","")), ("Point 2", r.get("point_2_status",""))]:
                    if val:
                        icon = "✅" if "ACCEPTED" in val.upper() and "NOT" not in val.upper() else "❌"
                        st.markdown(f"{icon} **{label}:** {val}")
                st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if r.get("strength_1") or r.get("strength_2"):
                st.markdown("**✅ Strengths**")
                st.markdown('<div class="card card-green">', unsafe_allow_html=True)
                if r.get("strength_1"): st.markdown(f"• {r['strength_1']}")
                if r.get("strength_2"): st.markdown(f"• {r['strength_2']}")
                st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            if r.get("gap_1") or r.get("gap_2"):
                st.markdown("**❌ What Lost Marks**")
                st.markdown('<div class="card card-red">', unsafe_allow_html=True)
                if r.get("gap_1"): st.markdown(f"• {r['gap_1']}")
                if r.get("gap_2"): st.markdown(f"• {r['gap_2']}")
                st.markdown('</div>', unsafe_allow_html=True)

        if r.get("eval_quality") and level == "AS":
            st.markdown("**Evaluation Quality**")
            st.markdown(f'<div class="card card-yellow">{r["eval_quality"]}</div>', unsafe_allow_html=True)

    with t2:
        st.markdown("**Assessment Objective Breakdown**")
        ao_info = AO_MARKS[level]
        ao_bar(ao_info["AO1"]["name"] + " (AO1)", r.get("ao1_mark"), r.get("ao1_max", 2), "#6366f1")
        if r.get("ao1_reason"):
            st.caption(f"↳ {r['ao1_reason']}")
        ao_bar(ao_info["AO2"]["name"] + " (AO2)", r.get("ao2_mark"), r.get("ao2_max", 6), "#3b82f6")
        if r.get("ao2_reason"):
            st.caption(f"↳ {r['ao2_reason']}")
        if level == "AS":
            ao_bar(ao_info["AO3"]["name"] + " (AO3)", r.get("ao3_mark"), r.get("ao3_max", 4), "#10b981")
            if r.get("ao3_reason"):
                st.caption(f"↳ {r['ao3_reason']}")

    with t3:
        key = "model_eval" if level == "AS" else "model_answer"
        content = r.get(key, "")
        if content:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);'
                f'border-left:5px solid #3b82f6;border-radius:0 10px 10px 0;padding:16px 20px">'
                f'<strong>✳ {"Top-band evaluation for this question" if level=="AS" else "Full-mark model answer"}</strong><br><br>'
                f'{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Model answer not generated. Try grading again.")

    with t4:
        if r.get("next_band"):
            st.markdown("**How to Reach the Next Band**")
            st.markdown(f'<div class="card card-blue">{r["next_band"]}</div>', unsafe_allow_html=True)

    with st.expander("🔍 Debug: Raw model output"):
        st.text(r.get("raw", "(empty)"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Cambridge Economics AI")
    st.markdown("---")

    page = st.radio("", [
        "📝 Grade Essay",
        "✏️ Edit to Perfection",
        "🔬 KAE Planner",
        "📄 Generate Essay",
        "📚 Knowledge Base",
        "🤖 Examiner Assistant",
        "📈 Analytics",
    ], label_visibility="collapsed")

    st.markdown("---")
    level     = st.selectbox("Exam Level", ["AS Level (12 marks)", "IGCSE (8 marks)"])
    level_code = "AS" if "AS" in level else "IGCSE"
    max_marks  = 12 if level_code == "AS" else 8

    # Student ID for analytics
    student_id = st.text_input("Student ID (optional)", placeholder="e.g. saikrish_001",
                                help="Enter any ID to track your progress over time")

    st.markdown("---")
    rag = load_rag()
    if rag["available"]:
        status = rag["get_status"]()
        st.markdown(
            f'<span class="rag-badge">📚 {status["doc_count"]} docs · {status["chunk_count"]} chunks in KB</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("💡 Install chromadb + sentence-transformers to enable the knowledge base")

    st.markdown("---")
    with st.expander("📈 Mark Bands"):
        bands = AS_MARKING_BANDS if level_code == "AS" else IGCSE_MARKING_BANDS
        for band, desc in bands.items():
            try:
                low = int(band.split("-")[0]) if "-" in band else int(band)
            except:
                low = 0
            r = low / max_marks
            c = "#22c55e" if r >= 0.75 else "#f59e0b" if r >= 0.5 else "#ef4444"
            st.markdown(
                f'<div style="border-left:3px solid {c};padding:4px 8px;margin:3px 0;'
                f'background:#f8fafc;border-radius:0 5px 5px 0;font-size:.78rem">'
                f'<b>{band}/{max_marks}</b> — {desc[:70]}...</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GRADE ESSAY
# ─────────────────────────────────────────────────────────────────────────────
if page == "📝 Grade Essay":
    st.title("📝 Grade Essay")
    st.caption(f"Cambridge {level} · AI Examiner Feedback with AO Breakdown")

    question = st.text_area("Exam Question", height=75,
        placeholder="e.g. Evaluate the effectiveness of fiscal policy in reducing unemployment. [12]")
    essay    = st.text_area("Student Essay", height=320,
        placeholder="Paste the full essay here...")

    topic = st.selectbox("Topic (for analytics)", ["--", "Macroeconomics", "Microeconomics",
        "Market Failure", "International Economics", "Price System", "Development", "Other"])

    if essay:
        wc = len(essay.split())
        c  = "green" if wc >= 200 else "orange" if wc >= 100 else "red"
        st.caption(f":{c}[{wc} words]")

    # RAG context retrieval
    rag_context = ""
    if rag["available"] and question.strip():
        try:
            rag_context = rag["retrieve"](f"{question} {level_code} mark scheme", n_results=3)
            if rag_context:
                st.markdown('<span class="rag-badge">📚 Relevant mark scheme context retrieved</span>',
                            unsafe_allow_html=True)
        except Exception:
            pass

    if st.button("🎯 Grade Essay", type="primary",
                 disabled=not (question.strip() and essay.strip())):
        with st.spinner("Grading... 30–90 seconds"):
            try:
                g = load_grader()
                r = g.grade(question, essay, level_code, max_marks,
                            rag_context=rag_context, verbose=False)
                st.markdown("---")
                render_grade_result(r, level_code, max_marks)

                # Save to analytics
                if student_id.strip() and r.get("mark") is not None:
                    try:
                        from analytics import record_session
                        record_session(
                            student_id=student_id.strip(),
                            question=question, level=level_code, max_marks=max_marks,
                            mark=r["mark"], ao1_mark=r.get("ao1_mark"),
                            ao2_mark=r.get("ao2_mark"), ao3_mark=r.get("ao3_mark"),
                            topic=topic if topic != "--" else "",
                        )
                        st.caption(f"✓ Session recorded for {student_id}")
                    except Exception as e:
                        st.caption(f"Analytics error: {e}")

                save_result({"mode": "grade", "level": level_code, "question": question,
                             "essay": essay, "result": r})
            except Exception as e:
                st.error(f"Grading failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDIT TO PERFECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "✏️ Edit to Perfection":
    st.title("✏️ Edit Essay to Perfection")
    st.caption(f"Cambridge {level} · Rewrites essay for highest mark")

    if level_code == "IGCSE":
        st.info("**IGCSE:** Editor ensures both points match Cambridge MS accepted answers and keeps evaluation to 1-2 sentences.")
    else:
        st.info("**AS Level:** Editor strengthens analytical chains, deepens evaluation with conditions and judgment.")

    question = st.text_area("Exam Question", height=75)
    essay    = st.text_area("Your Essay", height=300, placeholder="Paste essay here...")

    if st.button("✏️ Edit to Perfection", type="primary",
                 disabled=not (question.strip() and essay.strip())):
        with st.spinner("Editing... 45–90 seconds"):
            try:
                g = load_grader()
                r = g.edit_essay(question, essay, level_code, max_marks, verbose=False)

                if r.get("predicted_mark"):
                    st.success(f"🎯 Predicted mark after editing: **{r['predicted_mark']}**")

                tab1, tab2 = st.tabs(["📄 Edited Essay", "🔍 Changes Made"])
                with tab1:
                    if r.get("edited_essay"):
                        st.markdown(r["edited_essay"])
                    else:
                        st.text(r["raw"])
                with tab2:
                    if r.get("changes"):
                        for change in r["changes"]:
                            st.markdown(f"• {change}")
                    else:
                        st.text(r["raw"])

                with st.expander("Raw output"):
                    st.text(r["raw"])

                save_result({"mode": "edit", "level": level_code, "question": question,
                             "essay": essay, "result": r})
            except Exception as e:
                st.error(f"Edit failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KAE PLANNER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 KAE Planner":
    st.title("🔬 Knowledge / Analysis / Evaluation Planner")
    st.caption(f"Cambridge {level} · Enter your points, predict your mark before writing")

    ao = AO_MARKS[level_code]
    st.info(
        f"**AO split:** AO1 ({ao['AO1']['name']}) = {ao['AO1']['max']} marks · "
        f"AO2 ({ao['AO2']['name']}) = {ao['AO2']['max']} marks"
        + (f" · AO3 ({ao['AO3']['name']}) = {ao['AO3']['max']} marks" if level_code == "AS" else "")
    )

    question = st.text_area("Exam Question", height=75)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"**📖 Knowledge (AO1) — {ao['AO1']['max']} marks**")
        st.caption("Definitions, facts, concepts")
        knowledge = st.text_area("K points", height=200, label_visibility="collapsed",
            placeholder="e.g.\n- AD = C + I + G + (X-M)\n- Fiscal policy = gov tax & spending")

    with c2:
        st.markdown(f"**🔗 Analysis (AO2) — {ao['AO2']['max']} marks**")
        st.caption("Chains of reasoning — how/why")
        analysis = st.text_area("A points", height=200, label_visibility="collapsed",
            placeholder="e.g.\n- Higher G → AD shifts right → GDP rises\n- Multiplier amplifies the effect")

    with c3:
        st.markdown(f"**⚖️ Evaluation (AO3){' — ' + str(ao['AO3']['max']) + ' marks' if level_code == 'AS' else ' (brief)'}**")
        st.caption("Judgments, conditions, limitations" if level_code == "AS" else "1-2 sentence comment only")
        evaluation = st.text_area("E points", height=200, label_visibility="collapsed",
            placeholder="e.g.\n- Depends on size of multiplier\n- Less effective at full employment")

    if st.button("🔬 Analyse My Points", type="primary", disabled=not question.strip()):
        with st.spinner("Analysing... 30–60 seconds"):
            try:
                g = load_grader()
                r = g.kae_analysis(question, level_code, max_marks,
                                   knowledge, analysis, evaluation, verbose=False)

                # Show predicted mark
                if r.get("total_predicted"):
                    st.success(f"🎯 Predicted mark: **{r['total_predicted']}**")

                # AO cards
                ca, cb, cc = st.columns(3)
                colors = ["#6366f1", "#3b82f6", "#10b981"]
                for col, ao_key, color in zip([ca, cb, cc], ["AO1", "AO2", "AO3"], colors):
                    with col:
                        score = r.get(f"{ao_key.lower()}_score")
                        max_v = r.get(f"{ao_key.lower()}_max")
                        fb    = r.get(f"{ao_key.lower()}_feedback", "")
                        pct   = int(score / max_v * 100) if score and max_v else 0
                        st.markdown(
                            f'<div style="border-left:4px solid {color};padding:10px;background:#f8fafc;border-radius:0 8px 8px 0">'
                            f'<b>{ao_key}</b>: {score}/{max_v}<br>'
                            f'<div style="background:#e2e8f0;border-radius:4px;height:8px;margin:4px 0">'
                            f'<div style="width:{pct}%;background:{color};height:8px;border-radius:4px"></div></div>'
                            f'<small>{fb[:120]}</small></div>',
                            unsafe_allow_html=True,
                        )

                # Priority fixes
                if r.get("fix_1") or r.get("fix_2") or r.get("fix_3"):
                    st.markdown("### 🛠️ Priority Fixes")
                    for i, fix in enumerate([r.get("fix_1"), r.get("fix_2"), r.get("fix_3")], 1):
                        if fix:
                            st.markdown(f"**{i}.** {fix}")

                # Essay plan
                if r.get("essay_plan"):
                    st.markdown("### ✳ Top-Band Essay Plan")
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);'
                        f'border-left:5px solid #3b82f6;border-radius:0 10px 10px 0;padding:16px">'
                        f'{r["essay_plan"]}</div>',
                        unsafe_allow_html=True,
                    )

                with st.expander("Raw output"):
                    st.text(r["raw"])

                save_result({"mode": "kae", "level": level_code, "question": question,
                             "k": knowledge, "a": analysis, "e": evaluation, "result": r})
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GENERATE ESSAY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📄 Generate Essay":
    st.title("📄 Generate Full-Mark Essay")
    st.caption(f"Cambridge {level} · AI generates a model answer")

    question = st.text_area("Exam Question", height=75,
        placeholder="Enter the question to generate a full-mark answer for...")

    gen_mode = st.radio("Generate from:", ["Question only", "Bullet-point notes", "KAE plan"],
                         horizontal=True)
    extra = ""
    if gen_mode == "Bullet-point notes":
        extra = st.text_area("Your notes/bullet points", height=150,
            placeholder="- Fiscal policy involves government spending and taxation\n- AD can be boosted by cutting taxes...")
    elif gen_mode == "KAE plan":
        extra = st.text_area("Your KAE plan", height=150,
            placeholder="K: Definition of fiscal policy, AD equation\nA: G increases → AD shifts right → multiplier effect\nE: Depends on size of multiplier, crowding out")

    if st.button("📄 Generate Essay", type="primary", disabled=not question.strip()):
        with st.spinner("Generating full-mark essay... 45–90 seconds"):
            try:
                g = load_grader()
                r = g.generate_essay(
                    question=question, level=level_code, max_marks=max_marks,
                    from_notes=extra if gen_mode == "Bullet-point notes" else "",
                    from_kae=extra if gen_mode == "KAE plan" else "",
                    verbose=False,
                )

                if r.get("essay"):
                    st.markdown("### 📄 Generated Essay")
                    st.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                        f'border-radius:10px;padding:20px;line-height:1.7">'
                        f'{r["essay"].replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True,
                    )
                    if r.get("examiner_notes"):
                        st.markdown("### 🎓 Examiner Notes")
                        st.info(r["examiner_notes"])
                else:
                    st.text(r["raw"])

                with st.expander("Raw output"):
                    st.text(r["raw"])

                save_result({"mode": "generate", "level": level_code, "question": question,
                             "result": r})
            except Exception as e:
                st.error(f"Generation failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    st.caption("Upload mark schemes, examiner reports, textbooks — used during grading")

    if not rag["available"]:
        st.error("RAG not available. Install: `pip install chromadb sentence-transformers`")
        st.code("pip install chromadb sentence-transformers")
        st.stop()

    status = rag["get_status"]()
    col1, col2 = st.columns(2)
    col1.metric("Documents", status["doc_count"])
    col2.metric("Indexed chunks", status["chunk_count"])

    st.markdown("---")
    st.markdown("### Upload Document")

    uploaded = st.file_uploader(
        "Upload PDF, mark scheme, examiner report, or notes",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    doc_type = st.selectbox("Document type", ["mark_scheme", "examiner_report", "textbook", "notes", "model_answer", "general"])
    topic    = st.text_input("Topic tag", placeholder="e.g. fiscal policy, market failure")

    if uploaded and st.button("📥 Add to Knowledge Base", type="primary"):
        for f in uploaded:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    result = rag["add"](f.read(), f.name, doc_type, topic)
                    if result.get("success"):
                        if result.get("skipped"):
                            st.info(f"⏭️ {f.name} already in knowledge base")
                        else:
                            st.success(f"✓ {f.name} — {result['chunks']} chunks, {result['word_count']} words")
                    else:
                        st.error(f"✗ {f.name}: {result.get('error')}")
                except Exception as e:
                    st.error(f"✗ {f.name}: {e}")

    # List docs
    st.markdown("---")
    st.markdown("### Indexed Documents")
    docs = rag["list"]()
    if docs:
        for doc in docs:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"📄 {doc['filename']} ({doc['doc_type']})")
            col2.write(f"{doc['chunks']} chunks")
            if col3.button("🗑️", key=doc["hash"]):
                rag["delete"](doc["hash"])
                st.rerun()
    else:
        st.info("No documents yet. Upload mark schemes or examiner reports above.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXAMINER ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Examiner Assistant":
    st.title("🤖 Cambridge Examiner Assistant")
    st.caption("Ask questions about marking, topics, examiner reports")

    examples = [
        "Why do students lose AO3 marks in evaluation questions?",
        "What do examiner reports say about fiscal policy essays?",
        "What are common mistakes in market failure 12-mark questions?",
        "How should I structure an IGCSE 8-mark response?",
        "What counts as a developed point for AO2?",
    ]

    selected = st.selectbox("Example questions:", ["-- Type your own below --"] + examples)
    question = st.text_area(
        "Your question:",
        value=selected if selected != "-- Type your own below --" else "",
        height=100,
    )

    if st.button("🤖 Ask Examiner", type="primary", disabled=not question.strip()):
        with st.spinner("Consulting knowledge base..."):
            try:
                context = ""
                if rag["available"]:
                    context = rag["retrieve"](question, n_results=4)

                g      = load_grader()
                answer = g.examiner_assistant(question, context, verbose=False)

                st.markdown("---")
                if context:
                    st.markdown('<span class="rag-badge">📚 Answer informed by uploaded resources</span>',
                                unsafe_allow_html=True)
                st.markdown(answer)

                with st.expander("Retrieved context"):
                    st.text(context or "No context retrieved")
            except Exception as e:
                st.error(f"Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.title("📈 Student Analytics")
    st.caption("Track AO performance, progress over time, predicted grade")

    try:
        from analytics import get_analytics, list_students
        import plotly.express as px
        import plotly.graph_objects as go
        PLOTLY = True
    except ImportError:
        PLOTLY = False
        st.warning("Install plotly for charts: `pip install plotly`")

    students = []
    try:
        from analytics import list_students
        students = list_students()
    except Exception:
        pass

    if not students:
        st.info("No analytics data yet. Enter a Student ID when grading to start tracking.")
        st.stop()

    sel = st.selectbox("Select student:", students if not student_id.strip() else
                       ([student_id.strip()] + [s for s in students if s != student_id.strip()]))

    if sel:
        from analytics import get_analytics
        a = get_analytics(sel)

        if a.get("sessions", 0) == 0:
            st.info("No sessions recorded yet for this student.")
            st.stop()

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions", a["sessions"])
        c2.metric("Avg Score", f"{a['avg_mark_pct']}%" if a.get("avg_mark_pct") else "N/A")
        c3.metric("Predicted Grade", a.get("predicted_grade", "N/A"))
        c4.metric("Trend", "📈 Improving" if a.get("trending_up") else "📉 Declining")

        if PLOTLY and a.get("trend_data"):
            st.markdown("### Progress Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=a["trend_data"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=8),
                name="Score %"
            ))
            fig.add_hline(y=75, line_dash="dash", line_color="#22c55e", annotation_text="A grade")
            fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="C grade")
            fig.update_layout(
                yaxis_title="Score %", yaxis_range=[0, 100],
                height=280, margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # AO radar/bar
        if PLOTLY:
            st.markdown("### AO Performance")
            ao_labels = ["AO1 Knowledge", "AO2 Analysis", "AO3 Evaluation"]
            ao_values = [
                a.get("avg_ao1_pct") or 0,
                a.get("avg_ao2_pct") or 0,
                a.get("avg_ao3_pct") or 0,
            ]
            fig2 = go.Figure(go.Bar(
                x=ao_labels, y=ao_values,
                marker_color=["#6366f1","#3b82f6","#10b981"],
                text=[f"{v:.0f}%" for v in ao_values], textposition="outside",
            ))
            fig2.update_layout(
                yaxis_range=[0,110], height=250,
                margin=dict(l=0,r=0,t=20,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        if a.get("topic_breakdown"):
            st.markdown("### Topic Breakdown")
            topics = list(a["topic_breakdown"].keys())
            scores = list(a["topic_breakdown"].values())
            if PLOTLY and topics:
                fig3 = px.bar(x=scores, y=topics, orientation="h",
                               color=scores, color_continuous_scale="RdYlGn",
                               range_color=[0,100])
                fig3.update_layout(height=max(150, len(topics)*40),
                                   margin=dict(l=0,r=0,t=10,b=0),
                                   coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)

        # Improvement plan
        if a.get("improvement_plan"):
            st.markdown("### 🎯 Personalised Improvement Plan")
            for item in a["improvement_plan"]:
                st.markdown(item)

        # Session history
        with st.expander("Session History"):
            sessions = a.get("all_sessions", [])
            for s in reversed(sessions[-10:]):
                st.markdown(
                    f"`{s['timestamp'][:10]}` | {s['level']} | **{s['mark']}/{s['max_marks']}** "
                    f"({s['percentage']}%) | {s.get('topic','')}"
                )