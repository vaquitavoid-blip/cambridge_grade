# src/analytics.py
# Phase 9 — Student Analytics: track AO performance, progress, predict grades

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent))
from config import ANALYTICS_DIR


def _get_student_file(student_id: str) -> Path:
    return ANALYTICS_DIR / f"{student_id}.json"


def _load_student(student_id: str) -> dict:
    f = _get_student_file(student_id)
    if f.exists():
        with open(f, encoding="utf-8") as fp:
            return json.load(fp)
    return {
        "student_id": student_id,
        "sessions": [],
        "created": datetime.now().isoformat(),
    }


def _save_student(student_id: str, data: dict):
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_get_student_file(student_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_session(
    student_id:  str,
    question:    str,
    level:       str,
    max_marks:   int,
    mark:        int,
    ao1_mark:    Optional[int],
    ao2_mark:    Optional[int],
    ao3_mark:    Optional[int],
    topic:       str = "",
    mode:        str = "grade",
):
    """Records a grading session for analytics tracking."""
    data    = _load_student(student_id)
    session = {
        "timestamp":  datetime.now().isoformat(),
        "question":   question[:100],
        "level":      level,
        "max_marks":  max_marks,
        "mark":       mark,
        "percentage": round(mark / max_marks * 100) if max_marks else 0,
        "ao1_mark":   ao1_mark,
        "ao1_max":    2,
        "ao2_mark":   ao2_mark,
        "ao2_max":    6 if level == "AS" else 6,
        "ao3_mark":   ao3_mark if level == "AS" else None,
        "ao3_max":    4 if level == "AS" else None,
        "topic":      topic,
        "mode":       mode,
    }
    data["sessions"].append(session)
    _save_student(student_id, data)


def get_analytics(student_id: str) -> dict:
    """Returns full analytics dict for a student."""
    data     = _load_student(student_id)
    sessions = data.get("sessions", [])

    if not sessions:
        return {"student_id": student_id, "sessions": 0, "summary": None}

    marks       = [s["mark"] for s in sessions if s.get("mark") is not None]
    percentages = [s["percentage"] for s in sessions if s.get("percentage") is not None]
    ao1s = [s["ao1_mark"] for s in sessions if s.get("ao1_mark") is not None]
    ao2s = [s["ao2_mark"] for s in sessions if s.get("ao2_mark") is not None]
    ao3s = [s["ao3_mark"] for s in sessions if s.get("ao3_mark") is not None]

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    def safe_pct(lst, max_val):
        if not lst or not max_val:
            return None
        return round(safe_avg(lst) / max_val * 100, 1)

    # Topic breakdown
    topic_marks = {}
    for s in sessions:
        t = s.get("topic", "Unknown") or "Unknown"
        if t not in topic_marks:
            topic_marks[t] = []
        if s.get("percentage") is not None:
            topic_marks[t].append(s["percentage"])

    topic_avg = {t: safe_avg(v) for t, v in topic_marks.items()}

    # Weakest topic
    weakest = min(topic_avg, key=topic_avg.get) if topic_avg else None

    # Trend (last 5 sessions)
    trend = [s["percentage"] for s in sessions[-5:] if s.get("percentage") is not None]
    trending_up = len(trend) >= 2 and trend[-1] > trend[0]

    # Grade prediction (rough Cambridge grade thresholds)
    avg_pct = safe_avg(percentages)
    predicted_grade = None
    if avg_pct is not None:
        if avg_pct >= 80:   predicted_grade = "A*"
        elif avg_pct >= 70: predicted_grade = "A"
        elif avg_pct >= 60: predicted_grade = "B"
        elif avg_pct >= 50: predicted_grade = "C"
        elif avg_pct >= 40: predicted_grade = "D"
        else:               predicted_grade = "E/U"

    # Improvement plan
    improvement_plan = _generate_improvement_plan(
        safe_pct(ao1s, 2),
        safe_pct(ao2s, 6),
        safe_pct(ao3s, 4) if ao3s else None,
        weakest,
    )

    return {
        "student_id":       student_id,
        "sessions":         len(sessions),
        "avg_mark_pct":     avg_pct,
        "avg_ao1_pct":      safe_pct(ao1s, 2),
        "avg_ao2_pct":      safe_pct(ao2s, 6),
        "avg_ao3_pct":      safe_pct(ao3s, 4),
        "predicted_grade":  predicted_grade,
        "trending_up":      trending_up,
        "trend_data":       trend,
        "topic_breakdown":  topic_avg,
        "weakest_topic":    weakest,
        "improvement_plan": improvement_plan,
        "all_sessions":     sessions,
    }


def _generate_improvement_plan(ao1_pct, ao2_pct, ao3_pct, weakest_topic) -> list[str]:
    plan = []

    if ao1_pct is not None and ao1_pct < 70:
        plan.append("📖 Knowledge (AO1): Review core economic definitions and terminology for your weak topics.")
    if ao2_pct is not None and ao2_pct < 65:
        plan.append("🔗 Analysis (AO2): Practise completing SEDE chains — every point needs State → Explain → Develop → Example.")
    if ao3_pct is not None and ao3_pct < 60:
        plan.append("⚖️ Evaluation (AO3): Focus on supported judgments — always answer 'to what extent' with conditions and reasoning.")
    if weakest_topic:
        plan.append(f"📌 Topic focus: Your weakest area is '{weakest_topic}' — practise essays specifically on this topic.")
    if not plan:
        plan.append("✅ Strong performance across all areas. Focus on consistency and maintaining your grade.")
    return plan


def list_students() -> list[str]:
    """Returns all student IDs with recorded sessions."""
    return [f.stem for f in ANALYTICS_DIR.glob("*.json")]