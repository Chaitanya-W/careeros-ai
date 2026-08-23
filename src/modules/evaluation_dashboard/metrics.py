"""Deterministic metric extraction from CareerOS session state.

Reads structured session-state objects (interview evaluations, match
analysis, skill-gap report, career roadmap, salary negotiator outputs)
and computes a deterministic :class:`EvaluationSummary`. No Gemini.

Missing data is represented via ``None`` / ``available=False`` — never
invented.

Interview readiness formula (reused from Step 9):

    readiness = 10 × (
        0.30 × technical_avg
      + 0.20 × behavioral_avg
      + 0.20 × communication_avg
      + 0.20 × job_alignment_avg
      + 0.10 × completeness_avg
    )

Clamped to 0–100. NOT a hiring probability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationMetric,
    EvaluationSummary,
)


def compute_metrics(session: dict[str, Any]) -> EvaluationSummary:
    """Compute a deterministic EvaluationSummary from session state.

    ``session`` is a dict with the same keys as ``st.session_state``.
    Missing keys → the corresponding metric is ``None`` (not invented).
    """
    now = datetime.now(timezone.utc).isoformat()

    interview_report = session.get("interview_report")
    interview_evaluations = session.get("interview_evaluations")
    interview_questions = session.get("interview_questions")
    interview_session = session.get("interview_session")
    match_analysis = session.get("match_analysis")
    skill_gap_report = session.get("skill_gap_report")
    career_roadmap = session.get("career_roadmap")
    salary_benchmark = session.get("salary_benchmark")
    salary_script = session.get("salary_script")
    salary_counter_offer = session.get("salary_counter_offer")

    # --- Interview metrics ---
    readiness: Optional[float] = None
    interview_completed = False
    total_q = 0
    completed_q = 0

    if interview_report is not None:
        readiness = getattr(interview_report, "readiness_score", None)
        interview_completed = getattr(interview_report, "completed_questions", 0) > 0
        total_q = getattr(interview_report, "total_questions", 0)
        completed_q = getattr(interview_report, "completed_questions", 0)
    elif interview_session is not None:
        completed_q = len(getattr(interview_session, "evaluations", {}) or {})
        total_q = len(getattr(interview_session, "question_ids", []) or [])
        interview_completed = completed_q > 0

    # --- Answer quality from evaluations ---
    aq = _compute_answer_quality(interview_evaluations)

    # If no interview_report readiness but evaluations exist, compute from
    # the Step 9 formula using evaluation averages.
    if readiness is None and aq.total_evaluated > 0 and interview_questions:
        readiness = _compute_readiness_from_evaluations(
            interview_evaluations, interview_questions
        )

    # --- Job match ---
    job_match: Optional[float] = None
    if match_analysis is not None:
        job_match = getattr(match_analysis, "overall_match_score", None)

    # --- Skill readiness ---
    skill_readiness: Optional[float] = None
    if skill_gap_report is not None:
        skill_readiness = _compute_skill_readiness(skill_gap_report)

    # --- Roadmap readiness ---
    roadmap_readiness: Optional[float] = None
    if career_roadmap is not None:
        roadmap_readiness = getattr(career_roadmap, "current_readiness", None)

    # --- Negotiation prepared ---
    negotiation_prepared: Optional[bool] = None
    if salary_benchmark is not None and salary_script is not None and salary_counter_offer is not None:
        negotiation_prepared = True
    elif salary_benchmark is not None or salary_script is not None:
        negotiation_prepared = False

    return EvaluationSummary(
        readiness_score=readiness,
        answer_quality=aq,
        job_match_score=job_match,
        skill_readiness=skill_readiness,
        roadmap_readiness=roadmap_readiness,
        negotiation_prepared=negotiation_prepared,
        interview_completed=interview_completed,
        interview_total_questions=total_q,
        interview_completed_questions=completed_q,
        generated_at=now,
    )


def compute_answer_quality(evaluations: dict) -> AnswerQuality:
    """Public wrapper for answer-quality aggregation."""
    return _compute_answer_quality(evaluations)


def compute_metric_list(summary: EvaluationSummary) -> list[EvaluationMetric]:
    """Convert an EvaluationSummary into a list of EvaluationMetric cards."""
    metrics: list[EvaluationMetric] = []
    if summary.readiness_score is not None:
        metrics.append(EvaluationMetric(
            name="readiness", label="Interview Readiness",
            value=summary.readiness_score, unit="%", category="interview",
            available=True,
        ))
    if summary.answer_quality.total_evaluated > 0:
        metrics.append(EvaluationMetric(
            name="answer_quality", label="Answer Quality",
            value=summary.answer_quality.overall_avg, unit="score",
            category="interview", available=True,
        ))
    if summary.job_match_score is not None:
        metrics.append(EvaluationMetric(
            name="job_match", label="Job Match",
            value=summary.job_match_score, unit="%", category="job_match",
            available=True,
        ))
    if summary.skill_readiness is not None:
        metrics.append(EvaluationMetric(
            name="skill_readiness", label="Skill Readiness",
            value=summary.skill_readiness, unit="%", category="skill",
            available=True,
        ))
    if summary.roadmap_readiness is not None:
        metrics.append(EvaluationMetric(
            name="roadmap_readiness", label="Roadmap Readiness",
            value=summary.roadmap_readiness, unit="%", category="roadmap",
            available=True,
        ))
    if summary.negotiation_prepared is not None:
        metrics.append(EvaluationMetric(
            name="negotiation", label="Negotiation Prepared",
            value=1.0 if summary.negotiation_prepared else 0.0,
            unit="status", category="salary", available=True,
        ))
    return metrics


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _compute_answer_quality(evaluations) -> AnswerQuality:
    if not evaluations:
        return AnswerQuality()
    evals = list(evaluations.values()) if isinstance(evaluations, dict) else list(evaluations)
    if not evals:
        return AnswerQuality()

    n = len(evals)
    tech = _safe_avg([getattr(e, "technical_score", 0.0) for e in evals])
    rel = _safe_avg([getattr(e, "relevance_score", 0.0) for e in evals])
    spec = _safe_avg([getattr(e, "specificity_score", 0.0) for e in evals])
    comm = _safe_avg([getattr(e, "communication_score", 0.0) for e in evals])
    comp = _safe_avg([getattr(e, "completeness_score", 0.0) for e in evals])
    overall = _safe_avg([getattr(e, "overall_score", 0.0) for e in evals])
    unsupported = sum(len(getattr(e, "unsupported_claims", []) or []) for e in evals)

    # Evidence alignment — worst-case across all evaluations.
    alignments = [getattr(e, "evidence_alignment", "") for e in evals]
    if any(a == "unsupported" for a in alignments):
        ev_align = "unsupported"
    elif any(a == "partial" for a in alignments):
        ev_align = "partial"
    elif any(a == "aligned" for a in alignments):
        ev_align = "aligned"
    else:
        ev_align = "not_available"

    return AnswerQuality(
        technical_avg=round(tech, 1),
        relevance_avg=round(rel, 1),
        specificity_avg=round(spec, 1),
        communication_avg=round(comm, 1),
        completeness_avg=round(comp, 1),
        overall_avg=round(overall, 1),
        unsupported_claim_count=unsupported,
        evidence_alignment=ev_align,
        total_evaluated=n,
    )


def _compute_readiness_from_evaluations(evaluations, questions) -> Optional[float]:
    """Compute readiness using the Step 9 formula from raw evaluations."""
    if not evaluations or not questions:
        return None
    evals = list(evaluations.values()) if isinstance(evaluations, dict) else list(evaluations)
    if not evals:
        return None

    # Build a question lookup for category-based averaging.
    q_by_id = {}
    for q in questions:
        qid = getattr(q, "question_id", "")
        if qid:
            q_by_id[qid] = q

    tech_scores = []
    beh_scores = []
    comm_scores = []
    job_scores = []
    comp_scores = []
    overall_scores = []

    for ev in evals:
        qid = getattr(ev, "question_id", "")
        q = q_by_id.get(qid)
        tech_scores.append(getattr(ev, "technical_score", 0.0))
        comm_scores.append(getattr(ev, "communication_score", 0.0))
        comp_scores.append(getattr(ev, "completeness_score", 0.0))
        overall_scores.append(getattr(ev, "overall_score", 0.0))
        cat = getattr(q, "category", None) if q else None
        cat_val = getattr(cat, "value", str(cat)) if cat else ""
        if cat_val == "behavioral":
            beh_scores.append(getattr(ev, "overall_score", 0.0))
        if cat_val in ("job_specific", "skill_gap"):
            job_scores.append(getattr(ev, "relevance_score", 0.0))

    tech_avg = _safe_avg(tech_scores)
    beh_avg = _safe_avg(beh_scores) if beh_scores else _safe_avg(overall_scores)
    comm_avg = _safe_avg(comm_scores)
    job_avg = _safe_avg(job_scores) if job_scores else _safe_avg(overall_scores)
    comp_avg = _safe_avg(comp_scores)

    readiness = 10.0 * (
        0.30 * tech_avg
        + 0.20 * beh_avg
        + 0.20 * comm_avg
        + 0.20 * job_avg
        + 0.10 * comp_avg
    )
    return max(0.0, min(100.0, round(readiness)))


def _compute_skill_readiness(skill_gap_report) -> Optional[float]:
    """Heuristic: 100 - (high*20 + medium*10 + low*5), clamped 0-100."""
    high = getattr(skill_gap_report, "high_priority_count", 0)
    medium = getattr(skill_gap_report, "medium_priority_count", 0)
    low = getattr(skill_gap_report, "low_priority_count", 0)
    readiness = 100.0 - (high * 20 + medium * 10 + low * 5)
    return max(0.0, min(100.0, readiness))


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)
