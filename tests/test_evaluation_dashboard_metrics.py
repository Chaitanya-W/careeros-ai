"""Tests for the deterministic metric extraction."""

from __future__ import annotations

import pytest

from src.modules.evaluation_dashboard.metrics import (
    compute_answer_quality,
    compute_metric_list,
    compute_metrics,
)
from src.modules.evaluation_dashboard.model import EvaluationSummary


# ---- empty metrics ------------------------------------------------


def test_empty_session() -> None:
    s = compute_metrics({})
    assert s.readiness_score is None
    assert s.job_match_score is None
    assert s.skill_readiness is None
    assert s.roadmap_readiness is None
    assert s.negotiation_prepared is None
    assert s.interview_completed is False


# ---- interview metrics --------------------------------------------


def test_interview_report_readiness(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    assert s.readiness_score == 72
    assert s.interview_completed is True
    assert s.interview_total_questions == 2
    assert s.interview_completed_questions == 2


def test_answer_quality_aggregation(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    aq = s.answer_quality
    assert aq.total_evaluated == 2
    assert aq.technical_avg == 6.5  # (6+7)/2
    assert aq.communication_avg == 7.5  # (8+7)/2
    assert aq.overall_avg == 6.8  # (6.5+7)/2 = 6.75 → rounded to 6.8
    assert aq.unsupported_claim_count == 0
    assert aq.evidence_alignment == "aligned"


def test_unsupported_claims_counted() -> None:
    from types import SimpleNamespace
    from src.modules.voice_interview.model import AnswerEvaluation

    evals = {
        "Q1": AnswerEvaluation(
            question_id="Q1", technical_score=5, overall_score=5,
            unsupported_claims=["claim1", "claim2"],
            evidence_alignment="unsupported",
        ),
        "Q2": AnswerEvaluation(
            question_id="Q2", technical_score=7, overall_score=7,
            unsupported_claims=["claim3"],
            evidence_alignment="partial",
        ),
    }
    aq = compute_answer_quality(evals)
    assert aq.unsupported_claim_count == 3
    assert aq.evidence_alignment == "unsupported"  # worst-case


def test_evidence_alignment_worst_case() -> None:
    from types import SimpleNamespace
    from src.modules.voice_interview.model import AnswerEvaluation

    evals = {
        "Q1": AnswerEvaluation(evidence_alignment="aligned"),
        "Q2": AnswerEvaluation(evidence_alignment="partial"),
    }
    aq = compute_answer_quality(evals)
    assert aq.evidence_alignment == "partial"


# ---- category averages + readiness from evaluations ----------------


def test_readiness_from_evaluations() -> None:
    """When no interview_report exists, compute readiness from evaluations."""
    from types import SimpleNamespace
    from src.modules.voice_interview.model import (
        AnswerEvaluation, InterviewCategory, InterviewQuestion,
    )

    evals = {
        "Q1": AnswerEvaluation(
            question_id="Q1", relevance_score=7, technical_score=6,
            specificity_score=5, communication_score=8, completeness_score=6,
            overall_score=6.5, evidence_alignment="aligned",
        ),
        "Q2": AnswerEvaluation(
            question_id="Q2", relevance_score=8, technical_score=7,
            specificity_score=6, communication_score=7, completeness_score=7,
            overall_score=7, evidence_alignment="aligned",
        ),
    }
    questions = [
        InterviewQuestion(question_id="Q1", category=InterviewCategory.TECHNICAL),
        InterviewQuestion(question_id="Q2", category=InterviewCategory.BEHAVIORAL),
    ]
    session = {"interview_evaluations": evals, "interview_questions": questions}
    s = compute_metrics(session)
    assert s.readiness_score is not None
    assert 0 <= s.readiness_score <= 100


# ---- job-match metrics --------------------------------------------


def test_job_match_score(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    assert s.job_match_score == 68


def test_no_match_analysis() -> None:
    s = compute_metrics({})
    assert s.job_match_score is None


# ---- skill-gap metrics --------------------------------------------


def test_skill_readiness(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    # 1 high + 1 medium + 0 low → 100 - (20 + 10 + 0) = 70
    assert s.skill_readiness == 70


def test_no_skill_gap() -> None:
    s = compute_metrics({})
    assert s.skill_readiness is None


# ---- roadmap metrics ----------------------------------------------


def test_roadmap_readiness(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    assert s.roadmap_readiness == 68


def test_no_roadmap() -> None:
    s = compute_metrics({})
    assert s.roadmap_readiness is None


# ---- salary-preparation metric ------------------------------------


def test_negotiation_prepared(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    assert s.negotiation_prepared is True


def test_negotiation_partial() -> None:
    from types import SimpleNamespace
    session = {"salary_benchmark": SimpleNamespace(mid=100000)}
    s = compute_metrics(session)
    assert s.negotiation_prepared is False


def test_no_salary_data() -> None:
    s = compute_metrics({})
    assert s.negotiation_prepared is None


# --- metric list ---------------------------------------------------


def test_metric_list(sample_eval_session) -> None:
    s = compute_metrics(sample_eval_session)
    ml = compute_metric_list(s)
    names = [m.name for m in ml]
    assert "readiness" in names
    assert "job_match" in names
    assert "skill_readiness" in names
    assert "roadmap_readiness" in names
    assert "negotiation" in names


def test_metric_list_empty() -> None:
    s = EvaluationSummary()
    ml = compute_metric_list(s)
    assert ml == []


# --- determinism ---------------------------------------------------


def test_metrics_deterministic(sample_eval_session) -> None:
    a = compute_metrics(sample_eval_session)
    b = compute_metrics(sample_eval_session)
    assert a.readiness_score == b.readiness_score
    assert a.answer_quality.technical_avg == b.answer_quality.technical_avg
    assert a.job_match_score == b.job_match_score


# --- clamping ------------------------------------------------------


def test_skill_readiness_clamped() -> None:
    from types import SimpleNamespace
    # 10 high gaps → 100 - 200 = -100 → clamped to 0
    gap = SimpleNamespace(high_priority_count=10, medium_priority_count=0, low_priority_count=0)
    session = {"skill_gap_report": gap}
    s = compute_metrics(session)
    assert s.skill_readiness == 0
