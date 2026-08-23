"""Tests for the Evaluation Dashboard data models."""

from __future__ import annotations

from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationError,
    EvaluationMetric,
    EvaluationReport,
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)


# ---- TrendDirection -----------------------------------------------


def test_trend_direction_members() -> None:
    assert {t.value for t in TrendDirection} == {
        "improving", "declining", "stable", "insufficient_data"
    }


def test_trend_direction_from_value() -> None:
    assert TrendDirection.from_value("IMPROVING") is TrendDirection.IMPROVING
    assert TrendDirection.from_value(None) is TrendDirection.INSUFFICIENT_DATA
    assert TrendDirection.from_value("bogus") is TrendDirection.INSUFFICIENT_DATA


# ---- EvaluationMetric ---------------------------------------------


def test_metric_defaults() -> None:
    m = EvaluationMetric()
    assert m.name == ""
    assert m.value == 0.0
    assert m.available is False


def test_metric_to_dict() -> None:
    m = EvaluationMetric(name="readiness", value=72, unit="%", available=True)
    d = m.to_dict()
    assert d["name"] == "readiness"
    assert d["value"] == 72
    assert d["available"] is True


# ---- AnswerQuality -------------------------------------------------


def test_answer_quality_defaults() -> None:
    aq = AnswerQuality()
    assert aq.technical_avg == 0.0
    assert aq.total_evaluated == 0
    assert aq.evidence_alignment == "not_available"


def test_answer_quality_to_dict() -> None:
    aq = AnswerQuality(technical_avg=6.5, total_evaluated=3, evidence_alignment="aligned")
    d = aq.to_dict()
    assert d["technical_avg"] == 6.5
    assert d["total_evaluated"] == 3


# ---- EvaluationSummary --------------------------------------------


def test_summary_defaults() -> None:
    s = EvaluationSummary()
    assert s.readiness_score is None
    assert s.job_match_score is None
    assert s.interview_completed is False


def test_summary_to_dict() -> None:
    s = EvaluationSummary(readiness_score=72, job_match_score=68)
    d = s.to_dict()
    assert d["readiness_score"] == 72
    assert d["job_match_score"] == 68
    assert isinstance(d["answer_quality"], dict)


# ---- TrendPoint ---------------------------------------------------


def test_trend_point_defaults() -> None:
    tp = TrendPoint()
    assert tp.direction is TrendDirection.INSUFFICIENT_DATA
    assert tp.values == []


def test_trend_point_to_dict() -> None:
    tp = TrendPoint(label="Readiness", direction=TrendDirection.IMPROVING, values=[55, 72])
    d = tp.to_dict()
    assert d["label"] == "Readiness"
    assert d["direction"] == "improving"
    assert d["values"] == [55, 72]


# ---- EvaluationReport ---------------------------------------------


def test_report_defaults() -> None:
    r = EvaluationReport()
    assert r.metrics == []
    assert r.trends == []
    assert r.insights == []
    assert r.enriched is False


def test_report_to_dict() -> None:
    r = EvaluationReport(
        summary=EvaluationSummary(readiness_score=72),
        metrics=[EvaluationMetric(name="readiness", value=72, available=True)],
        insights=["Test insight."],
        enriched=True,
    )
    d = r.to_dict()
    assert d["summary"]["readiness_score"] == 72
    assert d["metrics"][0]["name"] == "readiness"
    assert d["insights"] == ["Test insight."]
    assert d["enriched"] is True


def test_error_hierarchy() -> None:
    from src.modules.evaluation_dashboard.model import EvaluationParseError
    assert issubclass(EvaluationParseError, EvaluationError)
