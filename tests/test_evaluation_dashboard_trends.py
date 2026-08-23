"""Tests for the deterministic trend calculation."""

from __future__ import annotations

import pytest

from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationSummary,
    TrendDirection,
)
from src.modules.evaluation_dashboard.trends import compute_trends


# ---- insufficient data --------------------------------------------


def test_empty_history() -> None:
    trends = compute_trends([])
    assert all(t.direction is TrendDirection.INSUFFICIENT_DATA for t in trends)
    assert len(trends) == 6


def test_single_snapshot() -> None:
    """One data point → insufficient_data."""
    history = [EvaluationSummary(readiness_score=72)]
    trends = compute_trends(history)
    assert all(t.direction is TrendDirection.INSUFFICIENT_DATA for t in trends)


# ---- improving trend ----------------------------------------------


def test_improving_trend(sample_eval_history) -> None:
    trends = compute_trends(sample_eval_history)
    readiness = next(t for t in trends if t.label == "Interview Readiness")
    assert readiness.direction is TrendDirection.IMPROVING
    assert readiness.delta > 0
    assert readiness.current == 72
    assert readiness.previous == 65


def test_technical_improving(sample_eval_history) -> None:
    trends = compute_trends(sample_eval_history)
    tech = next(t for t in trends if t.label == "Technical Average")
    assert tech.direction is TrendDirection.IMPROVING


def test_communication_improving(sample_eval_history) -> None:
    trends = compute_trends(sample_eval_history)
    comm = next(t for t in trends if t.label == "Communication Average")
    assert comm.direction is TrendDirection.IMPROVING


# ---- declining trend ----------------------------------------------


def test_declining_trend() -> None:
    history = [
        EvaluationSummary(
            readiness_score=72,
            answer_quality=AnswerQuality(
                technical_avg=7, communication_avg=8, specificity_avg=6,
                completeness_avg=7, overall_avg=7, total_evaluated=2,
            ),
        ),
        EvaluationSummary(
            readiness_score=55,
            answer_quality=AnswerQuality(
                technical_avg=5, communication_avg=6, specificity_avg=4,
                completeness_avg=5, overall_avg=5, total_evaluated=2,
            ),
        ),
    ]
    trends = compute_trends(history)
    readiness = next(t for t in trends if t.label == "Interview Readiness")
    assert readiness.direction is TrendDirection.DECLINING
    assert readiness.delta < 0


# ---- stable trend -------------------------------------------------


def test_stable_trend() -> None:
    history = [
        EvaluationSummary(
            readiness_score=72,
            answer_quality=AnswerQuality(
                technical_avg=6.5, communication_avg=7.5, specificity_avg=5.5,
                completeness_avg=6.5, overall_avg=7.0, total_evaluated=2,
            ),
        ),
        EvaluationSummary(
            readiness_score=72,
            answer_quality=AnswerQuality(
                technical_avg=6.5, communication_avg=7.5, specificity_avg=5.5,
                completeness_avg=6.5, overall_avg=7.0, total_evaluated=2,
            ),
        ),
    ]
    trends = compute_trends(history)
    readiness = next(t for t in trends if t.label == "Interview Readiness")
    assert readiness.direction is TrendDirection.STABLE


# --- multi-snapshot with None values --------------------------------


def test_none_values_filtered() -> None:
    """If some snapshots have None readiness, they're filtered out."""
    history = [
        EvaluationSummary(readiness_score=50),
        EvaluationSummary(readiness_score=None),  # no interview yet
        EvaluationSummary(readiness_score=70),
    ]
    trends = compute_trends(history)
    readiness = next(t for t in trends if t.label == "Interview Readiness")
    # Only 2 valid points → trend computable.
    assert readiness.direction is TrendDirection.IMPROVING
    assert readiness.current == 70
    assert readiness.previous == 50


# --- determinism ---------------------------------------------------


def test_trends_deterministic(sample_eval_history) -> None:
    a = compute_trends(sample_eval_history)
    b = compute_trends(sample_eval_history)
    assert [t.direction for t in a] == [t.direction for t in b]
    assert [t.delta for t in a] == [t.delta for t in b]
