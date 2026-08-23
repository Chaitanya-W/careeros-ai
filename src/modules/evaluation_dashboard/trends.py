"""Deterministic trend calculation from evaluation history.

Compares metric values across multiple :class:`EvaluationSummary`
snapshots. Trends are deterministic — Gemini never calculates trend values.

Directions:
- ``improving``: current > previous (delta > threshold)
- ``declining``: current < previous (delta < -threshold)
- ``stable``: |delta| <= threshold
- ``insufficient_data``: fewer than 2 data points

The threshold is a small epsilon (0.5) to avoid noise.
"""

from __future__ import annotations

from typing import Optional

from src.modules.evaluation_dashboard.model import (
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)

_THRESHOLD: float = 0.1  # minimum |delta| to be improving/declining


def compute_trends(history: list[EvaluationSummary]) -> list[TrendPoint]:
    """Compute trend points from a history of evaluation snapshots.

    Args:
        history: list of past EvaluationSummary objects (chronological).

    Returns:
        Trend points for: readiness, technical, communication,
        specificity, completeness, overall.
    """
    if not history or len(history) < 2:
        return _insufficient_trends()

    trend_defs = [
        ("Interview Readiness", lambda s: s.readiness_score),
        ("Technical Average", lambda s: s.answer_quality.technical_avg if s.answer_quality.total_evaluated > 0 else None),
        ("Communication Average", lambda s: s.answer_quality.communication_avg if s.answer_quality.total_evaluated > 0 else None),
        ("Specificity Average", lambda s: s.answer_quality.specificity_avg if s.answer_quality.total_evaluated > 0 else None),
        ("Completeness Average", lambda s: s.answer_quality.completeness_avg if s.answer_quality.total_evaluated > 0 else None),
        ("Overall Answer Quality", lambda s: s.answer_quality.overall_avg if s.answer_quality.total_evaluated > 0 else None),
    ]

    trends: list[TrendPoint] = []
    for label, getter in trend_defs:
        values = [getter(s) for s in history]
        # Filter out None values.
        valid = [v for v in values if v is not None]
        if len(valid) < 2:
            trends.append(TrendPoint(
                label=label,
                direction=TrendDirection.INSUFFICIENT_DATA,
                values=valid,
                current=valid[-1] if valid else None,
                previous=valid[-2] if len(valid) >= 2 else None,
                delta=None,
            ))
            continue

        current = valid[-1]
        previous = valid[-2]
        delta = round(current - previous, 1)

        if delta > _THRESHOLD:
            direction = TrendDirection.IMPROVING
        elif delta < -_THRESHOLD:
            direction = TrendDirection.DECLINING
        else:
            direction = TrendDirection.STABLE

        trends.append(TrendPoint(
            label=label,
            direction=direction,
            values=[round(v, 1) for v in valid],
            current=round(current, 1),
            previous=round(previous, 1),
            delta=delta,
        ))

    return trends


def _insufficient_trends() -> list[TrendPoint]:
    """Return insufficient-data trend points for all metrics."""
    labels = [
        "Interview Readiness",
        "Technical Average",
        "Communication Average",
        "Specificity Average",
        "Completeness Average",
        "Overall Answer Quality",
    ]
    return [
        TrendPoint(label=l, direction=TrendDirection.INSUFFICIENT_DATA)
        for l in labels
    ]
