"""Tests for the evaluation exporter (JSON + CSV)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.modules.evaluation_dashboard.exporter import export_csv, export_json
from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationMetric,
    EvaluationReport,
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)


def _sample_report() -> EvaluationReport:
    return EvaluationReport(
        summary=EvaluationSummary(
            readiness_score=72,
            answer_quality=AnswerQuality(
                technical_avg=6.5, total_evaluated=2, overall_avg=6.75,
            ),
            job_match_score=68,
        ),
        metrics=[EvaluationMetric(name="readiness", value=72, available=True)],
        trends=[TrendPoint(label="Readiness", direction=TrendDirection.IMPROVING)],
        insights=["Test insight with 72%."],
        enriched=False,
    )


# ---- JSON export ---------------------------------------------------


def test_export_json_structure() -> None:
    report = _sample_report()
    data = json.loads(export_json(report))
    assert data["summary"]["readiness_score"] == 72
    assert data["metrics"][0]["name"] == "readiness"
    assert data["trends"][0]["direction"] == "improving"
    assert data["insights"][0] == "Test insight with 72%."


def test_export_json_no_secrets() -> None:
    report = _sample_report()
    raw = export_json(report)
    assert "api_key" not in raw.lower()
    assert "secret" not in raw.lower()


# ---- CSV export ----------------------------------------------------


def test_export_csv_headers() -> None:
    raw = export_csv({})
    reader = csv.reader(io.StringIO(raw))
    headers = next(reader)
    assert "question_id" in headers
    assert "technical_score" in headers
    assert "overall_score" in headers
    assert "evidence_alignment" in headers


def test_export_csv_rows() -> None:
    from types import SimpleNamespace
    from src.modules.voice_interview.model import AnswerEvaluation

    evals = {
        "Q1": AnswerEvaluation(
            question_id="Q1", relevance_score=7, technical_score=6,
            specificity_score=5, communication_score=8, completeness_score=6,
            overall_score=6.5, evidence_alignment="aligned",
            unsupported_claims=[],
        ),
        "Q2": AnswerEvaluation(
            question_id="Q2", relevance_score=8, technical_score=7,
            specificity_score=6, communication_score=7, completeness_score=7,
            overall_score=7, evidence_alignment="aligned",
            unsupported_claims=["fake"],
        ),
    }
    raw = export_csv(evals)
    reader = csv.reader(io.StringIO(raw))
    headers = next(reader)
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0][0] == "Q1"
    assert rows[1][0] == "Q2"
    # Unsupported claims count.
    assert rows[0][-1] == "0"
    assert rows[1][-1] == "1"


def test_export_csv_empty() -> None:
    raw = export_csv({})
    reader = csv.reader(io.StringIO(raw))
    headers = next(reader)
    rows = list(reader)
    assert len(rows) == 0  # no data rows


def test_export_csv_with_questions() -> None:
    from src.modules.voice_interview.model import (
        AnswerEvaluation, InterviewCategory, InterviewQuestion,
    )

    evals = {"Q1": AnswerEvaluation(question_id="Q1", overall_score=7, evidence_alignment="aligned")}
    questions = [InterviewQuestion(question_id="Q1", category=InterviewCategory.TECHNICAL)]
    raw = export_csv(evals, questions)
    reader = csv.reader(io.StringIO(raw))
    next(reader)  # headers
    rows = list(reader)
    assert rows[0][0] == "Q1"
    # Category should be present.
    assert "technical" in rows[0][2]


def test_export_csv_no_secrets() -> None:
    raw = export_csv({})
    assert "api_key" not in raw.lower()
    assert "secret" not in raw.lower()
