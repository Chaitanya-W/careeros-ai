"""Tests for the Evaluation Dashboard UI (NO API key required)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationMetric,
    EvaluationReport,
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)
from src.modules.evaluation_dashboard.metrics import compute_metrics
from src.modules.resume_intelligence.model import ResumeAnalysis


def _sample_report() -> EvaluationReport:
    summary = EvaluationSummary(
        readiness_score=72,
        answer_quality=AnswerQuality(
            technical_avg=6.5, communication_avg=7.5, specificity_avg=5.5,
            completeness_avg=6.5, overall_avg=7.0, total_evaluated=2,
            evidence_alignment="aligned",
        ),
        job_match_score=68,
        skill_readiness=70,
        roadmap_readiness=68,
        negotiation_prepared=True,
        interview_completed=True,
        interview_total_questions=2,
        interview_completed_questions=2,
    )
    return EvaluationReport(
        summary=summary,
        metrics=[EvaluationMetric(name="readiness", value=72, available=True)],
        trends=[TrendPoint(label="Readiness", direction=TrendDirection.IMPROVING, values=[55, 72])],
        strengths=["Technical is your strongest dimension."],
        improvement_areas=["Specificity needs practice."],
        insights=["Interview readiness improved from 55% to 72%."],
        enriched=False,
    )


# ---- UI empty state (no resume) -----------------------------------


def test_eval_no_resume(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.run()
    assert not at.exception
    assert any("analyze your resume first" in m.value.lower() for m in at.markdown)
    assert any("Resume Intelligence" in b.label for b in at.button)


# ---- UI with resume + metrics -------------------------------------


def test_eval_with_resume_and_report(app_path: str, sample_copilot_resume) -> None:
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    # Metric cards render as markdown (not st.metric).
    all_md = " ".join(m.value for m in at.markdown)
    assert "72" in all_md or "Readiness" in all_md
    # Strengths section.
    assert any("Strengths" in s.value for s in at.subheader)
    # Insights.
    assert any("Insights" in s.value for s in at.subheader)


# ---- UI trends -----------------------------------------------------


def test_eval_trends_render(app_path: str, sample_copilot_resume) -> None:
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    assert any("Trends" in s.value for s in at.subheader)


# ---- UI answer quality --------------------------------------------


def test_eval_answer_quality(app_path: str, sample_copilot_resume) -> None:
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    assert any("Answer Quality" in s.value for s in at.subheader)


# ---- UI insights --------------------------------------------------


def test_eval_insights_render(app_path: str, sample_copilot_resume) -> None:
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "72%" in all_md  # insight mentions readiness


# ---- UI exports ---------------------------------------------------


def test_eval_exports(app_path: str, sample_copilot_resume) -> None:
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    assert any("Export" in s.value for s in at.subheader)


# ---- session-state preservation ----------------------------------


def test_eval_preserves_existing_state(app_path: str, sample_copilot_resume) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["resume_text"] = "PRESERVED"
    at.run()
    assert not at.exception
    assert at.session_state["resume_analysis"] is sample_copilot_resume
    assert at.session_state["resume_text"] == "PRESERVED"


def test_eval_initializes_state(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.run()
    assert not at.exception
    for key in ("evaluation_metrics", "evaluation_trends", "evaluation_report", "evaluation_insights"):
        assert key in at.session_state


# ---- no Gemini key ------------------------------------------------


def test_eval_no_key_message(app_path: str, sample_copilot_resume, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    report = _sample_report()
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "evaluation_dashboard"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["evaluation_report"] = report
    at.session_state["evaluation_metrics"] = {"current": report.summary, "history": []}
    at.run()
    assert not at.exception
    assert any("requires Gemini configuration" in i.value for i in at.info)
