"""Tests for the Application Copilot UI workflow (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no AI button click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.jd_analysis.matcher import match_jobs
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis


# ---- (30) missing ResumeAnalysis -----------------------------------


def test_copilot_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.run()

    assert not at.exception
    assert any(
        "analyze your resume first" in m.value.lower() for m in at.markdown
    ), "no-resume message missing"
    assert any(
        "Resume Intelligence" in b.label for b in at.button
    ), "navigation to Resume Intelligence missing"


# ---- (31)(32) missing JobAnalysis / MatchAnalysis ------------------


def test_copilot_renders_missing_job_message(
    app_path: str,
    sample_copilot_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = sample_copilot_resume
    # job/match NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze a target job first" in m.value.lower() for m in at.markdown
    ), "no-job message missing"
    assert any("Job Match" in b.label for b in at.button)


def test_copilot_renders_missing_match_message(
    app_path: str,
    sample_copilot_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = sample_copilot_resume
    at.session_state["job_analysis"] = JobAnalysis(job_title="Eng", required_skills=["Python"])
    # match_analysis NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze a target job first" in m.value.lower() for m in at.markdown
    ), "no-match message missing"


# ---- (34) UI empty state (prerequisites missing, covered above) ----


# ---- (35) UI result rendering ---------------------------------------


def test_copilot_renders_tabs_and_status(
    app_path: str,
    sample_copilot_inputs,
) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception, "render raised an exception"
    # The three tabs must be present.
    tab_titles = [t.label for t in at.tabs]
    assert "Resume Tailor" in tab_titles
    assert "Cover Letter" in tab_titles
    assert "Bullet Optimizer" in tab_titles

    # Status bar metrics must render.
    labels = [m.label for m in at.metric]
    assert "Match score" in labels
    assert "Evidence items" in labels


def test_copilot_renders_no_key_message_when_gemini_unconfigured(
    app_path: str,
    sample_copilot_inputs,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, evidence = sample_copilot_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    # At least one no-key info message must appear across the tabs.
    assert any(
        "requires Gemini configuration" in i.value for i in at.info
    ), "no-key message missing"


# ---- (33) session-state behavior ------------------------------------


def test_copilot_caches_evidence_in_session_state(
    app_path: str,
    sample_copilot_inputs,
) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    assert "evidence_items" in at.session_state
    cached = at.session_state["evidence_items"]
    assert isinstance(cached, list)
    assert len(cached) > 0


def test_copilot_does_not_overwrite_existing_state(
    app_path: str,
    sample_copilot_inputs,
) -> None:
    """The copilot must not overwrite resume/job/match/roadmap keys."""
    resume, job, match, evidence = sample_copilot_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "application_copilot"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    # Existing keys preserved.
    assert at.session_state["resume_analysis"] is resume
    assert at.session_state["job_analysis"] is job
    assert at.session_state["match_analysis"] is match
