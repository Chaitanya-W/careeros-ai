"""Tests for the Skill Gap UI workflow (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no enrichment button click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.jd_analysis.matcher import match_jobs
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import Priority, SkillGapReport
from src.modules.skill_gap.report import build_skill_gap_report


# ---- (14)(15) missing resume behavior -----------------------------------


def test_skill_gap_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.run()

    assert not at.exception, "render raised an exception"
    assert not at.error, "render reported an error"
    assert any(
        "analyze your resume first" in m.value.lower() for m in at.markdown
    ), "no-resume message missing"
    # Navigation to Resume Intelligence must be present.
    assert any(
        "Resume Intelligence" in b.label for b in at.button
    ), "navigation to Resume Intelligence missing"
    # No summary metrics when prerequisites are missing.
    assert all("Total Skill Gaps" not in m.label for m in at.metric) if at.metric else True


# ---- (16) missing job analysis behavior ---------------------------------


def test_skill_gap_renders_missing_job_message(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    # job_analysis / match_analysis NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze a target job first" in m.value.lower() for m in at.markdown
    ), "no-job message missing"
    assert any(
        "Job Match" in b.label for b in at.button
    ), "navigation to Job Match missing"


# ---- (17) missing match analysis behavior ------------------------------


def test_skill_gap_renders_missing_match_message(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    at.session_state["job_analysis"] = JobAnalysis(job_title="Eng")
    # match_analysis NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze a target job first" in m.value.lower() for m in at.markdown
    ), "no-match message missing"


# ---- (25) empty state (no gaps) ----------------------------------------


def test_skill_gap_renders_no_gaps_empty_state(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    """Resume covers everything -> 'No skill gaps detected' empty state."""
    job = JobAnalysis(
        job_title="Backend Engineer",
        required_skills=["Python", "SQL"],
        preferred_skills=[],
    )
    # match with no missing skills
    match = match_jobs(
        ResumeAnalysis(skills=["Python", "SQL"]), job
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    assert any(
        "No skill gaps detected" in m.value for m in at.markdown
    ), "no-gaps empty state missing"


# ---- (26) result rendering --------------------------------------------


def test_skill_gap_renders_results(
    app_path: str,
    sample_skill_gap_inputs,
) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = report  # pre-seed cached report
    at.run()

    assert not at.exception, "results render raised an exception"
    # NOTE: HIGH-priority badges are rendered via st.error("HIGH") (a colored
    # badge), so at.error legitimately contains those — they are not real
    # errors. We do NOT assert `not at.error` here.

    subheaders = [s.value for s in at.subheader]
    assert "Summary" in subheaders
    assert "Skill gaps" in subheaders

    # Summary metrics must render.
    labels = [m.label for m in at.metric]
    assert "Total Skill Gaps" in labels
    assert "High Priority" in labels
    assert "Medium Priority" in labels
    assert "Low Priority" in labels

    # Gap skills must appear (Docker/AWS HIGH, Kubernetes/Terraform MEDIUM).
    all_md = " ".join(m.value for m in at.markdown)
    for skill in ("Docker", "AWS", "Kubernetes", "Terraform"):
        assert skill in all_md, f"skill {skill} not rendered"


def test_skill_gap_shows_no_key_message_when_gemini_unconfigured(
    app_path: str,
    sample_skill_gap_inputs,
    monkeypatch,
) -> None:
    """Without a Gemini key, the deterministic report renders + a clear
    'AI requires configuration' message is shown."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match = sample_skill_gap_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    # Deterministic gaps still render.
    labels = [m.label for m in at.metric]
    assert "Total Skill Gaps" in labels
    # The no-key info message must appear.
    assert any(
        "AI learning recommendations require Gemini configuration" in i.value
        for i in at.info
    ), "no-key message missing"
    # The "Generate AI" button must NOT be present without a key.
    assert all(
        "Generate AI" not in b.label and "Regenerate AI" not in b.label
        for b in at.button
    ), "AI button should not render without a key"


# ---- (24) session-state storage ----------------------------------------


def test_skill_gap_report_is_cached_in_session_state(
    app_path: str,
    sample_skill_gap_inputs,
) -> None:
    """Re-rendering with the same inputs must reuse the cached report."""
    resume, job, match = sample_skill_gap_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()
    assert not at.exception
    assert "skill_gap_report" in at.session_state
    cached = at.session_state["skill_gap_report"]
    assert isinstance(cached, SkillGapReport)
    assert cached.total_gaps == 4  # AWS, Docker (HIGH) + Kubernetes, Terraform (MEDIUM)


def test_skill_gap_report_rebuilt_when_inputs_change(
    app_path: str,
    sample_skill_gap_inputs,
) -> None:
    """A stale cached report (different missing skills) must be rebuilt."""
    resume, job, match = sample_skill_gap_inputs
    # A stale report with a different signature.
    stale = build_skill_gap_report(resume, job, match)
    stale.skill_gaps = []  # mutate so the signature won't match
    object.__setattr__(stale, "_source_signature", ("old",), )

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = stale
    at.run()

    assert not at.exception
    assert "skill_gap_report" in at.session_state
    refreshed = at.session_state["skill_gap_report"]
    assert isinstance(refreshed, SkillGapReport)
    assert refreshed.total_gaps == 4  # rebuilt from current inputs


# ---- priority badges render -------------------------------------------


def test_skill_gap_renders_priority_badges(
    app_path: str,
    sample_skill_gap_inputs,
) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "skill_gap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = report
    at.run()

    assert not at.exception
    # HIGH badges (error), MEDIUM badges (warning) must appear.
    assert any("HIGH" in e.value for e in at.error)
    assert any("MEDIUM" in w.value for w in at.warning)
