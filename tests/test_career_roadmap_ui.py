"""Tests for the Career Roadmap UI workflow (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no enrichment button click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.career_roadmap.builder import build_career_roadmap
from src.modules.career_roadmap.model import CareerRoadmap
from src.modules.jd_analysis.matcher import match_jobs
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.report import build_skill_gap_report


# ---- (6)(7) missing prerequisites / missing SkillGapReport ------------


def test_roadmap_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.run()

    assert not at.exception
    assert any(
        "analyze your resume first" in m.value.lower() for m in at.markdown
    ), "no-resume message missing"
    assert any(
        "Resume Intelligence" in b.label for b in at.button
    ), "navigation to Resume Intelligence missing"


def test_roadmap_renders_missing_job_message(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    # job/match NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze a target job first" in m.value.lower() for m in at.markdown
    ), "no-job message missing"
    assert any("Job Match" in b.label for b in at.button)


def test_roadmap_renders_missing_skill_gap_report_message(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    job = JobAnalysis(job_title="Eng", required_skills=["Python"])
    match = match_jobs(ResumeAnalysis(skills=["Python"]), job)
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    # skill_gap_report NOT set.
    at.run()

    assert not at.exception
    assert any(
        "analyze your skill gaps first" in m.value.lower() for m in at.markdown
    ), "no-skill-gap message missing"
    assert any("Skill Gap" in b.label for b in at.button)


# ---- (8) empty skill gaps ---------------------------------------------


def test_roadmap_renders_empty_gaps_positive_state(
    app_path: str,
    sample_skill_gap_resume: ResumeAnalysis,
) -> None:
    """No skill gaps -> positive 'covers the requirements' message."""
    job = JobAnalysis(job_title="Eng", required_skills=["Python", "SQL"])
    match = match_jobs(ResumeAnalysis(skills=["Python", "SQL"]), job)
    report = build_skill_gap_report(
        ResumeAnalysis(skills=["Python", "SQL"]), job, match
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = sample_skill_gap_resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = report
    at.run()

    assert not at.exception
    # The positive empty-gaps hint is rendered as a caption by empty_state_panel.
    assert any(
        "No major skill gaps" in c.value for c in at.caption
    ), "positive empty-gaps state missing"


# ---- (24) roadmap UI empty state (no prerequisites) ------------------ (covered above)


# ---- (25) roadmap UI result rendering --------------------------------


def test_roadmap_renders_results(
    app_path: str,
    sample_skill_gap_report,
    sample_skill_gap_inputs,
) -> None:
    resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sample_skill_gap_report
    at.session_state["career_roadmap"] = roadmap  # pre-seed cached
    at.run()

    assert not at.exception, "results render raised an exception"
    subheaders = [s.value for s in at.subheader]
    assert "Roadmap summary" in subheaders
    assert "Learning phases" in subheaders

    # Summary metrics must render.
    labels = [m.label for m in at.metric]
    assert "Current readiness" in labels
    assert "Phases" in labels

    # Phase titles must render ("Phase 1: ...").
    all_md = " ".join(m.value for m in at.markdown)
    assert "Phase 1" in all_md
    # Roadmap skills must appear (Docker/AWS HIGH, Kubernetes/Terraform MEDIUM).
    for skill in ("Docker", "AWS", "Kubernetes", "Terraform"):
        assert skill in all_md, f"skill {skill} not rendered"


def test_roadmap_shows_no_key_message_when_gemini_unconfigured(
    app_path: str,
    sample_skill_gap_report,
    sample_skill_gap_inputs,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match = sample_skill_gap_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sample_skill_gap_report
    at.run()

    assert not at.exception
    # Deterministic roadmap still renders.
    labels = [m.label for m in at.metric]
    assert "Current readiness" in labels
    # The no-key info message must appear.
    assert any(
        "AI roadmap enrichment requires Gemini configuration" in i.value
        for i in at.info
    ), "no-key message missing"
    # AI buttons must NOT render without a key.
    assert all(
        "Generate AI" not in b.label and "Regenerate AI" not in b.label
        for b in at.button
    ), "AI button should not render without a key"


# ---- (23) session-state storage --------------------------------------


def test_roadmap_is_cached_in_session_state(
    app_path: str,
    sample_skill_gap_report,
    sample_skill_gap_inputs,
) -> None:
    resume, job, match = sample_skill_gap_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sample_skill_gap_report
    at.run()

    assert not at.exception
    assert "career_roadmap" in at.session_state
    cached = at.session_state["career_roadmap"]
    assert isinstance(cached, CareerRoadmap)
    assert cached.phase_count >= 1
    # Source of truth: cached roadmap skills == skill_gap_report skills.
    report_skills = {g.skill.lower() for g in sample_skill_gap_report.skill_gaps}
    cached_skills = {s.lower() for s in cached.all_skills}
    assert cached_skills == report_skills


def test_roadmap_rebuilt_when_skill_gap_report_changes(
    app_path: str,
    sample_skill_gap_report,
    sample_skill_gap_inputs,
) -> None:
    """A stale cached roadmap (different signature) must be rebuilt."""
    resume, job, match = sample_skill_gap_inputs
    # A stale roadmap with a different skill set.
    from src.modules.career_roadmap.model import CareerRoadmap, RoadmapPhase

    stale = CareerRoadmap(target_role="Old", phases=[RoadmapPhase(phase_number=1, skills=["OldSkill"])])
    object.__setattr__(stale, "_source_signature", (("OldSkill",), "Old"))

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "career_roadmap"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sample_skill_gap_report
    at.session_state["career_roadmap"] = stale
    at.run()

    assert not at.exception
    refreshed = at.session_state["career_roadmap"]
    assert isinstance(refreshed, CareerRoadmap)
    # "OldSkill" must be gone; real gap skills present.
    assert "OldSkill" not in refreshed.all_skills
    assert "Docker" in refreshed.all_skills
