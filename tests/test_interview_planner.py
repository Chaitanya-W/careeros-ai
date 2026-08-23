"""Tests for the deterministic interview question planner."""

from __future__ import annotations

import pytest

from src.modules.voice_interview.model import Difficulty, InterviewCategory
from src.modules.voice_interview.planner import (
    DEFAULT_RATIOS,
    build_questions,
    plan_distribution,
)
from src.modules.resume_intelligence.model import (
    ExperienceItem,
    ProjectItem,
    ResumeAnalysis,
)
from src.modules.jd_analysis.model import JobAnalysis
from types import SimpleNamespace


def _match(matching=None, missing=None):
    return SimpleNamespace(matching_skills=matching or [], missing_skills=missing or [])


# ---- (9) deterministic question planning --------------------------


def test_plan_distribution_default_10() -> None:
    dist = plan_distribution(10)
    assert len(dist) == 10
    # Default ratios: 3 technical, 2 job_specific, 1 resume, 1 project, 2 behavioral, 1 skill_gap.
    counts = {}
    for c in dist:
        counts[c] = counts.get(c, 0) + 1
    assert counts.get(InterviewCategory.TECHNICAL, 0) == 3
    assert counts.get(InterviewCategory.BEHAVIORAL, 0) == 2


def test_plan_distribution_n_zero() -> None:
    assert plan_distribution(0) == []


def test_plan_distribution_n_one() -> None:
    assert len(plan_distribution(1)) == 1


# ---- (10) question IDs -------------------------------------------


def test_question_ids_stable_and_sequential(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    dist = plan_distribution(6)
    qs = build_questions(dist, resume, job, match, sgr, difficulty=Difficulty.MEDIUM)
    assert [q.question_id for q in qs] == ["IQ-001", "IQ-002", "IQ-003", "IQ-004", "IQ-005", "IQ-006"]


def test_question_ids_deterministic(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    dist = plan_distribution(5)
    a = build_questions(dist, resume, job, match, sgr)
    b = build_questions(dist, resume, job, match, sgr)
    assert [q.question_id for q in a] == [q.question_id for q in b]
    assert [q.question for q in a] == [q.question for q in b]


# ---- (11) category distribution ---------------------------------


def test_build_questions_categories_present(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    dist = plan_distribution(8)
    qs = build_questions(dist, resume, job, match, sgr)
    cats = {q.category for q in qs}
    # With resume+job+match+skill_gap, technical + job_specific + behavioral should appear.
    assert InterviewCategory.TECHNICAL in cats
    assert InterviewCategory.JOB_SPECIFIC in cats
    assert InterviewCategory.BEHAVIORAL in cats


# ---- (12-17) category-specific question generation ---------------


def test_technical_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.TECHNICAL], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.TECHNICAL
    assert qs[0].target_skill  # references a real skill
    assert qs[0].question


def test_behavioral_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.BEHAVIORAL], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.BEHAVIORAL
    assert "Tell me about a time" in qs[0].question or "Describe" in qs[0].question


def test_project_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.PROJECT], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.PROJECT
    assert "CareerOS" in qs[0].question  # references the real project


def test_resume_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.RESUME], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.RESUME
    assert "Acme" in qs[0].question or "role" in qs[0].question.lower()


def test_job_specific_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.JOB_SPECIFIC], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.JOB_SPECIFIC
    # Must ask about a required skill (learning approach), not assert the candidate has it.
    assert "The role requires" in qs[0].question


def test_skill_gap_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions([InterviewCategory.SKILL_GAP], resume, job, match, sgr)
    assert qs[0].category is InterviewCategory.SKILL_GAP
    # Must ask about understanding/improving, not assert the candidate has it.
    assert "understanding" in qs[0].question.lower() or "improve" in qs[0].question.lower()


# ---- (18-21) missing prerequisites --------------------------------


def test_build_questions_missing_resume() -> None:
    """Without a resume, only behavioral questions (no invented facts)."""
    qs = build_questions([InterviewCategory.TECHNICAL], None, None, None, None)
    # Technical category reassigned to behavioral (no resume data).
    assert all(q.category is InterviewCategory.BEHAVIORAL for q in qs)


def test_build_questions_missing_job() -> None:
    """Without a job, job_specific reassigned to an available category."""
    resume = ResumeAnalysis(skills=["Python"])
    qs = build_questions([InterviewCategory.JOB_SPECIFIC], resume, None, _match(["Python"]), None)
    # Reassigned away from job_specific (no job data).
    assert all(q.category is not InterviewCategory.JOB_SPECIFIC for q in qs)


def test_build_questions_missing_match() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    job = JobAnalysis(required_skills=["Python"])
    qs = build_questions([InterviewCategory.TECHNICAL], resume, job, None, None)
    # Still produces a technical question from resume skills.
    assert any(q.category is InterviewCategory.TECHNICAL for q in qs)


def test_build_questions_missing_skill_gap() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    job = JobAnalysis(required_skills=["Python"])
    qs = build_questions([InterviewCategory.SKILL_GAP], resume, job, _match(["Python"]), None)
    # Reassigned away from skill_gap (no gap data).
    assert all(q.category is not InterviewCategory.SKILL_GAP for q in qs)


# ---- (22) empty skills -------------------------------------------


def test_build_questions_empty_skills() -> None:
    resume = ResumeAnalysis()  # no skills/experience/projects
    qs = build_questions(plan_distribution(3), resume, None, None, None)
    # All reassigned to behavioral (no resume data).
    assert all(q.category is InterviewCategory.BEHAVIORAL for q in qs)


# ---- (23) deterministic fallback ---------------------------------


def test_build_questions_never_invents_facts(sample_interview_inputs) -> None:
    resume, job, match, sgr, _ = sample_interview_inputs
    qs = build_questions(plan_distribution(8), resume, job, match, sgr)
    # No question should mention a skill NOT in the resume or job.
    resume_skills = {s.lower() for s in resume.skills}
    job_skills = {s.lower() for s in (job.required_skills + job.preferred_skills)}
    for q in qs:
        # Skill-gap questions mention a missing skill — that's allowed (asking about it).
        # Project questions' target_skill is the technologies field (may be a
        # comma-joined string), so skip the single-skill check for them.
        if q.category in (InterviewCategory.SKILL_GAP, InterviewCategory.PROJECT,
                          InterviewCategory.BEHAVIORAL):
            continue
        # The target_skill must be in resume or job.
        if q.target_skill:
            assert q.target_skill.lower() in resume_skills | job_skills, (
                f"Question invents skill {q.target_skill!r}"
            )


def test_default_ratios_sum_to_one() -> None:
    assert round(sum(DEFAULT_RATIOS.values()), 6) == 1.0
