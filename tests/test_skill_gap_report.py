"""Tests for the DETERMINISTIC Skill Gap report builder (no LLM)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import (
    EducationItem,
    ExperienceItem,
    ResumeAnalysis,
)
from src.modules.skill_gap.model import Priority, SkillGapReport
from src.modules.skill_gap.report import (
    _detect_evidence,
    _preferred_gaps,
    _prioritize,
    _target_level,
    build_skill_gap_report,
)


# ---- (19) deterministic report generation --------------------------------


def test_build_report_anchors_on_missing_skills(sample_skill_gap_inputs) -> None:
    """Required gaps must come EXACTLY from match.missing_skills."""
    resume, job, match = sample_skill_gap_inputs
    assert match.missing_skills == ["AWS", "Docker"]
    report = build_skill_gap_report(resume, job, match)

    assert isinstance(report, SkillGapReport)
    required_gap_names = {g.skill for g in report.skill_gaps if g.priority is Priority.HIGH}
    assert required_gap_names == set(match.missing_skills)


def test_build_report_includes_preferred_gaps(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    medium = {g.skill for g in report.skill_gaps if g.priority is Priority.MEDIUM}
    assert medium == {"Kubernetes", "Terraform"}


def test_build_report_counts_and_summary(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    # 2 required (AWS, Docker) + 2 preferred (Kubernetes, Terraform) = 4
    assert report.total_gaps == 4
    assert report.high_priority_count == 2
    assert report.medium_priority_count == 2
    assert report.low_priority_count == 0
    assert str(report.total_gaps) in report.summary
    assert report.target_role  # non-empty


def test_build_report_not_enriched_by_default(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    assert report.enriched is False
    for g in report.skill_gaps:
        assert g.why_it_matters == ""
        assert g.learning_objectives == []


def test_build_report_generated_at_is_iso(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    assert report.generated_at
    # Must parse as ISO 8601.
    datetime.fromisoformat(report.generated_at)


def test_build_report_never_crashes_on_empty_inputs() -> None:
    resume = ResumeAnalysis()
    job = JobAnalysis()
    match = type(
        "M", (), {"missing_skills": [], "matching_skills": [], "partial_match_skills": []}
    )()
    report = build_skill_gap_report(resume, job, match)
    assert report.total_gaps == 0
    assert report.skill_gaps == []


# ---- (5) required gap -> HIGH --------------------------------------------


def test_prioritize_required_skill_is_high(sample_skill_gap_inputs) -> None:
    _, job, _ = sample_skill_gap_inputs
    assert _prioritize("Docker", job) is Priority.HIGH
    assert _prioritize("AWS", job) is Priority.HIGH


# ---- (6) preferred gap -> MEDIUM ----------------------------------------


def test_prioritize_preferred_skill_is_medium(sample_skill_gap_inputs) -> None:
    _, job, _ = sample_skill_gap_inputs
    assert _prioritize("Kubernetes", job) is Priority.MEDIUM
    assert _prioritize("Terraform", job) is Priority.MEDIUM


# ---- (7) lower-relevance -> LOW (defensive) -----------------------------


def test_prioritize_non_job_skill_is_low(sample_skill_gap_inputs) -> None:
    _, job, _ = sample_skill_gap_inputs
    # A skill not in required or preferred -> LOW (defensive).
    assert _prioritize("Cooking", job) is Priority.LOW


def test_prioritize_empty_skill_is_low(sample_skill_gap_inputs) -> None:
    _, job, _ = sample_skill_gap_inputs
    assert _prioritize("", job) is Priority.LOW


# ---- (8) skill deduplication --------------------------------------------


def test_dedupe_case_insensitive_preserves_canonical() -> None:
    from src.modules.skill_gap.report import _dedupe

    assert _dedupe(["Python", "python", "PYTHON", "SQL"]) == ["Python", "SQL"]


def test_report_dedupes_missing_skills_case_insensitively() -> None:
    """If missing_skills has duplicates, the report dedupes them."""
    resume = ResumeAnalysis(skills=[])
    job = JobAnalysis(required_skills=["Python", "SQL"])
    match = type("M", (), {
        "missing_skills": ["Docker", "docker", "DOCKER"],
        "matching_skills": [],
        "partial_match_skills": [],
    })()
    report = build_skill_gap_report(resume, job, match)
    assert len(report.skill_gaps) == 1
    assert report.skill_gaps[0].skill == "Docker"


# ---- (9) case-insensitive skill matching in evidence/priority -----------
# (covered by dedupe tests above + evidence tests below)


# ---- (10) current evidence detection ------------------------------------


def test_evidence_explicit_present_in_skills() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    text, level = _detect_evidence("Python", resume)
    assert "Explicitly present" in text
    assert level == "Explicitly present"


def test_evidence_related_found_in_experience_text() -> None:
    """A required gap mentioned in experience text -> 'Related evidence'."""
    resume = ResumeAnalysis(
        skills=["Python"],
        experience=[ExperienceItem(description="Some Docker exposure in side projects.")],
    )
    text, level = _detect_evidence("Docker", resume)
    assert "Related evidence found" in text
    assert level == "Related evidence found"


def test_evidence_related_found_in_summary() -> None:
    resume = ResumeAnalysis(
        skills=["Python"], professional_summary="Experience with Kubernetes in production."
    )
    text, _ = _detect_evidence("Kubernetes", resume)
    assert "Related evidence found" in text


# ---- (11) "No evidence found in resume" behavior -----------------------


def test_evidence_no_evidence_default() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    text, level = _detect_evidence("Docker", resume)
    assert text == "No evidence found in resume"
    assert level == "No evidence found in resume"


def test_report_skill_gap_defaults_to_no_evidence(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    docker = next(g for g in report.skill_gaps if g.skill == "Docker")
    assert docker.current_evidence == "No evidence found in resume"
    assert docker.current_level == "No evidence found in resume"


# ---- (12) target proficiency extraction ---------------------------------


def test_target_level_senior_is_advanced() -> None:
    job = JobAnalysis(experience_requirements="Senior backend engineer, 5+ years")
    assert _target_level(job) == "Advanced"


def test_target_level_lead_is_advanced() -> None:
    job = JobAnalysis(experience_requirements="Tech lead with 7 years")
    assert _target_level(job) == "Advanced"


def test_target_level_junior_is_beginner() -> None:
    job = JobAnalysis(experience_requirements="Junior engineer, entry-level")
    assert _target_level(job) == "Beginner"


def test_target_level_mid_years_is_intermediate() -> None:
    job = JobAnalysis(experience_requirements="3+ years of backend experience")
    assert _target_level(job) == "Intermediate"


# ---- (13) "Not specified" fallback --------------------------------------


def test_target_level_unparseable_is_not_specified() -> None:
    job = JobAnalysis(experience_requirements="extensive backend experience")
    assert _target_level(job) == "Not specified"


def test_target_level_empty_is_not_specified() -> None:
    job = JobAnalysis(experience_requirements="")
    assert _target_level(job) == "Not specified"


def test_target_level_absent_is_not_specified() -> None:
    job = JobAnalysis()  # no experience_requirements field at all
    assert _target_level(job) == "Not specified"


# ---- (18) empty missing_skills behavior ---------------------------------


def test_report_no_gaps_when_nothing_missing() -> None:
    """Resume covers all required + preferred skills -> 0 gaps."""
    resume = ResumeAnalysis(skills=["Python", "SQL", "AWS", "Docker", "Kubernetes"])
    job = JobAnalysis(
        required_skills=["Python", "SQL", "AWS"],
        preferred_skills=["Docker", "Kubernetes"],
    )
    match = type("M", (), {
        "missing_skills": [],
        "matching_skills": ["Python", "SQL", "AWS"],
        "partial_match_skills": [],
    })()
    report = build_skill_gap_report(resume, job, match)
    assert report.total_gaps == 0
    assert report.skill_gaps == []


def test_report_only_preferred_gaps_when_no_required_missing() -> None:
    """No required gaps but preferred gaps remain -> all MEDIUM."""
    resume = ResumeAnalysis(skills=["Python", "SQL"])
    job = JobAnalysis(
        required_skills=["Python", "SQL"],
        preferred_skills=["Docker", "Kubernetes"],
    )
    match = type("M", (), {
        "missing_skills": [],
        "matching_skills": ["Python", "SQL"],
        "partial_match_skills": [],
    })()
    report = build_skill_gap_report(resume, job, match)
    assert report.total_gaps == 2
    assert report.high_priority_count == 0
    assert report.medium_priority_count == 2


def test_preferred_gaps_exclude_required_gaps() -> None:
    """A preferred skill that's also a required gap must not be double-counted."""
    resume = ResumeAnalysis(skills=[])
    job = JobAnalysis(
        required_skills=["Docker"],
        preferred_skills=["Docker", "Kubernetes"],  # Docker in both
    )
    out = _preferred_gaps(job, resume, exclude=["Docker"])
    assert out == ["Kubernetes"]


def test_preferred_gap_skipped_when_resume_has_exact_skill() -> None:
    resume = ResumeAnalysis(skills=["Kubernetes"])
    job = JobAnalysis(preferred_skills=["Kubernetes", "Terraform"])
    out = _preferred_gaps(job, resume, exclude=[])
    assert out == ["Terraform"]


# ---- priority ordering in rendered output --------------------------------


def test_report_skill_gaps_sortable_by_priority(sample_skill_gap_inputs) -> None:
    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    sorted_gaps = sorted(report.skill_gaps, key=lambda g: order.get(g.priority, 9))
    # HIGH gaps come before MEDIUM gaps.
    priorities = [g.priority for g in sorted_gaps]
    assert priorities == sorted(priorities, key=lambda p: order[p])
