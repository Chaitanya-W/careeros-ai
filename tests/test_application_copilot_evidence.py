"""Tests for the deterministic evidence extraction (stable IDs)."""

from __future__ import annotations

from src.modules.application_copilot.evidence import (
    evidence_corpus,
    evidence_ids_for_skills,
    extract_evidence,
)
from src.modules.resume_intelligence.model import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    ResumeAnalysis,
)


# ---- (4) evidence extraction ------------------------------------------


def test_extract_evidence_returns_items(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    assert items
    assert all(i.evidence_id for i in items)
    assert all(i.source_type for i in items)


# ---- (5) stable evidence IDs -----------------------------------------


def test_stable_ids_are_deterministic(sample_copilot_resume) -> None:
    a = extract_evidence(sample_copilot_resume)
    b = extract_evidence(sample_copilot_resume)
    assert [i.evidence_id for i in a] == [i.evidence_id for i in b]


def test_ids_follow_prefix_conventions(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    ids = [i.evidence_id for i in items]
    assert "EXP-001" in ids
    assert "PROJ-001" in ids
    assert "SKILL-001" in ids
    assert "EDU-001" in ids
    assert "CERT-001" in ids
    assert "SUMMARY-001" in ids


def test_ids_are_zero_padded_and_sequential(sample_copilot_resume) -> None:
    resume = ResumeAnalysis(
        skills=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
    )
    items = extract_evidence(resume)
    skill_ids = [i.evidence_id for i in items if i.source_type == "skill"]
    assert skill_ids == ["SKILL-001", "SKILL-002", "SKILL-003", "SKILL-004", "SKILL-005",
                         "SKILL-006", "SKILL-007", "SKILL-008", "SKILL-009", "SKILL-010", "SKILL-011"]


# ---- (6) experience evidence -----------------------------------------


def test_experience_evidence(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    exp = [i for i in items if i.source_type == "experience"]
    assert len(exp) == 1
    assert exp[0].evidence_id == "EXP-001"
    assert "Software Engineer" in exp[0].source_text
    assert "Acme" in exp[0].source_text
    assert "Python services on AWS" in exp[0].source_text


def test_experience_evidence_multiple_entries() -> None:
    resume = ResumeAnalysis(
        experience=[
            ExperienceItem(title="A", company="X", duration="2020-2021", description="d1"),
            ExperienceItem(title="B", company="Y", duration="2021-present", description="d2"),
        ]
    )
    items = extract_evidence(resume)
    exp_ids = [i.evidence_id for i in items if i.source_type == "experience"]
    assert exp_ids == ["EXP-001", "EXP-002"]


# ---- (7) project evidence -------------------------------------------


def test_project_evidence(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    proj = [i for i in items if i.source_type == "project"]
    assert len(proj) == 1
    assert proj[0].evidence_id == "PROJ-001"
    assert "CareerOS" in proj[0].source_text
    assert "Streamlit" in proj[0].source_text


# ---- (8) skill evidence ---------------------------------------------


def test_skill_evidence(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    skills = [i for i in items if i.source_type == "skill"]
    assert len(skills) == 3
    assert skills[0].source_text == "Python"
    assert skills[0].evidence_id == "SKILL-001"


# ---- (9) education evidence -----------------------------------------


def test_education_evidence(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    edu = [i for i in items if i.source_type == "education"]
    assert len(edu) == 1
    assert edu[0].evidence_id == "EDU-001"
    assert "Computer Science" in edu[0].source_text
    assert "MIT" in edu[0].source_text


# ---- (10) certification evidence when available --------------------


def test_certification_evidence_present(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    cert = [i for i in items if i.source_type == "certification"]
    assert len(cert) == 1
    assert cert[0].evidence_id == "CERT-001"
    assert "AWS Certified Developer" in cert[0].source_text


def test_certification_evidence_absent_when_none() -> None:
    """If the resume has no certifications, no CERT-* evidence is invented."""
    resume = ResumeAnalysis(skills=["Python"])
    items = extract_evidence(resume)
    assert not any(i.source_type == "certification" for i in items)


def test_summary_evidence_present(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    summary = [i for i in items if i.source_type == "summary"]
    assert len(summary) == 1
    assert summary[0].evidence_id == "SUMMARY-001"


def test_no_evidence_for_empty_resume() -> None:
    assert extract_evidence(ResumeAnalysis()) == []


# ---- corpus + skill-id helpers ---------------------------------------


def test_evidence_corpus_contains_all_source_text(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    corpus = evidence_corpus(items)
    assert "python" in corpus
    assert "acme" in corpus
    assert "careeros" in corpus


def test_evidence_ids_for_skills(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    ids = evidence_ids_for_skills(["Python", "AWS"], items)
    assert ids == ["SKILL-001", "SKILL-003"]


def test_evidence_ids_for_skills_case_insensitive(sample_copilot_resume) -> None:
    items = extract_evidence(sample_copilot_resume)
    ids = evidence_ids_for_skills(["python", "Sql"], items)
    assert ids == ["SKILL-001", "SKILL-002"]
