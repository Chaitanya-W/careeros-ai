"""Tests for the ResumeAnalysis data model (schema validation + robustness)."""

from __future__ import annotations

import json

import pytest

from src.modules.resume_intelligence.model import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    ResumeAnalysis,
    ResumeAnalysisError,
    ResumeAnalysisParseError,
)


# ---- (7) schema validation: full valid payload round-trips -----------------


def test_from_dict_full_payload(sample_analysis_dict: dict) -> None:
    analysis = ResumeAnalysis.from_dict(sample_analysis_dict)
    assert analysis.overall_score == 78
    assert analysis.professional_summary == (
        "Backend engineer with 5 years building distributed systems."
    )
    assert analysis.skills == ["Python", "Go", "PostgreSQL", "Kubernetes"]
    assert len(analysis.experience) == 1
    assert isinstance(analysis.experience[0], ExperienceItem)
    assert analysis.experience[0].title == "Senior Engineer"
    assert analysis.experience[0].company == "Acme Corp"
    assert len(analysis.education) == 1
    assert isinstance(analysis.education[0], EducationItem)
    assert analysis.education[0].institution == "MIT"
    assert analysis.projects == []
    assert analysis.certifications == []
    assert analysis.strengths == ["Strong systems design", "Quantified impact"]
    assert analysis.weaknesses == ["Sparse project detail"]
    assert len(analysis.improvement_suggestions) == 2
    assert analysis.recommended_skills == ["Terraform", "Rust"]


def test_from_json_full_payload(sample_analysis_json: str) -> None:
    analysis = ResumeAnalysis.from_json(sample_analysis_json)
    assert analysis.overall_score == 78
    assert analysis.professional_summary is not None


# ---- (8) missing optional fields ------------------------------------------


def test_from_dict_missing_all_optional_fields() -> None:
    """A payload with only the score must not crash — everything else empty."""
    analysis = ResumeAnalysis.from_dict({"overall_score": 50})
    assert analysis.overall_score == 50
    assert analysis.professional_summary is None
    assert analysis.skills == []
    assert analysis.experience == []
    assert analysis.education == []
    assert analysis.projects == []
    assert analysis.certifications == []
    assert analysis.strengths == []
    assert analysis.weaknesses == []
    assert analysis.improvement_suggestions == []
    assert analysis.recommended_skills == []
    assert analysis.is_empty() is False  # has a score


def test_from_dict_empty_dict_yields_empty_analysis() -> None:
    analysis = ResumeAnalysis.from_dict({})
    assert analysis.overall_score == 0
    assert analysis.professional_summary is None
    assert analysis.is_empty() is True


def test_from_dict_none_summary_becomes_none() -> None:
    analysis = ResumeAnalysis.from_dict({"professional_summary": None})
    assert analysis.professional_summary is None


def test_from_dict_empty_string_summary_becomes_none() -> None:
    analysis = ResumeAnalysis.from_dict({"professional_summary": "   "})
    assert analysis.professional_summary is None


def test_from_dict_drops_empty_and_none_skills() -> None:
    analysis = ResumeAnalysis.from_dict(
        {"skills": ["Python", "", None, "  ", "Go"]}
    )
    assert analysis.skills == ["Python", "Go"]


def test_from_dict_coerces_score_variants_and_clamps() -> None:
    assert ResumeAnalysis.from_dict({"overall_score": "85"}).overall_score == 85
    assert ResumeAnalysis.from_dict({"overall_score": 85.0}).overall_score == 85
    assert ResumeAnalysis.from_dict({"overall_score": "85.7"}).overall_score == 85
    assert ResumeAnalysis.from_dict({"overall_score": -5}).overall_score == 0
    assert ResumeAnalysis.from_dict({"overall_score": 250}).overall_score == 100
    assert ResumeAnalysis.from_dict({"overall_score": "bad"}).overall_score == 0
    assert ResumeAnalysis.from_dict({"overall_score": None}).overall_score == 0


def test_from_dict_tolerates_non_dict_subitems() -> None:
    """Non-dict entries in lists must not crash — they become empty items."""
    analysis = ResumeAnalysis.from_dict(
        {"experience": ["not a dict", None, 42, {"title": "SWE"}]}
    )
    assert len(analysis.experience) == 4
    assert analysis.experience[0].title == ""
    assert analysis.experience[3].title == "SWE"


def test_from_dict_tolerates_scalar_where_list_expected() -> None:
    analysis = ResumeAnalysis.from_dict({"skills": "Python"})
    assert analysis.skills == ["Python"]


def test_from_dict_ignores_unknown_keys() -> None:
    analysis = ResumeAnalysis.from_dict({"overall_score": 60, "unknown": "x"})
    assert analysis.overall_score == 60


def test_to_dict_round_trips() -> None:
    original = ResumeAnalysis(
        overall_score=70,
        professional_summary="A summary.",
        skills=["Python"],
        experience=[ExperienceItem(title="SWE")],
    )
    data = original.to_dict()
    again = ResumeAnalysis.from_dict(data)
    assert again.overall_score == 70
    assert again.skills == ["Python"]
    assert again.experience[0].title == "SWE"


# ---- (10) invalid Gemini JSON --------------------------------------------


def test_from_json_invalid_json_raises() -> None:
    with pytest.raises(ResumeAnalysisParseError):
        ResumeAnalysis.from_json("not json { at all")


def test_from_json_empty_string_raises() -> None:
    with pytest.raises(ResumeAnalysisParseError):
        ResumeAnalysis.from_json("")


def test_from_json_code_fenced_json_is_parsed() -> None:
    payload = "```json\n" + json.dumps({"overall_score": 90}) + "\n```"
    analysis = ResumeAnalysis.from_json(payload)
    assert analysis.overall_score == 90


def test_from_json_non_object_raises() -> None:
    """A JSON array or scalar is not a valid analysis object."""
    with pytest.raises(ResumeAnalysisParseError):
        ResumeAnalysis.from_json(json.dumps(["not", "an", "object"]))
    with pytest.raises(ResumeAnalysisParseError):
        ResumeAnalysis.from_json(json.dumps("just a string"))


def test_resume_analysis_error_is_base() -> None:
    assert issubclass(ResumeAnalysisParseError, ResumeAnalysisError)


# ---- sub-item dataclasses -------------------------------------------------


def test_subitem_dataclasses_default_empty() -> None:
    assert ExperienceItem().title == ""
    assert EducationItem().institution == ""
    assert ProjectItem().name == ""
    assert CertificationItem().issuer == ""


def test_subitem_from_dict_non_dict_returns_empty() -> None:
    assert ExperienceItem.from_dict("nope").title == ""
    assert EducationItem.from_dict(None).degree == ""
    assert ProjectItem.from_dict(42).name == ""
