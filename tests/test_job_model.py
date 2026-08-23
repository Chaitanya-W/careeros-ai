"""Tests for the JobAnalysis + MatchAnalysis data models."""

from __future__ import annotations

import json

import pytest

from src.modules.jd_analysis.model import (
    JobAnalysis,
    JobAnalysisError,
    JobAnalysisParseError,
    MatchAnalysis,
    MatchAnalysisError,
)


# ---- (1) JobAnalysis schema validation: full payload round-trips --------


def test_job_analysis_from_dict_full(sample_job_analysis_dict: dict) -> None:
    job = JobAnalysis.from_dict(sample_job_analysis_dict)
    assert job.job_title == "Senior Backend Engineer"
    assert job.company == "Acme Corp"
    assert job.required_skills == ["Python", "SQL", "AWS", "Docker"]
    assert job.preferred_skills == ["Kubernetes", "Terraform"]
    assert len(job.responsibilities) == 2
    assert job.experience_requirements == "5+ years of backend engineering"
    assert job.education_requirements == "B.S. in Computer Science or equivalent"
    assert "distributed systems" in job.keywords
    assert job.domain == "Backend Engineering"
    assert job.has_any_data() is True


def test_job_analysis_from_json_full(sample_job_analysis_json: str) -> None:
    job = JobAnalysis.from_json(sample_job_analysis_json)
    assert job.company == "Acme Corp"
    assert "AWS" in job.required_skills


# ---- (2) missing optional JobAnalysis fields -----------------------------


def test_job_analysis_missing_all_fields_yields_empty() -> None:
    job = JobAnalysis.from_dict({})
    assert job.job_title == ""
    assert job.company == ""
    assert job.required_skills == []
    assert job.preferred_skills == []
    assert job.responsibilities == []
    assert job.experience_requirements == ""
    assert job.education_requirements == ""
    assert job.keywords == []
    assert job.domain == ""
    assert job.has_any_data() is False


def test_job_analysis_drops_empty_skills() -> None:
    job = JobAnalysis.from_dict(
        {"required_skills": ["Python", "", None, "  ", "SQL"]}
    )
    assert job.required_skills == ["Python", "SQL"]


def test_job_analysis_tolerates_scalar_where_list_expected() -> None:
    job = JobAnalysis.from_dict({"keywords": "Python"})
    assert job.keywords == ["Python"]


def test_job_analysis_tolerates_non_dict_subitems_in_lists() -> None:
    # Non-dict items are coerced to strings and kept; None is dropped
    # (consistent with the str-list coercion contract).
    job = JobAnalysis.from_dict(
        {"responsibilities": ["build", None, 42, {"x": 1}]}
    )
    assert job.responsibilities == ["build", "42", "{'x': 1}"]


def test_job_analysis_ignores_unknown_keys() -> None:
    job = JobAnalysis.from_dict({"job_title": "Eng", "unknown": "x"})
    assert job.job_title == "Eng"


def test_job_analysis_strips_whitespace_on_strings() -> None:
    job = JobAnalysis.from_dict({"job_title": "  Engineer  "})
    assert job.job_title == "Engineer"


# ---- (15) invalid Gemini JSON ------------------------------------------


def test_job_analysis_from_json_invalid_json_raises() -> None:
    with pytest.raises(JobAnalysisParseError):
        JobAnalysis.from_json("not json { at all")


def test_job_analysis_from_json_empty_raises() -> None:
    with pytest.raises(JobAnalysisParseError):
        JobAnalysis.from_json("")


def test_job_analysis_from_json_code_fenced_is_parsed() -> None:
    payload = "```json\n" + json.dumps({"job_title": "Eng"}) + "\n```"
    job = JobAnalysis.from_json(payload)
    assert job.job_title == "Eng"


def test_job_analysis_from_json_non_object_raises() -> None:
    with pytest.raises(JobAnalysisParseError):
        JobAnalysis.from_json(json.dumps(["not", "an", "object"]))
    with pytest.raises(JobAnalysisParseError):
        JobAnalysis.from_json(json.dumps("a string"))


def test_job_analysis_from_dict_non_dict_raises() -> None:
    with pytest.raises(JobAnalysisParseError):
        JobAnalysis.from_dict(["not", "a", "dict"])


def test_job_analysis_error_is_base() -> None:
    assert issubclass(JobAnalysisParseError, JobAnalysisError)


# ---- (9) MatchAnalysis schema + (10) score boundaries -------------------


def test_match_analysis_defaults_are_empty() -> None:
    m = MatchAnalysis()
    assert m.overall_match_score == 0
    assert m.matching_skills == []
    assert m.missing_skills == []
    assert m.partial_match_skills == []
    assert m.experience_match == 0
    assert m.education_match == 0
    assert m.keyword_match == 0
    assert m.strengths == []
    assert m.gaps == []
    assert m.explanation == ""
    assert m.recommended_actions == []


def test_match_analysis_from_dict_clamps_scores() -> None:
    m = MatchAnalysis.from_dict(
        {
            "overall_match_score": 150,  # clamps to 100
            "experience_match": -10,  # clamps to 0
            "education_match": "75",  # coerced
            "keyword_match": "bad",  # -> 0
            "matching_skills": ["Python"],
        }
    )
    assert m.overall_match_score == 100
    assert m.experience_match == 0
    assert m.education_match == 75
    assert m.keyword_match == 0
    assert m.matching_skills == ["Python"]


def test_match_analysis_from_dict_non_dict_raises() -> None:
    with pytest.raises(MatchAnalysisError):
        MatchAnalysis.from_dict("not a dict")


def test_match_analysis_to_dict_round_trips() -> None:
    original = MatchAnalysis(
        overall_match_score=70,
        matching_skills=["Python"],
        missing_skills=["AWS"],
        experience_match=80,
    )
    data = original.to_dict()
    again = MatchAnalysis.from_dict(data)
    assert again.overall_match_score == 70
    assert again.matching_skills == ["Python"]
    assert again.missing_skills == ["AWS"]
    assert again.experience_match == 80
