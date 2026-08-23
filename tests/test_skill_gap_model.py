"""Tests for the Skill Gap data models (Priority, SkillGap, SkillGapReport)."""

from __future__ import annotations

import pytest

from src.modules.skill_gap.model import (
    Priority,
    SkillGap,
    SkillGapError,
    SkillGapParseError,
    SkillGapReport,
)


# ---- (3) Priority enum validation ----------------------------------------


def test_priority_has_three_members() -> None:
    assert {p.value for p in Priority} == {"high", "medium", "low"}


def test_priority_from_value_known() -> None:
    assert Priority.from_value("high") is Priority.HIGH
    assert Priority.from_value("MEDIUM") is Priority.MEDIUM
    assert Priority.from_value(Priority.LOW) is Priority.LOW


def test_priority_from_value_unknown_defaults_low() -> None:
    assert Priority.from_value("critical") is Priority.LOW
    assert Priority.from_value(None) is Priority.LOW
    assert Priority.from_value(42) is Priority.LOW


def test_priority_is_str_enum() -> None:
    assert Priority.HIGH == "high"
    assert isinstance(Priority.HIGH, str)


# ---- (1) SkillGap model validation ---------------------------------------


def test_skillgap_defaults() -> None:
    g = SkillGap()
    assert g.skill == ""
    assert g.priority is Priority.LOW
    assert g.current_evidence == "No evidence found in resume"
    assert g.target_level == "Not specified"
    assert g.why_it_matters == ""
    assert g.learning_objectives == []
    assert g.learning_path == []
    assert g.practice_project == ""
    assert g.estimated_effort == ""


def test_skillgap_from_dict_full() -> None:
    g = SkillGap.from_dict(
        {
            "skill": "Docker",
            "priority": "high",
            "current_level": "No evidence found",
            "target_level": "Intermediate",
            "current_evidence": "No evidence found in resume",
            "gap_reason": "required but missing",
            "why_it_matters": "Containers matter.",
            "learning_objectives": ["a", "b"],
            "learning_path": ["step 1", "step 2"],
            "practice_project": "Dockerize an app",
            "estimated_effort": "2 weeks",
        }
    )
    assert g.skill == "Docker"
    assert g.priority is Priority.HIGH
    assert g.target_level == "Intermediate"
    assert g.why_it_matters == "Containers matter."
    assert g.learning_objectives == ["a", "b"]
    assert g.estimated_effort == "2 weeks"


# ---- (4) missing optional fields ----------------------------------------


def test_skillgap_from_dict_missing_optional_fields() -> None:
    g = SkillGap.from_dict({"skill": "Docker"})
    assert g.skill == "Docker"
    assert g.priority is Priority.LOW  # default
    assert g.current_evidence == "No evidence found in resume"
    assert g.target_level == "Not specified"
    assert g.why_it_matters == ""
    assert g.learning_objectives == []


def test_skillgap_from_dict_invalid_priority_defaults_low() -> None:
    g = SkillGap.from_dict({"skill": "X", "priority": "critical"})
    assert g.priority is Priority.LOW


def test_skillgap_from_dict_empty_target_falls_back() -> None:
    g = SkillGap.from_dict({"skill": "X", "target_level": "  "})
    assert g.target_level == "Not specified"


def test_skillgap_from_dict_non_dict_raises() -> None:
    with pytest.raises(SkillGapParseError):
        SkillGap.from_dict("not a dict")  # type: ignore[arg-type]


def test_skillgap_to_dict_round_trips() -> None:
    g = SkillGap(skill="Docker", priority=Priority.HIGH, why_it_matters="x")
    d = g.to_dict()
    assert d["skill"] == "Docker"
    assert d["priority"] == "high"  # serialized as the string value
    again = SkillGap.from_dict(d)
    assert again.skill == "Docker"
    assert again.priority is Priority.HIGH


# ---- (2) SkillGapReport validation ---------------------------------------


def test_skillgap_report_defaults() -> None:
    r = SkillGapReport()
    assert r.target_role == ""
    assert r.total_gaps == 0
    assert r.high_priority_count == 0
    assert r.medium_priority_count == 0
    assert r.low_priority_count == 0
    assert r.skill_gaps == []
    assert r.summary == ""
    assert r.generated_at == ""
    assert r.enriched is False


def test_skillgap_report_from_dict_with_gaps() -> None:
    r = SkillGapReport.from_dict(
        {
            "target_role": "Backend Engineer",
            "total_gaps": 2,
            "high_priority_count": 1,
            "medium_priority_count": 1,
            "low_priority_count": 0,
            "skill_gaps": [
                {"skill": "Docker", "priority": "high"},
                {"skill": "Kubernetes", "priority": "medium"},
            ],
            "summary": "2 gaps.",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "enriched": True,
        }
    )
    assert r.target_role == "Backend Engineer"
    assert r.total_gaps == 2
    assert len(r.skill_gaps) == 2
    assert isinstance(r.skill_gaps[0], SkillGap)
    assert r.skill_gaps[0].priority is Priority.HIGH
    assert r.enriched is True


def test_skillgap_report_from_dict_coerces_counts() -> None:
    r = SkillGapReport.from_dict({"total_gaps": "3", "high_priority_count": None})
    assert r.total_gaps == 3
    assert r.high_priority_count == 0


def test_skillgap_report_from_dict_non_dict_raises() -> None:
    with pytest.raises(SkillGapParseError):
        SkillGapReport.from_dict("nope")  # type: ignore[arg-type]


def test_skillgap_report_from_dict_missing_gaps_defaults_empty() -> None:
    r = SkillGapReport.from_dict({"target_role": "Eng"})
    assert r.skill_gaps == []
    assert r.target_role == "Eng"


def test_skillgap_report_to_dict_round_trips() -> None:
    r = SkillGapReport(
        target_role="Eng",
        total_gaps=1,
        high_priority_count=1,
        skill_gaps=[SkillGap(skill="Docker", priority=Priority.HIGH)],
        enriched=True,
    )
    d = r.to_dict()
    again = SkillGapReport.from_dict(d)
    assert again.target_role == "Eng"
    assert again.total_gaps == 1
    assert again.enriched is True
    assert again.skill_gaps[0].skill == "Docker"


def test_skillgap_error_hierarchy() -> None:
    assert issubclass(SkillGapParseError, SkillGapError)
