"""Tests for the Career Roadmap data models (CareerRoadmap, RoadmapPhase, RoadmapMilestone)."""

from __future__ import annotations

import json

import pytest

from src.modules.career_roadmap.model import (
    CareerRoadmap,
    CareerRoadmapError,
    CareerRoadmapParseError,
    RoadmapMilestone,
    RoadmapPhase,
)


# ---- (1)(2)(3) schema validation -----------------------------------------


def test_roadmap_milestone_defaults() -> None:
    ms = RoadmapMilestone()
    assert ms.title == ""
    assert ms.skill == ""
    assert ms.completion_criteria == ""


def test_roadmap_milestone_from_dict_full() -> None:
    ms = RoadmapMilestone.from_dict(
        {
            "title": "Containerize an app",
            "description": "Practice Docker.",
            "skill": "Docker",
            "completion_criteria": "Run the container locally.",
        }
    )
    assert ms.title == "Containerize an app"
    assert ms.skill == "Docker"
    assert ms.completion_criteria == "Run the container locally."


def test_roadmap_milestone_from_dict_non_dict_raises() -> None:
    with pytest.raises(CareerRoadmapParseError):
        RoadmapMilestone.from_dict("nope")  # type: ignore[arg-type]


def test_roadmap_phase_defaults() -> None:
    p = RoadmapPhase()
    assert p.phase_number == 0
    assert p.skills == []
    assert p.prerequisites == []
    assert p.milestones == []
    assert p.practice_project == ""


def test_roadmap_phase_from_dict_full() -> None:
    p = RoadmapPhase.from_dict(
        {
            "phase_number": 1,
            "title": "Phase 1",
            "objective": "Build skills.",
            "duration": "4-8 weeks",
            "skills": ["Docker", "Kubernetes"],
            "prerequisites": [],
            "milestones": [
                {"skill": "Docker", "title": "t", "description": "d", "completion_criteria": "c"},
            ],
            "practice_project": "Containerize an app.",
        }
    )
    assert p.phase_number == 1
    assert p.skills == ["Docker", "Kubernetes"]
    assert len(p.milestones) == 1
    assert isinstance(p.milestones[0], RoadmapMilestone)
    assert p.milestones[0].skill == "Docker"


def test_roadmap_phase_from_dict_coerces_phase_number() -> None:
    p = RoadmapPhase.from_dict({"phase_number": "2"})
    assert p.phase_number == 2


def test_roadmap_phase_from_dict_non_dict_raises() -> None:
    with pytest.raises(CareerRoadmapParseError):
        RoadmapPhase.from_dict(123)  # type: ignore[arg-type]


def test_career_roadmap_defaults() -> None:
    r = CareerRoadmap()
    assert r.target_role == ""
    assert r.current_readiness == 0
    assert r.estimated_total_effort == ""
    assert r.phases == []
    assert r.final_outcome == ""
    assert r.enriched is False
    assert r.phase_count == 0
    assert r.all_skills == []


def test_career_roadmap_from_dict_full() -> None:
    r = CareerRoadmap.from_dict(
        {
            "target_role": "Backend Engineer",
            "current_readiness": 60,
            "estimated_total_effort": "8 weeks",
            "phases": [
                {"phase_number": 1, "skills": ["Docker"]},
                {"phase_number": 2, "skills": ["Kubernetes"]},
            ],
            "final_outcome": "Job-ready.",
            "enriched": True,
        }
    )
    assert r.target_role == "Backend Engineer"
    assert r.current_readiness == 60
    assert len(r.phases) == 2
    assert r.phase_count == 2
    assert r.all_skills == ["Docker", "Kubernetes"]
    assert r.enriched is True


def test_career_roadmap_from_dict_coerces_readiness() -> None:
    r = CareerRoadmap.from_dict({"current_readiness": "75.5"})
    assert r.current_readiness == 75


def test_career_roadmap_from_dict_non_dict_raises() -> None:
    with pytest.raises(CareerRoadmapParseError):
        CareerRoadmap.from_dict([])  # type: ignore[arg-type]


def test_career_roadmap_from_json_code_fenced() -> None:
    payload = "```json\n" + json.dumps({"target_role": "Eng", "current_readiness": 50}) + "\n```"
    r = CareerRoadmap.from_json(payload)
    assert r.target_role == "Eng"
    assert r.current_readiness == 50


def test_career_roadmap_from_json_invalid_raises() -> None:
    with pytest.raises(CareerRoadmapParseError):
        CareerRoadmap.from_json("not json {{{")
    with pytest.raises(CareerRoadmapParseError):
        CareerRoadmap.from_json("")


def test_career_roadmap_to_dict_round_trips() -> None:
    r = CareerRoadmap(
        target_role="Eng",
        current_readiness=40,
        phases=[RoadmapPhase(phase_number=1, skills=["Docker"], milestones=[RoadmapMilestone(skill="Docker")])],
        enriched=True,
    )
    d = r.to_dict()
    again = CareerRoadmap.from_dict(d)
    assert again.target_role == "Eng"
    assert again.current_readiness == 40
    assert again.phase_count == 1
    assert again.all_skills == ["Docker"]
    assert again.enriched is True


def test_error_hierarchy() -> None:
    assert issubclass(CareerRoadmapParseError, CareerRoadmapError)
