"""Tests for the DETERMINISTIC Career Roadmap builder (no LLM)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.modules.career_roadmap.builder import (
    DEPENDENCY_GRAPH,
    build_career_roadmap,
    compute_readiness,
    order_skills,
)
from src.modules.career_roadmap.model import CareerRoadmap, RoadmapPhase
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import Priority, SkillGap, SkillGapReport


def _gap(skill: str, priority: Priority = Priority.HIGH) -> SkillGap:
    return SkillGap(skill=skill, priority=priority)


def _match(score: int, missing: list[str]) -> SimpleNamespace:
    return SimpleNamespace(overall_match_score=score, missing_skills=missing)


# ---- (4)(5) readiness score bounds + calculation ------------------------


def test_readiness_reuses_match_score() -> None:
    """readiness must equal match.overall_match_score (no second score)."""
    assert compute_readiness(_match(73, ["Docker"])) == 73


def test_readiness_clamps_to_0_100() -> None:
    assert compute_readiness(_match(-5, [])) == 0
    assert compute_readiness(_match(150, [])) == 100


def test_readiness_handles_missing_score_attr() -> None:
    m = SimpleNamespace()  # no overall_match_score
    assert compute_readiness(m) == 0


def test_readiness_handles_non_numeric() -> None:
    m = SimpleNamespace(overall_match_score="bad")
    assert compute_readiness(m) == 0


# ---- (9) skill source-of-truth validation -------------------------------


def test_roadmap_skills_come_exclusively_from_skill_gap_report(
    sample_skill_gap_report, sample_skill_gap_inputs
) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    report_skills = {g.skill.lower() for g in sample_skill_gap_report.skill_gaps}
    roadmap_skills = {s.lower() for s in roadmap.all_skills}
    assert roadmap_skills == report_skills, "roadmap invented or dropped skills"


def test_roadmap_does_not_invent_skills() -> None:
    """A report with only Docker/AWS must not produce Kubernetes/Terraform."""
    report = SkillGapReport(
        target_role="Eng",
        skill_gaps=[_gap("Docker"), _gap("AWS")],
    )
    match = _match(50, ["Docker", "AWS"])
    job = JobAnalysis(job_title="Eng", required_skills=["Docker", "AWS"])
    roadmap = build_career_roadmap(report, match, job)
    assert set(roadmap.all_skills) == {"Docker", "AWS"}
    assert "Kubernetes" not in roadmap.all_skills
    assert "Terraform" not in roadmap.all_skills


# ---- (10) duplicate skill removal ---------------------------------------


def test_roadmap_deduplicates_skills_case_insensitively() -> None:
    report = SkillGapReport(
        target_role="Eng",
        skill_gaps=[_gap("Docker"), _gap("docker"), _gap("DOCKER"), _gap("AWS")],
    )
    roadmap = build_career_roadmap(report, _match(10, ["Docker"]), JobAnalysis())
    assert sorted(s.lower() for s in roadmap.all_skills) == ["aws", "docker"]
    # Exactly one milestone per unique skill.
    total_milestones = sum(len(p.milestones) for p in roadmap.phases)
    assert total_milestones == 2


# ---- (11) deterministic priority ordering -------------------------------


def test_order_skills_high_before_medium_before_low() -> None:
    gaps = [
        _gap("Terraform", Priority.LOW),
        _gap("Kubernetes", Priority.MEDIUM),
        _gap("Docker", Priority.HIGH),
    ]
    ordered = order_skills(gaps)
    # HIGH (Docker) first, then MEDIUM (Kubernetes), then LOW (Terraform).
    assert ordered[0] == "Docker"
    assert ordered[1] == "Kubernetes"
    assert ordered[2] == "Terraform"


def test_roadmap_high_priority_phase_first(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    # Phase 1 must contain HIGH-priority gaps (AWS, Docker).
    phase1_skills = {s.lower() for s in roadmap.phases[0].skills}
    assert {"aws", "docker"}.issubset(phase1_skills)


# ---- (12) prerequisite ordering ----------------------------------------


def test_order_skills_respects_dependency_graph() -> None:
    """Kubernetes depends on Docker -> Docker before Kubernetes (same priority)."""
    gaps = [_gap("Kubernetes"), _gap("Docker")]  # both HIGH
    ordered = order_skills(gaps)
    assert ordered.index("Docker") < ordered.index("Kubernetes")


def test_order_skills_dependency_only_within_set() -> None:
    """If the prerequisite is NOT in the set, no edge is applied."""
    # Kubernetes depends on Docker, but Docker isn't here -> alphabetical.
    gaps = [_gap("Kubernetes"), _gap("AWS")]
    ordered = order_skills(gaps)
    assert ordered == ["AWS", "Kubernetes"]  # alphabetical


# ---- (13) unknown-skill fallback ordering -------------------------------


def test_order_skills_unknown_skills_alphabetical() -> None:
    gaps = [_gap("Zebra"), _gap("Antelope")]  # neither in DEPENDENCY_GRAPH
    ordered = order_skills(gaps)
    assert ordered == ["Antelope", "Zebra"]


def test_dependency_graph_is_small_curated() -> None:
    """Sanity: the graph is NOT an enormous ontology."""
    assert 10 < len(DEPENDENCY_GRAPH) < 100


# ---- (14) phase generation ---------------------------------------------


def test_roadmap_generates_phases(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert roadmap.phase_count >= 1
    assert isinstance(roadmap.phases[0], RoadmapPhase)


def test_roadmap_phase_numbers_are_sequential(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert [p.phase_number for p in roadmap.phases] == list(range(1, roadmap.phase_count + 1))


def test_roadmap_phase_titles_derived_from_priority(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    # Phase 1 (HIGH) title mentions "Core required skills".
    assert "Core required skills" in roadmap.phases[0].title


# ---- (15) no empty phases ----------------------------------------------


def test_roadmap_has_no_empty_phases() -> None:
    """A report with only MEDIUM gaps -> one MEDIUM phase, no empty HIGH phase."""
    report = SkillGapReport(
        target_role="Eng",
        skill_gaps=[_gap("Kubernetes", Priority.MEDIUM)],
    )
    roadmap = build_career_roadmap(report, _match(50, []), JobAnalysis())
    assert roadmap.phase_count == 1
    assert len(roadmap.phases[0].skills) == 1


def test_roadmap_skips_empty_priority_groups() -> None:
    report = SkillGapReport(
        target_role="Eng",
        skill_gaps=[_gap("Docker", Priority.HIGH), _gap("Terraform", Priority.LOW)],
    )
    roadmap = build_career_roadmap(report, _match(50, ["Docker"]), JobAnalysis())
    # HIGH phase + LOW phase; no empty MEDIUM phase.
    assert roadmap.phase_count == 2
    assert all(len(p.skills) > 0 for p in roadmap.phases)


# ---- (8) empty skill gaps ----------------------------------------------


def test_roadmap_empty_gaps_returns_no_phases() -> None:
    report = SkillGapReport(target_role="Eng", skill_gaps=[])
    roadmap = build_career_roadmap(report, _match(90, []), JobAnalysis())
    assert roadmap.phase_count == 0
    assert roadmap.phases == []
    assert "No major skill gaps" in roadmap.final_outcome


# ---- (16) milestone generation -----------------------------------------


def test_roadmap_milestone_per_skill(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    for phase in roadmap.phases:
        assert len(phase.milestones) == len(phase.skills)
        for ms, skill in zip(phase.milestones, phase.skills):
            assert ms.skill == skill
            assert ms.title
            assert ms.completion_criteria


def test_roadmap_milestones_are_skill_specific() -> None:
    report = SkillGapReport(target_role="Eng", skill_gaps=[_gap("Docker")])
    roadmap = build_career_roadmap(report, _match(20, ["Docker"]), JobAnalysis())
    ms = roadmap.phases[0].milestones[0]
    assert "Docker" in ms.title
    assert "Docker" in ms.completion_criteria


# ---- (17) project generation -----------------------------------------


def test_roadmap_phase_has_practice_project(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    for phase in roadmap.phases:
        assert phase.practice_project
        # Project references the target role + skills.
        assert any(s in phase.practice_project for s in phase.skills)


# ---- prerequisites propagation ---------------------------------------


def test_roadmap_phase_prerequisites_are_earlier_phase_skills(
    sample_skill_gap_report, sample_skill_gap_inputs
) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    if roadmap.phase_count >= 2:
        earlier_skills = set(roadmap.phases[0].skills)
        later_prereqs = set(roadmap.phases[1].prerequisites)
        assert earlier_skills.issubset(later_prereqs)


def test_roadmap_first_phase_has_no_prerequisites(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert roadmap.phases[0].prerequisites == []


# ---- summary fields ---------------------------------------------------


def test_roadmap_readiness_matches_match_score(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert roadmap.current_readiness == match.overall_match_score


def test_roadmap_total_effort_and_outcome(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert roadmap.estimated_total_effort
    assert roadmap.final_outcome
    assert roadmap.target_role


def test_roadmap_not_enriched_by_default(sample_skill_gap_report, sample_skill_gap_inputs) -> None:
    _resume, job, match = sample_skill_gap_inputs
    roadmap = build_career_roadmap(sample_skill_gap_report, match, job)
    assert roadmap.enriched is False
