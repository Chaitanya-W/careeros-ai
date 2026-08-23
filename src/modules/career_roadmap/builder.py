"""Deterministic Career Roadmap builder.

Produces a :class:`CareerRoadmap` from a :class:`SkillGapReport` (+ the
underlying :class:`MatchAnalysis` and :class:`JobAnalysis`) using **pure
deterministic logic** — no LLM call.

Source of truth
--------------
Every roadmap skill comes EXACTLY from ``SkillGapReport.skill_gaps``
(deduplicated, case-insensitively). No skill is invented, and none beyond
the gap list is added. AI enrichment (separate) only fills wording fields.

Readiness score (deterministic, NOT scientifically validated)
-------------------------------------------------------------
``current_readiness`` reuses the existing
``MatchAnalysis.overall_match_score`` (the weighted skill-coverage match
score from Step 4). No second compatibility score is invented. The value
is rounded and clamped to the documented 0-100 range.

    current_readiness = clamp(round(match.overall_match_score), 0, 100)

Ordering (deterministic)
------------------------
1. Primary: skill priority from the SkillGapReport (HIGH -> MEDIUM -> LOW).
2. Secondary (within a priority): topological sort using a SMALL explicit
   dependency graph (e.g. Kubernetes depends on Docker; FastAPI depends on
   Python). Only edges whose endpoints are BOTH in the current skill set
   are applied — unknown skills get no invented dependencies.
3. Tertiary: alphabetical (stable tiebreak).

This is NOT an enormous ontology — ~25 common relationships. Unknown skills
fall back to priority + alphabetical ordering.

Phase generation
----------------
One phase per non-empty priority group (so there are never empty phases).
Within a phase, skills are topologically ordered. Phase prerequisites are
the skills from earlier phases.
"""

from __future__ import annotations

import heapq
import re
from typing import Optional

from src.modules.career_roadmap.model import (
    CareerRoadmap,
    RoadmapMilestone,
    RoadmapPhase,
)
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.skill_gap.model import Priority, SkillGapReport

# Rough effort estimate per skill (weeks of part-time study). Tunable, not
# scientifically derived.
_WEEKS_PER_SKILL_LOW: int = 2
_WEEKS_PER_SKILL_HIGH: int = 4

# Small, explicit dependency graph: normalized skill -> list of normalized
# prerequisite skills. Kept small on purpose — NOT an enormous ontology.
# For skills not listed here, no dependencies are invented.
DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "python": [],
    "sql": [],
    "git": [],
    "javascript": [],
    "java": [],
    "go": [],
    "rust": [],
    "c++": [],
    "docker": [],
    "kubernetes": ["docker"],
    "terraform": [],
    "aws": [],
    "gcp": [],
    "azure": [],
    "fastapi": ["python"],
    "flask": ["python"],
    "django": ["python"],
    "pandas": ["python"],
    "numpy": ["python"],
    "tensorflow": ["python"],
    "pytorch": ["python"],
    "scikit-learn": ["python"],
    "spark": ["python"],
    "airflow": ["python"],
    "postgresql": ["sql"],
    "mysql": ["sql"],
    "typescript": ["javascript"],
    "react": ["javascript"],
    "node": ["javascript"],
    "express": ["javascript"],
}


def build_career_roadmap(
    skill_gap_report: SkillGapReport,
    match,
    job: JobAnalysis,
) -> CareerRoadmap:
    """Build a deterministic :class:`CareerRoadmap`.

    Skills come EXCLUSIVELY from ``skill_gap_report.skill_gaps``. Readiness
    comes from ``match.overall_match_score``. No AI.
    """
    target_role = (skill_gap_report.target_role or _target_role(job))
    readiness = compute_readiness(match)

    if not skill_gap_report.skill_gaps:
        return CareerRoadmap(
            target_role=target_role,
            current_readiness=readiness,
            estimated_total_effort="No skill gaps to address.",
            phases=[],
            final_outcome=(
                "Your resume currently covers the identified requirements. "
                "No major skill gaps were detected for this target role."
            ),
            enriched=False,
        )

    gaps = _dedupe_gaps(skill_gap_report.skill_gaps)
    phases = _build_phases(gaps, target_role)
    total_skills = sum(len(p.skills) for p in phases)

    return CareerRoadmap(
        target_role=target_role,
        current_readiness=readiness,
        estimated_total_effort=_total_effort(total_skills),
        phases=phases,
        final_outcome=_final_outcome(target_role),
        enriched=False,
    )


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


def compute_readiness(match) -> int:
    """Deterministic readiness 0-100, reused from MatchAnalysis.

    Reuses ``match.overall_match_score`` (the weighted skill-coverage match
    score from Step 4) — no second compatibility score is invented.
    """
    score = getattr(match, "overall_match_score", 0) or 0
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def order_skills(gaps) -> list[str]:
    """Order skills: priority (HIGH->MEDIUM->LOW), then topo within priority,
    then alphabetical. Deterministic."""
    by_priority: dict[Priority, list[str]] = {
        Priority.HIGH: [],
        Priority.MEDIUM: [],
        Priority.LOW: [],
    }
    for g in gaps:
        by_priority.setdefault(g.priority, []).append(g.skill)
    out: list[str] = []
    for pri in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
        out.extend(_topo_sort(by_priority.get(pri, [])))
    return out


def _topo_sort(skills: list[str]) -> list[str]:
    """Topological sort within a skill set using DEPENDENCY_GRAPH.

    Only edges whose endpoints are BOTH in the set are applied. Alphabetical
    tiebreak via a heap. Cycles (should not occur) fall back to alphabetical.
    """
    if not skills:
        return []
    norm_to_orig: dict[str, str] = {}
    for s in skills:
        n = _normalize(s)
        if n and n not in norm_to_orig:
            norm_to_orig[n] = s  # first canonical wins
    nodes = sorted(norm_to_orig.keys())
    if not nodes:
        return []

    in_deg: dict[str, int] = {n: 0 for n in nodes}
    edges: dict[str, list[str]] = {n: [] for n in nodes}
    for n in nodes:
        for prereq in DEPENDENCY_GRAPH.get(n, []):
            if prereq in norm_to_orig:  # in-set edge only
                edges[prereq].append(n)
                in_deg[n] += 1

    queue = [n for n in nodes if in_deg[n] == 0]
    heapq.heapify(queue)
    out_norm: list[str] = []
    while queue:
        n = heapq.heappop(queue)
        out_norm.append(n)
        for dep in sorted(edges[n]):
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                heapq.heappush(queue, dep)

    if len(out_norm) != len(nodes):
        # Cycle (shouldn't happen with the curated DAG) — fallback alphabetical.
        out_norm = nodes
    return [norm_to_orig[n] for n in out_norm]


# --------------------------------------------------------------------------- #
# Phase / milestone / project generation
# --------------------------------------------------------------------------- #


def _build_phases(gaps: list, target_role: str) -> list[RoadmapPhase]:
    """One phase per non-empty priority group; skills topo-ordered within."""
    by_priority: dict[Priority, list] = {
        Priority.HIGH: [],
        Priority.MEDIUM: [],
        Priority.LOW: [],
    }
    for g in gaps:
        by_priority.setdefault(g.priority, []).append(g)

    phases: list[RoadmapPhase] = []
    cumulative: list[str] = []
    phase_num = 0
    for pri in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
        group = by_priority.get(pri, [])
        if not group:
            continue  # no empty phases
        phase_num += 1
        skills = _topo_sort([g.skill for g in group])
        milestones = [_build_milestone(s, target_role) for s in skills]
        phases.append(
            RoadmapPhase(
                phase_number=phase_num,
                title=f"Phase {phase_num}: {_priority_descriptor(pri)}",
                objective=_phase_objective(skills, target_role),
                duration=_phase_duration(skills),
                skills=skills,
                prerequisites=list(cumulative),
                milestones=milestones,
                practice_project=_practice_project(skills, target_role),
            )
        )
        cumulative.extend(skills)
    return phases


def _priority_descriptor(priority: Priority) -> str:
    if priority is Priority.HIGH:
        return "Core required skills"
    if priority is Priority.MEDIUM:
        return "Recommended skills"
    return "Additional gaps"


def _phase_objective(skills: list[str], target_role: str) -> str:
    return (
        f"Build proficiency in {', '.join(skills)} for the {target_role} role."
    )


def _phase_duration(skills: list[str]) -> str:
    n = len(skills)
    return f"{n * _WEEKS_PER_SKILL_LOW}-{n * _WEEKS_PER_SKILL_HIGH} weeks"


def _build_milestone(skill: str, target_role: str) -> RoadmapMilestone:
    return RoadmapMilestone(
        title=f"Apply {skill} in context",
        description=(
            f"Practice {skill} through a focused exercise relevant to the "
            f"{target_role} role."
        ),
        skill=skill,
        completion_criteria=(
            f"Demonstrate a working understanding of {skill} by completing "
            f"a small exercise or project."
        ),
    )


def _practice_project(skills: list[str], target_role: str) -> str:
    return (
        f"Build a small {target_role}-focused project that applies "
        f"{', '.join(skills)}."
    )


def _total_effort(total_skills: int) -> str:
    return (
        f"{total_skills * _WEEKS_PER_SKILL_LOW}-"
        f"{total_skills * _WEEKS_PER_SKILL_HIGH} weeks of part-time study"
    )


def _final_outcome(target_role: str) -> str:
    return (
        f"Reach job-ready alignment for the {target_role} role by closing "
        f"the identified skill gaps."
    )


def _target_role(job: JobAnalysis) -> str:
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dedupe_gaps(gaps) -> list:
    """Deduplicate gaps by normalized skill name (case-insensitive)."""
    seen: set[str] = set()
    out = []
    for g in gaps:
        n = _normalize(g.skill)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(g)
    return out


def _normalize(skill: str) -> str:
    if not skill:
        return ""
    s = skill.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b\d+(\.\d+)*\+?\b", "", s)
    s = re.sub(r"[^a-z0-9+#./\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
