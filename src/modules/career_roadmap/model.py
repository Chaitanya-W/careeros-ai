"""Career Roadmap data models.

Defines :class:`RoadmapMilestone`, :class:`RoadmapPhase`, and
:class:`CareerRoadmap`. All are robust to missing information —
:meth:`from_dict` tolerates absent keys, wrong types, and extra keys.

Design notes
------------
- Every roadmap skill MUST originate from ``SkillGapReport.skill_gaps``;
  the model never invents skills.
- ``current_readiness`` is bounded 0-100 and is computed deterministically
  from :class:`MatchAnalysis` (no second compatibility score).
- AI-enriched fields default to empty so the roadmap stays useful without
  a Gemini key.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class CareerRoadmapError(Exception):
    """Base error for career-roadmap model / parsing problems."""


class CareerRoadmapParseError(CareerRoadmapError):
    """Raised when a model response cannot be parsed into the schema."""


@dataclass
class RoadmapMilestone:
    """A single milestone targeting one roadmap skill."""

    title: str = ""
    description: str = ""
    skill: str = ""
    completion_criteria: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "RoadmapMilestone":
        if not isinstance(data, dict):
            raise CareerRoadmapParseError("Milestone was not a JSON object.")
        return cls(
            title=_clean_str(data.get("title")),
            description=_clean_str(data.get("description")),
            skill=_clean_str(data.get("skill")),
            completion_criteria=_clean_str(data.get("completion_criteria")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoadmapPhase:
    """A phase of the roadmap — a group of skills learned together."""

    phase_number: int = 0
    title: str = ""
    objective: str = ""
    duration: str = ""
    skills: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    milestones: list[RoadmapMilestone] = field(default_factory=list)
    practice_project: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "RoadmapPhase":
        if not isinstance(data, dict):
            raise CareerRoadmapParseError("Phase was not a JSON object.")
        ms_raw = data.get("milestones")
        milestones = (
            [RoadmapMilestone.from_dict(m) for m in _as_list(ms_raw)]
            if ms_raw is not None
            else []
        )
        return cls(
            phase_number=_coerce_int(data.get("phase_number")),
            title=_clean_str(data.get("title")),
            objective=_clean_str(data.get("objective")),
            duration=_clean_str(data.get("duration")),
            skills=_coerce_str_list(data.get("skills")),
            prerequisites=_coerce_str_list(data.get("prerequisites")),
            milestones=milestones,
            practice_project=_clean_str(data.get("practice_project")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["milestones"] = [m.to_dict() for m in self.milestones]
        return d


@dataclass
class CareerRoadmap:
    """A personalized career roadmap built from a SkillGapReport."""

    target_role: str = ""
    current_readiness: int = 0
    estimated_total_effort: str = ""
    phases: list[RoadmapPhase] = field(default_factory=list)
    final_outcome: str = ""
    enriched: bool = False

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    @property
    def all_skills(self) -> list[str]:
        """Every skill across all phases (in phase order)."""
        out: list[str] = []
        for p in self.phases:
            out.extend(p.skills)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "CareerRoadmap":
        if not isinstance(data, dict):
            raise CareerRoadmapParseError("Roadmap was not a JSON object.")
        phases_raw = data.get("phases")
        phases = (
            [RoadmapPhase.from_dict(p) for p in _as_list(phases_raw)]
            if phases_raw is not None
            else []
        )
        return cls(
            target_role=_clean_str(data.get("target_role")),
            current_readiness=_coerce_int(data.get("current_readiness")),
            estimated_total_effort=_clean_str(data.get("estimated_total_effort")),
            phases=phases,
            final_outcome=_clean_str(data.get("final_outcome")),
            enriched=bool(data.get("enriched", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "CareerRoadmap":
        if not raw or not raw.strip():
            raise CareerRoadmapParseError("Empty response.")
        text = _strip_code_fence(raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CareerRoadmapParseError(
                f"Invalid JSON ({e.msg} at line {e.lineno} col {e.colno})."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phases"] = [p.to_dict() for p in self.phases]
        return d


# --------------------------------------------------------------------------- #
# Coercion helpers (module-private).
# --------------------------------------------------------------------------- #


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_str_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t
