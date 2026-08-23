"""Skill Gap data models.

Defines :class:`Priority` (a constrained enum), :class:`SkillGap` (one
gap with deterministic + AI-enriched fields), and :class:`SkillGapReport`
(the aggregate report). Both are robust to missing information —
:meth:`from_dict` tolerates absent keys, wrong types, and extra keys.

Design notes
------------
- ``current_evidence`` defaults to ``"No evidence found in resume"`` so a
  missing skill is never falsely claimed as something the candidate has.
- AI-enrichment fields (``why_it_matters``, ``learning_objectives``,
  ``learning_path``, ``practice_project``, ``estimated_effort``) default to
  empty so the report stays useful without a Gemini key.
- ``Priority`` is a ``str`` enum so it serializes naturally and compares
  to plain strings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class SkillGapError(Exception):
    """Base error for skill-gap model / parsing problems."""


class SkillGapParseError(SkillGapError):
    """Raised when a model response cannot be parsed into the schema."""


class Priority(str, Enum):
    """Constrained priority for a skill gap."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_value(cls, value: Any) -> "Priority":
        """Coerce any value to a :class:`Priority`; unknown -> LOW."""
        if isinstance(value, Priority):
            return value
        if value is None:
            return Priority.LOW
        s = str(value).strip().lower()
        for member in cls:
            if member.value == s:
                return member
        return Priority.LOW


@dataclass
class SkillGap:
    """A single skill gap."""

    skill: str = ""
    priority: Priority = Priority.LOW
    current_level: str = ""
    target_level: str = "Not specified"
    current_evidence: str = "No evidence found in resume"
    gap_reason: str = ""
    # AI-enriched fields (empty until Gemini enrichment is applied):
    why_it_matters: str = ""
    learning_objectives: list[str] = field(default_factory=list)
    learning_path: list[str] = field(default_factory=list)
    practice_project: str = ""
    estimated_effort: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "SkillGap":
        """Build a :class:`SkillGap` from a parsed JSON object (robust)."""
        if not isinstance(data, dict):
            raise SkillGapParseError("Skill gap was not a JSON object.")
        return cls(
            skill=_clean_str(data.get("skill")),
            priority=Priority.from_value(data.get("priority")),
            current_level=_clean_str(data.get("current_level")),
            target_level=_clean_str(data.get("target_level")) or "Not specified",
            current_evidence=_clean_str(data.get("current_evidence"))
            or "No evidence found in resume",
            gap_reason=_clean_str(data.get("gap_reason")),
            why_it_matters=_clean_str(data.get("why_it_matters")),
            learning_objectives=_coerce_str_list(
                data.get("learning_objectives")
            ),
            learning_path=_coerce_str_list(data.get("learning_path")),
            practice_project=_clean_str(data.get("practice_project")),
            estimated_effort=_clean_str(data.get("estimated_effort")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = self.priority.value
        return d


@dataclass
class SkillGapReport:
    """Aggregate skill-gap report for a target role."""

    target_role: str = ""
    total_gaps: int = 0
    high_priority_count: int = 0
    medium_priority_count: int = 0
    low_priority_count: int = 0
    skill_gaps: list[SkillGap] = field(default_factory=list)
    summary: str = ""
    generated_at: str = ""
    enriched: bool = False  # True once AI enrichment has been applied

    @classmethod
    def from_dict(cls, data: Any) -> "SkillGapReport":
        """Build a :class:`SkillGapReport` from a dict (robust)."""
        if not isinstance(data, dict):
            raise SkillGapParseError("Skill gap report was not a JSON object.")
        gaps_raw = data.get("skill_gaps")
        gaps = (
            [SkillGap.from_dict(g) for g in _as_list(gaps_raw)]
            if gaps_raw is not None
            else []
        )
        return cls(
            target_role=_clean_str(data.get("target_role")),
            total_gaps=_coerce_int(data.get("total_gaps")),
            high_priority_count=_coerce_int(data.get("high_priority_count")),
            medium_priority_count=_coerce_int(
                data.get("medium_priority_count")
            ),
            low_priority_count=_coerce_int(data.get("low_priority_count")),
            skill_gaps=gaps,
            summary=_clean_str(data.get("summary")),
            generated_at=_clean_str(data.get("generated_at")),
            enriched=bool(data.get("enriched", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SkillGapReport":
        if not raw or not raw.strip():
            raise SkillGapParseError("Empty response.")
        text = _strip_code_fence(raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SkillGapParseError(
                f"Invalid JSON ({e.msg} at line {e.lineno} col {e.colno})."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skill_gaps"] = [g.to_dict() for g in self.skill_gaps]
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
