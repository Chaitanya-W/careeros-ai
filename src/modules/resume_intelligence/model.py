"""Resume analysis data model.

Defines the structured :class:`ResumeAnalysis` schema produced by the
Resume Intelligence analyzer. The model is deliberately robust to missing
information — :meth:`ResumeAnalysis.from_dict` tolerates absent keys,
non-list values, wrong types, and extra/unknown keys without raising, so
a partial or sloppy model response degrades gracefully instead of
crashing the UI.

No information is ever fabricated: absent sections resolve to empty lists
and an absent professional summary resolves to ``None``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


class ResumeAnalysisError(Exception):
    """Base error for resume-analysis model / parsing problems."""


class ResumeAnalysisParseError(ResumeAnalysisError):
    """Raised when the model response cannot be parsed into the schema."""


# --------------------------------------------------------------------------- #
# Sub-item dataclasses — each is robust to non-dict / missing fields.
# --------------------------------------------------------------------------- #


@dataclass
class ExperienceItem:
    """A single work-experience entry."""

    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "ExperienceItem":
        if not isinstance(data, dict):
            return cls()
        return cls(
            title=_clean_str(data.get("title")),
            company=_clean_str(data.get("company")),
            duration=_clean_str(data.get("duration")),
            description=_clean_str(data.get("description")),
        )


@dataclass
class EducationItem:
    """A single education entry."""

    degree: str = ""
    institution: str = ""
    year: str = ""
    details: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "EducationItem":
        if not isinstance(data, dict):
            return cls()
        return cls(
            degree=_clean_str(data.get("degree")),
            institution=_clean_str(data.get("institution")),
            year=_clean_str(data.get("year")),
            details=_clean_str(data.get("details")),
        )


@dataclass
class ProjectItem:
    """A single project entry."""

    name: str = ""
    description: str = ""
    technologies: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "ProjectItem":
        if not isinstance(data, dict):
            return cls()
        return cls(
            name=_clean_str(data.get("name")),
            description=_clean_str(data.get("description")),
            technologies=_clean_str(data.get("technologies")),
        )


@dataclass
class CertificationItem:
    """A single certification entry."""

    name: str = ""
    issuer: str = ""
    year: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "CertificationItem":
        if not isinstance(data, dict):
            return cls()
        return cls(
            name=_clean_str(data.get("name")),
            issuer=_clean_str(data.get("issuer")),
            year=_clean_str(data.get("year")),
        )


# --------------------------------------------------------------------------- #
# Top-level analysis result.
# --------------------------------------------------------------------------- #


@dataclass
class ResumeAnalysis:
    """Structured result of analyzing a resume.

    All collections default to empty and the professional summary defaults
    to ``None`` so a sparse resume never produces fabricated content.
    """

    overall_score: int = 0
    professional_summary: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    experience: list[ExperienceItem] = field(default_factory=list)
    education: list[EducationItem] = field(default_factory=list)
    projects: list[ProjectItem] = field(default_factory=list)
    certifications: list[CertificationItem] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    recommended_skills: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "ResumeAnalysis":
        """Build a :class:`ResumeAnalysis` from a parsed JSON object.

        Robust to: non-dict input (raises — caller should validate JSON
        first), missing keys, wrong types, and extra keys. Sub-items that
        are not dicts become empty items so a malformed array never crashes.
        """
        if not isinstance(data, dict):
            raise ResumeAnalysisParseError(
                "Gemini response was not a JSON object."
            )
        return cls(
            overall_score=_coerce_score(data.get("overall_score")),
            professional_summary=_clean_optional_str(
                data.get("professional_summary")
            ),
            skills=_coerce_str_list(data.get("skills")),
            experience=[
                ExperienceItem.from_dict(x)
                for x in _as_list(data.get("experience"))
            ],
            education=[
                EducationItem.from_dict(x)
                for x in _as_list(data.get("education"))
            ],
            projects=[
                ProjectItem.from_dict(x)
                for x in _as_list(data.get("projects"))
            ],
            certifications=[
                CertificationItem.from_dict(x)
                for x in _as_list(data.get("certifications"))
            ],
            strengths=_coerce_str_list(data.get("strengths")),
            weaknesses=_coerce_str_list(data.get("weaknesses")),
            improvement_suggestions=_coerce_str_list(
                data.get("improvement_suggestions")
            ),
            recommended_skills=_coerce_str_list(
                data.get("recommended_skills")
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ResumeAnalysis":
        """Parse a JSON string into a :class:`ResumeAnalysis`."""
        if not raw or not raw.strip():
            raise ResumeAnalysisParseError("Gemini returned an empty response.")
        text = _strip_code_fence(raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ResumeAnalysisParseError(
                f"Gemini response was not valid JSON "
                f"({e.msg} at line {e.lineno} col {e.colno})."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (useful for caching / display)."""
        return asdict(self)

    def is_empty(self) -> bool:
        """True if no meaningful analysis content was produced."""
        return (
            self.overall_score == 0
            and self.professional_summary is None
            and not (
                self.skills
                or self.experience
                or self.education
                or self.projects
                or self.certifications
                or self.strengths
                or self.weaknesses
                or self.improvement_suggestions
                or self.recommended_skills
            )
        )


# --------------------------------------------------------------------------- #
# Coercion helpers (module-private).
# --------------------------------------------------------------------------- #


def _clean_str(value: Any) -> str:
    """Coerce to a stripped string; None / non-str -> ''."""
    if value is None:
        return ""
    return str(value).strip()


def _clean_optional_str(value: Any) -> Optional[str]:
    """None / empty -> None; otherwise stripped string."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_list(value: Any) -> list:
    """None -> []; list -> as-is; anything else -> [value]."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce to list[str], dropping None / empty items."""
    out: list[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _coerce_score(value: Any) -> int:
    """Coerce to int clamped to 0-100; invalid -> 0."""
    if value is None:
        return 0
    try:
        n = int(float(value))  # tolerate "85" or 85.0
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t
