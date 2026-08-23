"""Job Match data models.

Defines :class:`JobAnalysis` (structured extraction from a job
description) and :class:`MatchAnalysis` (the deterministic comparison of a
resume against a job). Both are robust to missing information —
:meth:`from_dict` tolerates absent keys, wrong types, and extra keys
without raising, so a partial model response degrades gracefully.

No information is fabricated: absent fields resolve to empty strings or
empty lists. Scores are clamped to the documented 0-100 range.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


class JobAnalysisError(Exception):
    """Base error for job-analysis model / parsing problems."""


class JobAnalysisParseError(JobAnalysisError):
    """Raised when the model response cannot be parsed into JobAnalysis."""


class MatchAnalysisError(Exception):
    """Base error for match-analysis model problems."""


# --------------------------------------------------------------------------- #
# JobAnalysis — structured extraction from a job description.
# --------------------------------------------------------------------------- #


@dataclass
class JobAnalysis:
    """Structured extraction from a single job description."""

    job_title: str = ""
    company: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    experience_requirements: str = ""
    education_requirements: str = ""
    keywords: list[str] = field(default_factory=list)
    domain: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "JobAnalysis":
        """Build a :class:`JobAnalysis` from a parsed JSON object.

        Robust to non-dict input (raises), missing keys, wrong types,
        and extra keys. Sub-strings stay strings; sub-lists are coerced.
        """
        if not isinstance(data, dict):
            raise JobAnalysisParseError(
                "Gemini response was not a JSON object."
            )
        return cls(
            job_title=_clean_str(data.get("job_title")),
            company=_clean_str(data.get("company")),
            required_skills=_coerce_str_list(data.get("required_skills")),
            preferred_skills=_coerce_str_list(data.get("preferred_skills")),
            responsibilities=_coerce_str_list(data.get("responsibilities")),
            experience_requirements=_clean_str(
                data.get("experience_requirements")
            ),
            education_requirements=_clean_str(
                data.get("education_requirements")
            ),
            keywords=_coerce_str_list(data.get("keywords")),
            domain=_clean_str(data.get("domain")),
        )

    @classmethod
    def from_json(cls, raw: str) -> "JobAnalysis":
        """Parse a JSON string into a :class:`JobAnalysis`."""
        if not raw or not raw.strip():
            raise JobAnalysisParseError("Gemini returned an empty response.")
        text = _strip_code_fence(raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise JobAnalysisParseError(
                f"Gemini response was not valid JSON "
                f"({e.msg} at line {e.lineno} col {e.colno})."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def has_any_data(self) -> bool:
        """True if at least one meaningful field is populated."""
        return any(
            (
                self.job_title,
                self.company,
                self.required_skills,
                self.preferred_skills,
                self.responsibilities,
                self.experience_requirements,
                self.education_requirements,
                self.keywords,
                self.domain,
            )
        )


# --------------------------------------------------------------------------- #
# MatchAnalysis — deterministic comparison of a resume against a job.
# --------------------------------------------------------------------------- #


@dataclass
class MatchAnalysis:
    """Result of comparing a :class:`ResumeAnalysis` against a
    :class:`JobAnalysis`.

    Scores are integers in the documented 0-100 range. Collections default
    to empty so a sparse match never produces fabricated content. The
    ``explanation`` and ``recommended_actions`` are deterministic template
    strings derived from the match data (no LLM call).
    """

    overall_match_score: int = 0
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    partial_match_skills: list[str] = field(default_factory=list)
    experience_match: int = 0
    education_match: int = 0
    keyword_match: int = 0
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    explanation: str = ""
    recommended_actions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "MatchAnalysis":
        """Build a :class:`MatchAnalysis` from a dict (for tests / caching)."""
        if not isinstance(data, dict):
            raise MatchAnalysisError("Match data was not a dict.")
        return cls(
            overall_match_score=_coerce_score(data.get("overall_match_score")),
            matching_skills=_coerce_str_list(data.get("matching_skills")),
            missing_skills=_coerce_str_list(data.get("missing_skills")),
            partial_match_skills=_coerce_str_list(
                data.get("partial_match_skills")
            ),
            experience_match=_coerce_score(data.get("experience_match")),
            education_match=_coerce_score(data.get("education_match")),
            keyword_match=_coerce_score(data.get("keyword_match")),
            strengths=_coerce_str_list(data.get("strengths")),
            gaps=_coerce_str_list(data.get("gaps")),
            explanation=_clean_str(data.get("explanation")),
            recommended_actions=_coerce_str_list(
                data.get("recommended_actions")
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Coercion helpers (module-private — kept independent of other modules).
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


def _coerce_score(value: Any) -> int:
    """Coerce to int clamped to 0-100; invalid -> 0."""
    if value is None:
        return 0
    try:
        n = int(float(value))
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
