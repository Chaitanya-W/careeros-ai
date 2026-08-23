"""Application Copilot data models.

Defines the evidence-grounded application-assistant schemas:

- :class:`ValidationStatus` — constrained PASS / WARNING / FAIL.
- :class:`EvidenceItem` — one piece of resume evidence with a stable ID.
- :class:`EvidenceValidation` — the validator's verdict on generated text.
- :class:`ApplicationDraft` — a generic generated draft (target role,
  source evidence IDs, content, validation).
- :class:`TailoredResumeSuggestion` — one resume-tailoring recommendation.
- :class:`CoverLetterDraft` — a cover letter + its evidence/validation.
- :class:`BulletSuggestion` — a bullet-optimization recommendation.

All models are robust to missing information (``from_dict`` tolerates
absent keys / wrong types). AI-enriched fields default to empty so the
deterministic layer stays useful without a Gemini key.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class ApplicationCopilotError(Exception):
    """Base error for application-copilot model / parsing problems."""


class ApplicationCopilotParseError(ApplicationCopilotError):
    """Raised when a model response cannot be parsed into the schema."""


class ValidationStatus(str, Enum):
    """Constrained validation status for evidence-grounded output."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

    @classmethod
    def from_value(cls, value: Any) -> "ValidationStatus":
        if isinstance(value, ValidationStatus):
            return value
        if value is None:
            return ValidationStatus.PASS
        s = str(value).strip().lower()
        for member in cls:
            if member.value == s:
                return member
        return ValidationStatus.PASS


@dataclass
class EvidenceItem:
    """One piece of candidate evidence extracted from ResumeAnalysis."""

    evidence_id: str = ""
    source_type: str = ""  # experience / project / education / certification / skill / summary
    source_text: str = ""
    category: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "EvidenceItem":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Evidence item was not a JSON object.")
        return cls(
            evidence_id=_clean_str(data.get("evidence_id")),
            source_type=_clean_str(data.get("source_type")),
            source_text=_clean_str(data.get("source_text")),
            category=_clean_str(data.get("category")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceValidation:
    """Validator verdict on a piece of generated text."""

    status: ValidationStatus = ValidationStatus.PASS
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "EvidenceValidation":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Evidence validation was not a JSON object.")
        return cls(
            status=ValidationStatus.from_value(data.get("status")),
            supported_claims=_coerce_str_list(data.get("supported_claims")),
            unsupported_claims=_coerce_str_list(data.get("unsupported_claims")),
            warnings=_coerce_str_list(data.get("warnings")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ApplicationDraft:
    """A generic generated application draft (target role, source evidence,
    content, validation)."""

    target_role: str = ""
    generated_at: str = ""
    source_evidence_ids: list[str] = field(default_factory=list)
    content: str = ""
    evidence_validation: EvidenceValidation = field(default_factory=EvidenceValidation)

    @classmethod
    def from_dict(cls, data: Any) -> "ApplicationDraft":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Application draft was not a JSON object.")
        ev = data.get("evidence_validation")
        return cls(
            target_role=_clean_str(data.get("target_role")),
            generated_at=_clean_str(data.get("generated_at")),
            source_evidence_ids=_coerce_str_list(data.get("source_evidence_ids")),
            content=_clean_str(data.get("content")),
            evidence_validation=(
                EvidenceValidation.from_dict(ev) if isinstance(ev, dict) else EvidenceValidation()
            ),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence_validation"] = self.evidence_validation.to_dict()
        return d


@dataclass
class TailoredResumeSuggestion:
    """One resume-tailoring recommendation."""

    section: str = ""
    original_content: str = ""
    suggested_content: str = ""
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    validation: EvidenceValidation = field(default_factory=EvidenceValidation)

    @classmethod
    def from_dict(cls, data: Any) -> "TailoredResumeSuggestion":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Tailor suggestion was not a JSON object.")
        v = data.get("validation")
        return cls(
            section=_clean_str(data.get("section")),
            original_content=_clean_str(data.get("original_content")),
            suggested_content=_clean_str(data.get("suggested_content")),
            reason=_clean_str(data.get("reason")),
            evidence_ids=_coerce_str_list(data.get("evidence_ids")),
            validation=(
                EvidenceValidation.from_dict(v) if isinstance(v, dict) else EvidenceValidation()
            ),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validation"] = self.validation.to_dict()
        return d


@dataclass
class CoverLetterDraft:
    """A cover letter draft + its evidence grounding."""

    cover_letter: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_validation: EvidenceValidation = field(default_factory=EvidenceValidation)
    warnings: list[str] = field(default_factory=list)
    target_role: str = ""
    generated_at: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "CoverLetterDraft":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Cover letter draft was not a JSON object.")
        v = data.get("evidence_validation")
        return cls(
            cover_letter=_clean_str(data.get("cover_letter")),
            evidence_ids=_coerce_str_list(data.get("evidence_ids")),
            evidence_validation=(
                EvidenceValidation.from_dict(v) if isinstance(v, dict) else EvidenceValidation()
            ),
            warnings=_coerce_str_list(data.get("warnings")),
            target_role=_clean_str(data.get("target_role")),
            generated_at=_clean_str(data.get("generated_at")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence_validation"] = self.evidence_validation.to_dict()
        return d


@dataclass
class BulletSuggestion:
    """A bullet-optimization recommendation."""

    original_bullet: str = ""
    optimized_bullet: str = ""
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    validation: EvidenceValidation = field(default_factory=EvidenceValidation)

    @classmethod
    def from_dict(cls, data: Any) -> "BulletSuggestion":
        if not isinstance(data, dict):
            raise ApplicationCopilotParseError("Bullet suggestion was not a JSON object.")
        v = data.get("validation")
        return cls(
            original_bullet=_clean_str(data.get("original_bullet")),
            optimized_bullet=_clean_str(data.get("optimized_bullet")),
            reason=_clean_str(data.get("reason")),
            evidence_ids=_coerce_str_list(data.get("evidence_ids")),
            validation=(
                EvidenceValidation.from_dict(v) if isinstance(v, dict) else EvidenceValidation()
            ),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validation"] = self.validation.to_dict()
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


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t
