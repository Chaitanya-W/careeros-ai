"""Salary Negotiator data models.

Defines the structured models for the deterministic salary-negotiation
pipeline:

- :class:`SalaryBenchmark` — low/mid/high market range (heuristic, not from
  a paid salary API).
- :class:`NegotiationPoint` — one talking point with evidence citations.
- :class:`NegotiationScript` — the full script (opening, leverage, ask, etc.).
- :class:`CounterOffer` — a structured counter-offer template.

All models are robust to missing information (``from_dict`` tolerates
absent keys / wrong types). Scores/numbers are clamped to sensible bounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class SalaryError(Exception):
    """Base error for salary-negotiation model / parsing problems."""


class SalaryParseError(SalaryError):
    """Raised when a model response cannot be parsed into the schema."""


class ValidationStatus(str, Enum):
    """Constrained validation status for negotiation points."""

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
        for m in cls:
            if m.value == s:
                return m
        return ValidationStatus.PASS


@dataclass
class SalaryBenchmark:
    """A heuristic market compensation range (NOT from a paid salary API)."""

    role: str = ""
    location: str = ""
    seniority: str = ""
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    currency: str = "USD"
    currency_symbol: str = "$"
    methodology: str = ""
    disclaimer: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "SalaryBenchmark":
        if not isinstance(data, dict):
            raise SalaryParseError("Salary benchmark was not a JSON object.")
        return cls(
            role=_clean_str(data.get("role")),
            location=_clean_str(data.get("location")),
            seniority=_clean_str(data.get("seniority")),
            low=_coerce_float(data.get("low")),
            mid=_coerce_float(data.get("mid")),
            high=_coerce_float(data.get("high")),
            currency=_clean_str(data.get("currency")) or "USD",
            currency_symbol=_clean_str(data.get("currency_symbol")) or "$",
            methodology=_clean_str(data.get("methodology")),
            disclaimer=_clean_str(data.get("disclaimer")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NegotiationPoint:
    """One negotiation talking point with evidence citations."""

    point_id: str = ""
    title: str = ""
    wording: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.PASS

    @classmethod
    def from_dict(cls, data: Any) -> "NegotiationPoint":
        if not isinstance(data, dict):
            raise SalaryParseError("Negotiation point was not a JSON object.")
        return cls(
            point_id=_clean_str(data.get("point_id")),
            title=_clean_str(data.get("title")),
            wording=_clean_str(data.get("wording")),
            evidence_ids=_coerce_str_list(data.get("evidence_ids")),
            validation_status=ValidationStatus.from_value(data.get("validation_status")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validation_status"] = self.validation_status.value
        return d


@dataclass
class NegotiationScript:
    """A full negotiation script grounded in candidate evidence."""

    target_role: str = ""
    opening_position: str = ""
    leverage_points: list[str] = field(default_factory=list)
    talking_points: list[NegotiationPoint] = field(default_factory=list)
    fallback_position: str = ""
    validation_status: ValidationStatus = ValidationStatus.PASS
    enriched: bool = False
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "NegotiationScript":
        if not isinstance(data, dict):
            raise SalaryParseError("Negotiation script was not a JSON object.")
        tps = data.get("talking_points")
        points = (
            [NegotiationPoint.from_dict(p) for p in _as_list(tps) if isinstance(p, dict)]
            if tps is not None
            else []
        )
        return cls(
            target_role=_clean_str(data.get("target_role")),
            opening_position=_clean_str(data.get("opening_position")),
            leverage_points=_coerce_str_list(data.get("leverage_points")),
            talking_points=points,
            fallback_position=_clean_str(data.get("fallback_position")),
            validation_status=ValidationStatus.from_value(data.get("validation_status")),
            enriched=bool(data.get("enriched", False)),
            warnings=_coerce_str_list(data.get("warnings")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validation_status"] = self.validation_status.value
        d["talking_points"] = [p.to_dict() for p in self.talking_points]
        return d


@dataclass
class CounterOffer:
    """A structured counter-offer template with a deterministic number."""

    current_offer: Optional[float] = None
    expected_salary: Optional[float] = None
    benchmark_mid: float = 0.0
    counter_number: float = 0.0
    rationale: str = ""
    template_wording: str = ""
    currency: str = "USD"

    @classmethod
    def from_dict(cls, data: Any) -> "CounterOffer":
        if not isinstance(data, dict):
            raise SalaryParseError("Counter offer was not a JSON object.")
        return cls(
            current_offer=_opt_float(data.get("current_offer")),
            expected_salary=_opt_float(data.get("expected_salary")),
            benchmark_mid=_coerce_float(data.get("benchmark_mid")),
            counter_number=_coerce_float(data.get("counter_number")),
            rationale=_clean_str(data.get("rationale")),
            template_wording=_clean_str(data.get("template_wording")),
            currency=_clean_str(data.get("currency")) or "USD",
        )

    def to_dict(self) -> dict:
        return asdict(self)


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


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
