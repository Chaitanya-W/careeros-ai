"""AI Evaluation Dashboard data models.

Defines the structured models for the deterministic evaluation pipeline:

- :class:`EvaluationMetric` — a single named metric with value/availability.
- :class:`AnswerQuality` — aggregated interview answer-quality scores.
- :class:`EvaluationSummary` — a snapshot of all available metrics.
- :class:`TrendPoint` — a trend across multiple evaluation snapshots.
- :class:`EvaluationReport` — the full report (summary + metrics + trends
  + strengths + improvements + insights).

All models are robust to missing information. Missing data is represented
explicitly via ``available=False`` / ``None`` / ``"not_available"``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class EvaluationError(Exception):
    """Base error for evaluation-dashboard model problems."""


class EvaluationParseError(EvaluationError):
    """Raised when a model response cannot be parsed."""


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"

    @classmethod
    def from_value(cls, v: Any) -> "TrendDirection":
        if isinstance(v, TrendDirection):
            return v
        if v is None:
            return TrendDirection.INSUFFICIENT_DATA
        s = str(v).strip().lower()
        for m in cls:
            if m.value == s:
                return m
        return TrendDirection.INSUFFICIENT_DATA


@dataclass
class EvaluationMetric:
    """A single named metric."""

    name: str = ""
    label: str = ""
    value: float = 0.0
    unit: str = ""  # "%", "count", "score", "status"
    category: str = ""  # "interview", "job_match", "skill", "roadmap", "salary"
    available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnswerQuality:
    """Aggregated interview answer-quality scores (0-10 each)."""

    technical_avg: float = 0.0
    relevance_avg: float = 0.0
    specificity_avg: float = 0.0
    communication_avg: float = 0.0
    completeness_avg: float = 0.0
    overall_avg: float = 0.0
    unsupported_claim_count: int = 0
    evidence_alignment: str = "not_available"
    total_evaluated: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """A snapshot of all available evaluation metrics."""

    readiness_score: Optional[float] = None
    answer_quality: AnswerQuality = field(default_factory=AnswerQuality)
    job_match_score: Optional[float] = None
    skill_readiness: Optional[float] = None
    roadmap_readiness: Optional[float] = None
    negotiation_prepared: Optional[bool] = None
    interview_completed: bool = False
    interview_total_questions: int = 0
    interview_completed_questions: int = 0
    generated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answer_quality"] = self.answer_quality.to_dict()
        return d


@dataclass
class TrendPoint:
    """A trend across multiple evaluation snapshots."""

    label: str = ""
    direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    values: list[float] = field(default_factory=list)
    current: Optional[float] = None
    previous: Optional[float] = None
    delta: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d


@dataclass
class EvaluationReport:
    """The full evaluation report."""

    summary: EvaluationSummary = field(default_factory=EvaluationSummary)
    metrics: list[EvaluationMetric] = field(default_factory=list)
    trends: list[TrendPoint] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    generated_at: str = ""
    enriched: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary.to_dict()
        d["metrics"] = [m.to_dict() for m in self.metrics]
        d["trends"] = [t.to_dict() for t in self.trends]
        return d
