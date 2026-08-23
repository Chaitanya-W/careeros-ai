"""AI Interview Coach data models.

Defines the structured models for the evidence-grounded interview coach:

- :class:`InterviewCategory`, :class:`Difficulty`, :class:`SessionStatus`
  (constrained str enums).
- :class:`InterviewQuestion` — one question with category/difficulty/source.
- :class:`InterviewAnswer` — a candidate's answer to a question.
- :class:`AnswerEvaluation` — heuristic 0-10 scores + coaching feedback.
- :class:`InterviewSession` — a full interview session (serializable).
- :class:`InterviewReport` — final readiness report.

Scores are 0-10 (per-evaluation) / 0-100 (readiness). They are heuristic
interview-coaching scores — NOT scientifically validated psychometric
measurements or hiring predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class InterviewError(Exception):
    """Base error for interview-coach model / parsing problems."""


class InterviewParseError(InterviewError):
    """Raised when a model response cannot be parsed into the schema."""


class InterviewCategory(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PROJECT = "project"
    RESUME = "resume"
    JOB_SPECIFIC = "job_specific"
    SKILL_GAP = "skill_gap"

    @classmethod
    def from_value(cls, value: Any) -> "InterviewCategory":
        if isinstance(value, InterviewCategory):
            return value
        if value is None:
            return InterviewCategory.TECHNICAL
        s = str(value).strip().lower()
        for m in cls:
            if m.value == s or m.name.lower() == s:
                return m
        return InterviewCategory.TECHNICAL


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def from_value(cls, value: Any) -> "Difficulty":
        if isinstance(value, Difficulty):
            return value
        if value is None:
            return Difficulty.MEDIUM
        s = str(value).strip().lower()
        for m in cls:
            if m.value == s:
                return m
        return Difficulty.MEDIUM


class SessionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

    @classmethod
    def from_value(cls, value: Any) -> "SessionStatus":
        if isinstance(value, SessionStatus):
            return value
        if value is None:
            return SessionStatus.NOT_STARTED
        s = str(value).strip().lower()
        for m in cls:
            if m.value == s:
                return m
        return SessionStatus.NOT_STARTED


@dataclass
class InterviewQuestion:
    """One interview question."""

    question_id: str = ""
    question: str = ""
    category: InterviewCategory = InterviewCategory.TECHNICAL
    difficulty: Difficulty = Difficulty.MEDIUM
    target_skill: str = ""
    source: str = ""  # RESUME / JOB / SKILL_GAP / PROJECT / GENERAL
    rationale: str = ""
    expected_evidence: str = ""
    related_job_requirement: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "InterviewQuestion":
        if not isinstance(data, dict):
            raise InterviewParseError("Question was not a JSON object.")
        return cls(
            question_id=_clean_str(data.get("question_id")),
            question=_clean_str(data.get("question")),
            category=InterviewCategory.from_value(data.get("category")),
            difficulty=Difficulty.from_value(data.get("difficulty")),
            target_skill=_clean_str(data.get("target_skill")),
            source=_clean_str(data.get("source")),
            rationale=_clean_str(data.get("rationale")),
            expected_evidence=_clean_str(data.get("expected_evidence")),
            related_job_requirement=_clean_str(data.get("related_job_requirement")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["difficulty"] = self.difficulty.value
        return d


@dataclass
class InterviewAnswer:
    """A candidate's answer to a question."""

    question_id: str = ""
    answer: str = ""
    submitted_at: str = ""
    evidence_references: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "InterviewAnswer":
        if not isinstance(data, dict):
            raise InterviewParseError("Answer was not a JSON object.")
        return cls(
            question_id=_clean_str(data.get("question_id")),
            answer=_clean_str(data.get("answer")),
            submitted_at=_clean_str(data.get("submitted_at")),
            evidence_references=_coerce_str_list(data.get("evidence_references")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnswerEvaluation:
    """Heuristic 0-10 evaluation of an answer."""

    question_id: str = ""
    relevance_score: float = 0.0
    technical_score: float = 0.0
    specificity_score: float = 0.0
    communication_score: float = 0.0
    completeness_score: float = 0.0
    overall_score: float = 0.0
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    missing_points: list[str] = field(default_factory=list)
    evidence_alignment: str = "aligned"  # aligned / partial / unsupported
    unsupported_claims: list[str] = field(default_factory=list)
    recommendation: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "AnswerEvaluation":
        if not isinstance(data, dict):
            raise InterviewParseError("Evaluation was not a JSON object.")
        return cls(
            question_id=_clean_str(data.get("question_id")),
            relevance_score=_coerce_score(data.get("relevance_score")),
            technical_score=_coerce_score(data.get("technical_score")),
            specificity_score=_coerce_score(data.get("specificity_score")),
            communication_score=_coerce_score(data.get("communication_score")),
            completeness_score=_coerce_score(data.get("completeness_score")),
            overall_score=_coerce_score(data.get("overall_score")),
            strengths=_coerce_str_list(data.get("strengths")),
            improvements=_coerce_str_list(data.get("improvements")),
            missing_points=_coerce_str_list(data.get("missing_points")),
            evidence_alignment=_clean_str(data.get("evidence_alignment")) or "aligned",
            unsupported_claims=_coerce_str_list(data.get("unsupported_claims")),
            recommendation=_clean_str(data.get("recommendation")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterviewSession:
    """A full interview session (serializable for Streamlit state)."""

    session_id: str = ""
    target_role: str = ""
    question_ids: list[str] = field(default_factory=list)
    current_question_index: int = 0
    answers: dict[str, InterviewAnswer] = field(default_factory=dict)
    evaluations: dict[str, AnswerEvaluation] = field(default_factory=dict)
    status: SessionStatus = SessionStatus.NOT_STARTED
    readiness_summary: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "InterviewSession":
        if not isinstance(data, dict):
            raise InterviewParseError("Session was not a JSON object.")
        answers = {
            _clean_str(k): InterviewAnswer.from_dict(v)
            for k, v in (data.get("answers") or {}).items()
            if isinstance(v, dict)
        }
        evaluations = {
            _clean_str(k): AnswerEvaluation.from_dict(v)
            for k, v in (data.get("evaluations") or {}).items()
            if isinstance(v, dict)
        }
        return cls(
            session_id=_clean_str(data.get("session_id")),
            target_role=_clean_str(data.get("target_role")),
            question_ids=_coerce_str_list(data.get("question_ids")),
            current_question_index=_coerce_int(data.get("current_question_index")),
            answers=answers,
            evaluations=evaluations,
            status=SessionStatus.from_value(data.get("status")),
            readiness_summary=_clean_str(data.get("readiness_summary")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answers"] = {k: v.to_dict() for k, v in self.answers.items()}
        d["evaluations"] = {k: v.to_dict() for k, v in self.evaluations.items()}
        d["status"] = self.status.value
        return d


@dataclass
class InterviewReport:
    """Final readiness report (heuristic scores, 0-100)."""

    readiness_score: float = 0.0
    technical_score: float = 0.0
    behavioral_score: float = 0.0
    communication_score: float = 0.0
    job_alignment_score: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommended_practice: list[str] = field(default_factory=list)
    completed_questions: int = 0
    total_questions: int = 0

    @classmethod
    def from_dict(cls, data: Any) -> "InterviewReport":
        if not isinstance(data, dict):
            raise InterviewParseError("Report was not a JSON object.")
        return cls(
            readiness_score=_coerce_score(data.get("readiness_score")),
            technical_score=_coerce_score(data.get("technical_score")),
            behavioral_score=_coerce_score(data.get("behavioral_score")),
            communication_score=_coerce_score(data.get("communication_score")),
            job_alignment_score=_coerce_score(data.get("job_alignment_score")),
            strengths=_coerce_str_list(data.get("strengths")),
            weaknesses=_coerce_str_list(data.get("weaknesses")),
            recommended_practice=_coerce_str_list(data.get("recommended_practice")),
            completed_questions=_coerce_int(data.get("completed_questions")),
            total_questions=_coerce_int(data.get("total_questions")),
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


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _coerce_score(value: Any) -> float:
    """Coerce to a float; invalid -> 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_score(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """Clamp a score to [lo, hi]."""
    return max(lo, min(hi, float(value)))
