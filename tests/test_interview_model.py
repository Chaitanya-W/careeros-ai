"""Tests for the Interview Coach data models + enums."""

from __future__ import annotations

import pytest

from src.modules.voice_interview.model import (
    AnswerEvaluation,
    Difficulty,
    InterviewAnswer,
    InterviewCategory,
    InterviewError,
    InterviewParseError,
    InterviewQuestion,
    InterviewReport,
    InterviewSession,
    SessionStatus,
    clamp_score,
)


# ---- (1) InterviewCategory -----------------------------------------


def test_interview_category_members() -> None:
    assert {c.value for c in InterviewCategory} == {
        "technical", "behavioral", "project", "resume", "job_specific", "skill_gap"
    }


def test_interview_category_from_value() -> None:
    assert InterviewCategory.from_value("technical") is InterviewCategory.TECHNICAL
    assert InterviewCategory.from_value("JOB_SPECIFIC") is InterviewCategory.JOB_SPECIFIC
    assert InterviewCategory.from_value(None) is InterviewCategory.TECHNICAL
    assert InterviewCategory.from_value("bogus") is InterviewCategory.TECHNICAL


# ---- (2) Difficulty ------------------------------------------------


def test_difficulty_members() -> None:
    assert {d.value for d in Difficulty} == {"easy", "medium", "hard"}


def test_difficulty_from_value() -> None:
    assert Difficulty.from_value("EASY") is Difficulty.EASY
    assert Difficulty.from_value(None) is Difficulty.MEDIUM


# ---- SessionStatus ------------------------------------------------


def test_session_status_members() -> None:
    assert {s.value for s in SessionStatus} == {"not_started", "in_progress", "completed"}


# ---- (3) InterviewQuestion ----------------------------------------


def test_interview_question_defaults() -> None:
    q = InterviewQuestion()
    assert q.question_id == ""
    assert q.category is InterviewCategory.TECHNICAL
    assert q.difficulty is Difficulty.MEDIUM
    assert q.source == ""


def test_interview_question_from_dict() -> None:
    q = InterviewQuestion.from_dict(
        {"question_id": "IQ-001", "question": "q?", "category": "behavioral", "difficulty": "hard", "source": "RESUME"}
    )
    assert q.question_id == "IQ-001"
    assert q.category is InterviewCategory.BEHAVIORAL
    assert q.difficulty is Difficulty.HARD


def test_interview_question_from_dict_non_dict_raises() -> None:
    with pytest.raises(InterviewParseError):
        InterviewQuestion.from_dict("nope")  # type: ignore[arg-type]


# ---- (4) InterviewAnswer ------------------------------------------


def test_interview_answer_defaults() -> None:
    a = InterviewAnswer()
    assert a.answer == ""
    assert a.evidence_references == []


def test_interview_answer_from_dict() -> None:
    a = InterviewAnswer.from_dict({"question_id": "IQ-001", "answer": "ans", "evidence_references": ["EXP-001"]})
    assert a.question_id == "IQ-001"
    assert a.evidence_references == ["EXP-001"]


# ---- (5) AnswerEvaluation ----------------------------------------


def test_answer_evaluation_defaults() -> None:
    e = AnswerEvaluation()
    assert e.overall_score == 0.0
    assert e.unsupported_claims == []
    assert e.evidence_alignment == "aligned"


def test_answer_evaluation_from_dict() -> None:
    e = AnswerEvaluation.from_dict(
        {"question_id": "IQ-001", "overall_score": 7.5, "strengths": ["x"], "unsupported_claims": ["y"]}
    )
    assert e.overall_score == 7.5
    assert e.strengths == ["x"]
    assert e.unsupported_claims == ["y"]


# ---- (6) InterviewSession ----------------------------------------


def test_interview_session_defaults() -> None:
    s = InterviewSession()
    assert s.status is SessionStatus.NOT_STARTED
    assert s.answers == {}
    assert s.evaluations == {}


def test_interview_session_from_dict_round_trip() -> None:
    s = InterviewSession(
        session_id="s1", target_role="Eng", question_ids=["IQ-001"],
        current_question_index=1, status=SessionStatus.IN_PROGRESS,
    )
    d = s.to_dict()
    assert d["status"] == "in_progress"
    again = InterviewSession.from_dict(d)
    assert again.status is SessionStatus.IN_PROGRESS
    assert again.question_ids == ["IQ-001"]


# ---- (7) InterviewReport -----------------------------------------


def test_interview_report_defaults() -> None:
    r = InterviewReport()
    assert r.readiness_score == 0.0
    assert r.completed_questions == 0


def test_interview_report_from_dict() -> None:
    r = InterviewReport.from_dict(
        {"readiness_score": 74, "technical_score": 7.8, "strengths": ["x"]}
    )
    assert r.readiness_score == 74
    assert r.technical_score == 7.8
    assert r.strengths == ["x"]


# ---- (8) score bounds / clamping ---------------------------------


def test_clamp_score_bounds() -> None:
    assert clamp_score(5.0) == 5.0
    assert clamp_score(-3.0) == 0.0
    assert clamp_score(15.0) == 10.0
    assert clamp_score(0.0) == 0.0
    assert clamp_score(10.0) == 10.0


def test_error_hierarchy() -> None:
    assert issubclass(InterviewParseError, InterviewError)
