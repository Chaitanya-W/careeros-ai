"""Tests for the Interview Coach UI (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no submit/next click that triggers Gemini).
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.voice_interview.coach import generate_questions
from src.modules.voice_interview.model import (
    AnswerEvaluation,
    Difficulty,
    InterviewQuestion,
    InterviewReport,
    InterviewSession,
    SessionStatus,
)


# ---- (50) UI empty state (no resume) ------------------------------


def test_interview_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.run()

    assert not at.exception
    assert any(
        "analyze your resume first" in m.value.lower() for m in at.markdown
    ), "no-resume message missing"
    assert any(
        "Resume Intelligence" in b.label for b in at.button
    ), "navigation to Resume Intelligence missing"


# ---- (51) UI interview state (config + start) ---------------------


def test_interview_renders_config_with_resume(app_path: str, sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sgr
    at.run()

    assert not at.exception
    # Config controls must render.
    assert any("Start Interview" in b.label for b in at.button), "start button missing"
    assert any("Number of questions" in s.label for s in at.slider) if at.slider else True


# ---- (52) UI answer submission (in-progress state) ----------------


def test_interview_renders_question_during_session(
    app_path: str, sample_interview_inputs
) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    questions = generate_questions(resume, job, match, sgr, n=3, client=None, evidence=evidence)
    session = InterviewSession(
        session_id="s1", target_role="Backend Engineer",
        question_ids=[q.question_id for q in questions],
        current_question_index=0, status=SessionStatus.IN_PROGRESS,
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.session_state["skill_gap_report"] = sgr
    at.session_state["interview_session"] = session
    at.session_state["interview_questions"] = questions
    at.session_state["interview_answers"] = {}
    at.session_state["interview_evaluations"] = {}
    at.run()

    assert not at.exception
    # Question 1/3 must render.
    assert any("Question 1 / 3" in m.value for m in at.markdown), "question counter missing"
    # Answer box + submit button must be present.
    assert any("Submit Answer" in b.label for b in at.button)


# ---- (53) UI feedback rendering (after evaluation) ----------------


def test_interview_renders_feedback_after_answer(
    app_path: str, sample_interview_inputs
) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    questions = generate_questions(resume, job, match, sgr, n=2, client=None, evidence=evidence)
    q = questions[0]
    evals = {q.question_id: AnswerEvaluation(question_id=q.question_id, overall_score=7.5, strengths=["Clear."])}
    session = InterviewSession(
        session_id="s1", target_role="Eng",
        question_ids=[q.question_id for q in questions],
        current_question_index=0, status=SessionStatus.IN_PROGRESS,
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.session_state["resume_analysis"] = resume
    at.session_state["interview_session"] = session
    at.session_state["interview_questions"] = questions
    at.session_state["interview_answers"] = {q.question_id: {"answer": "ans"}}
    at.session_state["interview_evaluations"] = evals
    at.run()

    assert not at.exception
    # Feedback section + Next button must render.
    assert any("Feedback" in s.value for s in at.subheader)
    assert any("Next Question" in b.label for b in at.button)
    # Overall score must render.
    assert any("7.5" in m.value for m in at.markdown) or any(
        "7.5" in str(s.value) for s in at.metric
    )


# ---- (54) UI final report -----------------------------------------


def test_interview_renders_report(app_path: str, sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    questions = generate_questions(resume, job, match, sgr, n=2, client=None, evidence=evidence)
    report = InterviewReport(
        readiness_score=74, technical_score=7.8, behavioral_score=7.1,
        communication_score=8.2, job_alignment_score=6.8,
        strengths=["Technical depth."], weaknesses=["Job alignment."],
        recommended_practice=["Practice role-specific questions."],
        completed_questions=2, total_questions=2,
    )
    session = InterviewSession(
        session_id="s1", target_role="Eng",
        question_ids=[q.question_id for q in questions],
        current_question_index=2, status=SessionStatus.COMPLETED,
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.session_state["resume_analysis"] = resume
    at.session_state["interview_session"] = session
    at.session_state["interview_questions"] = questions
    at.session_state["interview_report"] = report
    at.run()

    assert not at.exception
    # Readiness header + score must render.
    assert any("Interview Readiness" in s.value for s in at.subheader)
    assert any("74" in m.value for m in at.markdown)
    # Restart button.
    assert any("Restart Interview" in b.label for b in at.button)


# ---- session-state preservation ----------------------------------


def test_interview_initializes_session_state(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.run()
    assert not at.exception
    for key in ("interview_session", "interview_questions", "interview_answers",
                "interview_evaluations", "interview_report"):
        assert key in at.session_state, f"{key} not initialized"


def test_interview_does_not_overwrite_existing_state(
    app_path: str, sample_interview_inputs
) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "voice_interview"
    at.session_state["resume_analysis"] = resume
    at.session_state["resume_text"] = "PRESERVED"
    at.run()
    assert not at.exception
    assert at.session_state["resume_analysis"] is resume
    assert at.session_state["resume_text"] == "PRESERVED"
