"""AI Interview Coach UI workflow.

Flow: configure (role / # questions / difficulty / focus) -> start ->
display ONE question -> user writes answer -> submit -> evaluate ->
show feedback -> next -> final report.

Prerequisites: a ResumeAnalysis is required (resume-based questions). A
JobAnalysis / MatchAnalysis / SkillGapReport enable job-specific and
skill-gap questions but are optional. If no resume exists, show a clear
prerequisite message + navigation.

State keys: interview_session / interview_questions / interview_answers /
interview_evaluations / interview_report. All existing state is preserved.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import streamlit as st

from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import SkillGapReport
from src.modules.voice_interview.coach import (
    build_report,
    evaluate_answer,
    generate_questions,
)
from src.modules.voice_interview.model import (
    AnswerEvaluation,
    Difficulty,
    InterviewAnswer,
    InterviewQuestion,
    InterviewReport,
    InterviewSession,
    SessionStatus,
)
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong during the interview. Please try again."
)


def render() -> None:
    """Render the Interview Coach workflow."""
    section_header(
        "Interview Coach",
        "Practice interview questions tailored to your resume and target role.",
    )
    resume = st.session_state.get("resume_analysis")
    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")
    skill_gap_report = st.session_state.get("skill_gap_report")

    if not isinstance(resume, ResumeAnalysis):
        _render_no_resume_state()
        return

    session = st.session_state.get("interview_session")
    if not isinstance(session, InterviewSession) or session.status is SessionStatus.NOT_STARTED:
        _render_config(resume, job, match, skill_gap_report)
        return

    if session.status is SessionStatus.IN_PROGRESS:
        _render_interview(session)
        return

    if session.status is SessionStatus.COMPLETED:
        _render_report()
        return


# --------------------------------------------------------------------------- #
# No-resume prerequisite
# --------------------------------------------------------------------------- #


def _render_no_resume_state() -> None:
    empty_state_panel(
        title="Analyze your resume first to start a tailored interview.",
        hint="The Interview Coach builds questions from your analyzed resume, "
        "job match, and skill gaps. Head to Resume Intelligence to upload "
        "and analyze your resume.",
    )
    if st.button(
        "Go to Resume Intelligence",
        type="primary",
        key="interview_goto_resume",
    ):
        navigate_to("resume_intelligence")


# --------------------------------------------------------------------------- #
# Configuration + start
# --------------------------------------------------------------------------- #


def _render_config(resume, job, match, skill_gap_report) -> None:
    target_role = _target_role(job)
    st.markdown(f"**Target role:** {target_role}")

    has_job = isinstance(job, JobAnalysis)
    has_gap = isinstance(skill_gap_report, SkillGapReport)
    if not has_job:
        st.info(
            "No job analysis available — the interview will focus on "
            "resume-based and behavioral questions. Add a target job in "
            "Job Match for job-specific and skill-gap questions."
        )

    col_n, col_d = st.columns(2)
    with col_n:
        n = st.slider("Number of questions", min_value=3, max_value=15, value=8, key="interview_n")
    with col_d:
        difficulty = st.selectbox(
            "Difficulty",
            [d.value for d in Difficulty],
            index=1,
            key="interview_difficulty",
        )
    focus = st.selectbox(
        "Interview focus",
        ["Balanced", "Technical-heavy", "Behavioral-heavy"],
        key="interview_focus",
    )

    if st.button("Start Interview", type="primary", key="interview_start"):
        _start_interview(resume, job, match, skill_gap_report, n, Difficulty(difficulty), focus)


def _start_interview(resume, job, match, skill_gap_report, n, difficulty, focus) -> None:
    ratios = _focus_ratios(focus)
    with st.spinner("Planning your interview…"):
        try:
            client = _maybe_gemini_client()
            questions = generate_questions(
                resume, job, match, skill_gap_report,
                n=n, difficulty=difficulty, client=client,
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error starting interview")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    session = InterviewSession(
        session_id=uuid.uuid4().hex[:12],
        target_role=_target_role(job),
        question_ids=[q.question_id for q in questions],
        current_question_index=0,
        status=SessionStatus.IN_PROGRESS,
    )
    st.session_state["interview_session"] = session
    st.session_state["interview_questions"] = questions
    st.session_state["interview_answers"] = {}
    st.session_state["interview_evaluations"] = {}
    st.session_state["interview_report"] = None
    st.rerun()


def _focus_ratios(focus: str):
    from src.modules.voice_interview.model import InterviewCategory
    from src.modules.voice_interview.planner import DEFAULT_RATIOS
    if focus == "Technical-heavy":
        return {
            InterviewCategory.TECHNICAL: 0.45,
            InterviewCategory.JOB_SPECIFIC: 0.25,
            InterviewCategory.BEHAVIORAL: 0.10,
            InterviewCategory.RESUME: 0.05,
            InterviewCategory.PROJECT: 0.05,
            InterviewCategory.SKILL_GAP: 0.10,
        }
    if focus == "Behavioral-heavy":
        return {
            InterviewCategory.BEHAVIORAL: 0.40,
            InterviewCategory.RESUME: 0.20,
            InterviewCategory.TECHNICAL: 0.20,
            InterviewCategory.JOB_SPECIFIC: 0.10,
            InterviewCategory.PROJECT: 0.05,
            InterviewCategory.SKILL_GAP: 0.05,
        }
    return None  # Balanced (default ratios)


# --------------------------------------------------------------------------- #
# Interview (one question at a time)
# --------------------------------------------------------------------------- #


def _render_interview(session: InterviewSession) -> None:
    questions = st.session_state.get("interview_questions") or []
    idx = session.current_question_index
    if idx >= len(questions):
        _complete_interview(session)
        return
    q: InterviewQuestion = questions[idx]

    st.markdown(f"### Question {idx + 1} / {len(questions)}")
    col_cat, col_diff = st.columns(2)
    with col_cat:
        st.caption("Category")
        st.markdown(f"**{q.category.value.upper()}**")
    with col_diff:
        st.caption("Difficulty")
        st.markdown(f"**{q.difficulty.value.upper()}**")
    st.write(q.question)
    if q.target_skill:
        st.caption(f"Target skill: {q.target_skill}")
    if q.rationale:
        st.caption(f"Rationale: {q.rationale}")

    answer = st.text_area(
        "Your answer",
        height=180,
        key=f"interview_answer_{q.question_id}",
        placeholder="Write your answer here…",
    )
    submitted = st.button(
        "Submit Answer", type="primary", key=f"interview_submit_{q.question_id}",
        disabled=not (answer and answer.strip()),
    )
    if submitted:
        _submit_answer(session, q, answer)

    # Show last evaluation if this question was already answered.
    evals = st.session_state.get("interview_evaluations") or {}
    ev = evals.get(q.question_id)
    if ev is not None:
        _render_feedback(ev)
        if st.button("Next Question", type="primary", key=f"interview_next_{q.question_id}"):
            session.current_question_index = idx + 1
            st.session_state["interview_session"] = session
            st.rerun()


def _submit_answer(session: InterviewSession, q: InterviewQuestion, answer_text: str) -> None:
    resume = st.session_state.get("resume_analysis")
    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")
    with st.spinner("Evaluating your answer…"):
        try:
            client = _maybe_gemini_client()
            evaluation = evaluate_answer(
                q, answer_text, resume, job, match, client=client,
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error evaluating answer")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    st.session_state["interview_answers"][q.question_id] = InterviewAnswer(
        question_id=q.question_id, answer=answer_text, evidence_references=[],
    )
    st.session_state["interview_evaluations"][q.question_id] = evaluation
    st.rerun()


def _render_feedback(ev: AnswerEvaluation) -> None:
    section_header("Feedback", "Heuristic coaching scores (not psychometric).")
    col_o, col_e = st.columns([1, 3])
    with col_o:
        with st.container(border=True):
            st.caption("Overall")
            st.markdown(f"## {ev.overall_score:.1f} / 10")
    with col_e:
        st.progress(min(1.0, ev.overall_score / 10.0), text=f"{ev.overall_score:.1f}/10")
    cols = st.columns(5)
    for col, label, val in zip(
        cols,
        ("Relevance", "Technical", "Specificity", "Communication", "Completeness"),
        (ev.relevance_score, ev.technical_score, ev.specificity_score, ev.communication_score, ev.completeness_score),
    ):
        with col:
            with st.container(border=True):
                st.metric(label, f"{val:.1f}")
    if ev.strengths:
        st.caption("Strengths")
        for s in ev.strengths:
            st.markdown(f"- ✓ {s}")
    if ev.improvements:
        st.caption("Improve")
        for s in ev.improvements:
            st.markdown(f"- • {s}")
    if ev.unsupported_claims:
        st.warning("Unsupported claims detected — ground your answer in real experience:")
        for c in ev.unsupported_claims:
            st.markdown(f"- ⚠ {c}")
    if ev.recommendation:
        st.info(ev.recommendation)


# --------------------------------------------------------------------------- #
# Completion + report
# --------------------------------------------------------------------------- #


def _complete_interview(session: InterviewSession) -> None:
    questions = st.session_state.get("interview_questions") or []
    evals = st.session_state.get("interview_evaluations") or {}
    report = build_report(questions, evals, session.target_role)
    session.status = SessionStatus.COMPLETED
    session.readiness_summary = f"Readiness: {report.readiness_score:.0f}%"
    st.session_state["interview_session"] = session
    st.session_state["interview_report"] = report
    st.rerun()


def _render_report() -> None:
    report = st.session_state.get("interview_report")
    if not isinstance(report, InterviewReport):
        st.error("No interview report available.")
        return
    section_header("Interview Readiness", "Heuristic coaching scores — not a hiring prediction.")
    col_score, col_break = st.columns([1, 3])
    with col_score:
        with st.container(border=True):
            st.caption("Readiness")
            st.markdown(f"## {report.readiness_score:.0f}%")
    with col_break:
        st.progress(report.readiness_score / 100.0, text=f"{report.readiness_score:.0f}%")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.metric("Technical", f"{report.technical_score:.0f}/10")
    with c2:
        with st.container(border=True):
            st.metric("Behavioral", f"{report.behavioral_score:.0f}/10")
    with c3:
        with st.container(border=True):
            st.metric("Communication", f"{report.communication_score:.0f}/10")
    with c4:
        with st.container(border=True):
            st.metric("Job Alignment", f"{report.job_alignment_score:.0f}/10")

    if report.strengths:
        st.caption("Strong areas")
        for s in report.strengths:
            st.markdown(f"- {s}")
    if report.weaknesses:
        st.caption("Needs practice")
        for s in report.weaknesses:
            st.markdown(f"- {s}")
    if report.recommended_practice:
        st.caption("Recommended practice")
        for i, s in enumerate(report.recommended_practice, start=1):
            st.markdown(f"{i}. {s}")
    st.caption(f"Completed {report.completed_questions} / {report.total_questions} questions.")

    if st.button("Restart Interview", type="primary", key="interview_restart"):
        st.session_state["interview_session"] = InterviewSession(status=SessionStatus.NOT_STARTED)
        st.session_state["interview_questions"] = None
        st.session_state["interview_answers"] = {}
        st.session_state["interview_evaluations"] = {}
        st.session_state["interview_report"] = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _target_role(job) -> str:
    if not isinstance(job, JobAnalysis):
        return "the target role"
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")


def _maybe_gemini_client():
    try:
        from src.services.gemini import get_api_key

        get_api_key()
        from src.services.gemini import GeminiClient

        return GeminiClient()
    except GeminiConfigError:
        return None
    except Exception:
        _logger.exception("Unexpected error checking Gemini key")
        return None
