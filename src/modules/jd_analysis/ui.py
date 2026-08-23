"""Job Match UI workflow.

Flow:
1. Require an existing :class:`ResumeAnalysis` (from Step 3). If absent,
   show a clear "Please analyze your resume first." message + a button
   that navigates to Resume Intelligence. Gemini is NEVER called when
   there is no valid resume analysis.
2. Accept a pasted job description in a large text area, with input
   validation (empty / too short / too long / whitespace-only).
3. Cache: if a :class:`JobAnalysis` already exists for the exact same
   job description, reuse it (no Gemini call) unless the user explicitly
   requests re-analysis.
4. Run the deterministic :func:`match_jobs` matcher and cache the
   :class:`MatchAnalysis`.
5. Render the results dashboard.

State keys (see ``src/core/state.py``):
``job_description`` / ``job_analysis`` / ``match_analysis``. The existing
``resume_analysis`` is reused, never overwritten.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.jd_analysis.analyzer import (
    EmptyJobDescriptionError,
    analyze_job_description,
)
from src.modules.jd_analysis.matcher import match_jobs
from src.modules.jd_analysis.model import JobAnalysis, JobAnalysisError
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.services.gemini import GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

# Module-level logger. ``logger.exception`` writes the traceback to server
# logs (file paths / stack traces) for debugging — it does NOT print local
# variables, so resume / job-description text and API keys are never logged.
_logger = logging.getLogger(__name__)

_TEXT_AREA_KEY: str = "job_description_input"

# Generic message shown for unexpected internal errors. Typed errors
# (EmptyJobDescriptionError / GeminiError / JobAnalysisError) keep their
# specific user-facing messages.
_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while analyzing the job description. "
    "Please try again."
)


def render() -> None:
    """Render the Job Match workflow."""
    resume = st.session_state.get("resume_analysis")

    if not isinstance(resume, ResumeAnalysis):
        _render_no_resume_state()
        return

    _render_jd_input(resume)
    _render_results_or_empty(resume)


# --------------------------------------------------------------------------- #
# No-resume guard
# --------------------------------------------------------------------------- #


def _render_no_resume_state() -> None:
    section_header("Job Match", "Compare a job description against your resume.")
    empty_state_panel(
        title="Please analyze your resume first.",
        hint=(
            "Job Match compares a pasted job description against your "
            "already-analyzed resume. Head to Resume Intelligence, upload "
            "and analyze your resume, then come back here."
        ),
    )
    if st.button(
        "Go to Resume Intelligence",
        type="primary",
        key="jobmatch_goto_resume",
        use_container_width=False,
    ):
        navigate_to("resume_intelligence")


# --------------------------------------------------------------------------- #
# Job-description input
# --------------------------------------------------------------------------- #


def _render_jd_input(resume: ResumeAnalysis) -> None:
    section_header(
        "Job description",
        "Paste the full job description text, then click Analyze.",
    )
    st.text_area(
        "Job description",
        height=240,
        key=_TEXT_AREA_KEY,
        help=(
            "We extract skills/requirements via Gemini, then deterministically "
            "compare them against your resume. Your resume text is not re-sent."
        ),
    )

    col_analyze, col_reanalyze = st.columns([1, 1])
    with col_analyze:
        analyze_clicked = st.button(
            "Analyze Job Match",
            type="primary",
            key="jobmatch_analyze",
            use_container_width=True,
        )
    with col_reanalyze:
        has_cached = st.session_state.get("job_analysis") is not None
        reanalyze_clicked = st.button(
            "Re-analyze",
            type="secondary",
            key="jobmatch_reanalyze",
            use_container_width=True,
            disabled=not has_cached,
            help="Force a fresh Gemini extraction of the job description.",
        )

    if analyze_clicked or reanalyze_clicked:
        _run_analysis(resume, force=reanalyze_clicked)


def _run_analysis(resume: ResumeAnalysis, *, force: bool) -> None:
    text: str = st.session_state.get(_TEXT_AREA_KEY, "") or ""

    # Cached path: same JD as last time and not forced -> skip Gemini.
    cached_jd: Optional[str] = st.session_state.get("job_description")
    cached_job: Optional[JobAnalysis] = st.session_state.get("job_analysis")
    if (
        not force
        and cached_job is not None
        and isinstance(cached_jd, str)
        and cached_jd.strip() == text.strip()
        and text.strip()
    ):
        st.session_state["match_analysis"] = match_jobs(resume, cached_job)
        st.success("Reusing the cached job analysis for this description.")
        st.rerun()
        return

    try:
        job = analyze_job_description(text)
    except EmptyJobDescriptionError as e:
        st.error(str(e))
        return
    except GeminiError as e:
        st.error(f"Gemini error: {e}")
        return
    except JobAnalysisError as e:
        st.error(f"Could not interpret the job analysis: {e}")
        return
    except Exception:
        # Catch-all: NEVER expose internal details (stack traces, SDK
        # internals, file paths, configuration, exception messages) to the
        # user. Log safely for debugging — the traceback never includes
        # local variables, so API keys and resume/JD text are not logged.
        _logger.exception("Unexpected error during Job Match analysis")
        st.error(_UNEXPECTED_ERROR_MESSAGE)
        return

    st.session_state["job_description"] = text
    st.session_state["job_analysis"] = job
    st.session_state["match_analysis"] = match_jobs(resume, job)
    st.success("Job analysis complete.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Results / empty state
# --------------------------------------------------------------------------- #


def _render_results_or_empty(resume: ResumeAnalysis) -> None:
    match = st.session_state.get("match_analysis")
    job = st.session_state.get("job_analysis")
    if match is None or not isinstance(job, JobAnalysis):
        empty_state_panel(
            title="No job match yet",
            hint=(
                "Paste a job description above and click Analyze to see "
                "your match score, matching/missing skills, and a "
                "deterministic gap analysis."
            ),
        )
        return
    _render_job_summary(job)
    _render_match_results(match)


def _render_job_summary(job: JobAnalysis) -> None:
    section_header("Analyzed role", "What we extracted from the job description.")
    if job.job_title or job.company or job.domain:
        header = " · ".join(p for p in (job.job_title, job.company, job.domain) if p)
        st.markdown(f"**{header}**")
    _render_string_list("Required skills", job.required_skills)
    _render_string_list("Preferred skills", job.preferred_skills)
    _render_string_list("Responsibilities", job.responsibilities)
    if job.experience_requirements:
        st.markdown("**Experience:** " + job.experience_requirements)
    if job.education_requirements:
        st.markdown("**Education:** " + job.education_requirements)
    _render_string_list("Keywords", job.keywords)


def _render_match_results(match) -> None:
    section_header("Match Results", "Deterministic comparison against your resume.")
    _score_card(match.overall_match_score)

    _component_metrics(
        match.experience_match, match.education_match, match.keyword_match
    )

    _render_string_list("Matching skills", match.matching_skills)
    _render_string_list("Missing skills", match.missing_skills)
    _render_string_list("Partial / related skills", match.partial_match_skills)
    _render_string_list("Strengths", match.strengths)
    _render_string_list("Skill / experience gaps", match.gaps)
    if match.explanation:
        st.subheader("Explanation")
        st.write(match.explanation)
    _render_string_list("Recommended actions", match.recommended_actions)


def _score_card(score: int) -> None:
    col_metric, col_bar = st.columns([1, 3])
    with col_metric:
        with st.container(border=True):
            st.caption("Overall Match Score")
            st.markdown(f"## {score} / 100")
    with col_bar:
        st.progress(score / 100.0, text=f"{score}%")


def _component_metrics(experience: int, education: int, keywords: int) -> None:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.container(border=True):
            st.metric("Experience match", f"{experience}%")
    with col_b:
        with st.container(border=True):
            st.metric("Education match", f"{education}%")
    with col_c:
        with st.container(border=True):
            st.metric("Keyword alignment", f"{keywords}%")


def _render_string_list(title: str, items: list[str]) -> None:
    st.subheader(title)
    if not items:
        st.caption("No items found.")
        return
    for item in items:
        st.markdown(f"- {item}")
