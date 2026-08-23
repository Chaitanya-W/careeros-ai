"""Skill Gap UI workflow.

Flow:
1. Require resume_analysis + job_analysis + match_analysis in session
   state. If any is missing, show a friendly prerequisite message + a
   navigation button. Gemini is NEVER called when prerequisites are
   missing.
2. Build the deterministic :class:`SkillGapReport` (no AI). Cache it in
   session_state keyed to the current missing-skills signature so repeated
   views don't rebuild.
3. AI enrichment is OPTIONAL and user-triggered:
   - If no Gemini key is configured, show the deterministic report + a
     clear message that AI learning recommendations require configuration.
   - If a key is configured, a button generates/regenerates AI learning
     guidance. Repeated clicks re-call Gemini only on explicit request.
4. Render the report: summary metrics + a card per skill gap.

State keys (see ``src/core/state.py``): ``skill_gap_report``. The resume /
job / match keys are reused, never overwritten.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.enricher import (
    SkillGapEnrichmentError,
    apply_enrichment,
    enrich_skill_gaps,
)
from src.modules.skill_gap.model import Priority, SkillGapReport
from src.modules.skill_gap.report import build_skill_gap_report
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while generating AI learning recommendations. "
    "Please try again."
)


def render() -> None:
    """Render the Skill Gap workflow."""
    section_header(
        "Skill Gap Analysis",
        "CareerOS identifies the skills the target role requires that are "
        "not sufficiently represented in your resume.",
    )

    resume, job, match = _load_prerequisites()
    if resume is None or job is None or match is None:
        _render_prerequisite_state(resume, job, match)
        return

    report = _ensure_report(resume, job, match)

    if not report.skill_gaps:
        empty_state_panel(
            title="No skill gaps detected",
            hint=(
                "Your resume covers the required and preferred skills "
                "for this role. Great alignment!"
            ),
        )
        return

    _render_summary_metrics(report)
    _render_enrichment_actions(report, resume, job)
    _render_skill_gaps(report)


# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #


def _load_prerequisites():
    """Return (resume, job, match) or (None, None, None)-ish for missing."""
    resume = st.session_state.get("resume_analysis")
    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")
    if not isinstance(resume, ResumeAnalysis):
        resume = None
    if not isinstance(job, JobAnalysis):
        job = None
    if match is None or not hasattr(match, "missing_skills"):
        match = None
    return resume, job, match


def _render_prerequisite_state(resume, job, match) -> None:
    """Friendly message + navigation for missing prerequisites."""
    if resume is None:
        empty_state_panel(
            title="Please analyze your resume first.",
            hint="Skill Gap Analysis compares your analyzed resume against a "
            "target job. Head to Resume Intelligence to upload and analyze "
            "your resume.",
        )
        if st.button(
            "Go to Resume Intelligence",
            type="primary",
            key="skillgap_goto_resume",
        ):
            navigate_to("resume_intelligence")
        return
    if job is None or match is None:
        empty_state_panel(
            title="Analyze a target job first to identify your skill gaps.",
            hint="Skill Gap Analysis uses your Job Match results. Head to "
            "Job Match, paste a job description, and run the analysis.",
        )
        if st.button(
            "Go to Job Match",
            type="primary",
            key="skillgap_goto_jobmatch",
        ):
            navigate_to("jd_analysis")
        return


# --------------------------------------------------------------------------- #
# Report build / cache
# --------------------------------------------------------------------------- #


def _ensure_report(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
) -> SkillGapReport:
    """Return a cached report for the current inputs, or build a new one."""
    signature = _signature(match, job)
    cached: Optional[SkillGapReport] = st.session_state.get("skill_gap_report")
    if (
        isinstance(cached, SkillGapReport)
        and getattr(cached, "_source_signature", None) == signature
    ):
        return cached
    report = build_skill_gap_report(resume, job, match)
    # Stash the source signature on the report for cache validation. This
    # is a private, non-serialized attribute (not part of the schema).
    object.__setattr__(report, "_source_signature", signature)
    st.session_state["skill_gap_report"] = report
    return report


def _signature(match, job) -> tuple:
    """A cheap signature of the inputs that drive the gap list."""
    missing = tuple(getattr(match, "missing_skills", []) or [])
    preferred = tuple(job.preferred_skills or [])
    required = tuple(job.required_skills or [])
    return (missing, preferred, required)


# --------------------------------------------------------------------------- #
# AI enrichment actions
# --------------------------------------------------------------------------- #


def _render_enrichment_actions(
    report: SkillGapReport,
    resume: ResumeAnalysis,
    job: JobAnalysis,
) -> None:
    section_header(
        "AI learning recommendations",
        "Optional: enrich each gap with AI-generated learning guidance.",
    )

    key_available = _gemini_key_available()
    if not key_available:
        st.info(
            "AI learning recommendations require Gemini configuration. "
            "The deterministic skill-gap information above is fully "
            "available without a key. Add a `[gemini]` API key to "
            "`.streamlit/secrets.toml` to unlock AI guidance."
        )
        return

    if report.enriched:
        if st.button(
            "Regenerate AI enrichment",
            type="secondary",
            key="skillgap_regen_enrichment",
        ):
            _run_enrichment(report, resume, job)
    else:
        if st.button(
            "Generate AI learning recommendations",
            type="primary",
            key="skillgap_gen_enrichment",
        ):
            _run_enrichment(report, resume, job)


def _gemini_key_available() -> bool:
    """True if a Gemini API key is configured (without raising)."""
    try:
        from src.services.gemini import get_api_key

        get_api_key()
        return True
    except GeminiConfigError:
        return False
    except Exception:  # never let a key-check crash the page
        _logger.exception("Unexpected error while checking Gemini key")
        return False


def _run_enrichment(
    report: SkillGapReport,
    resume: ResumeAnalysis,
    job: JobAnalysis,
) -> None:
    with st.spinner("Generating AI learning recommendations…"):
        try:
            enrichment = enrich_skill_gaps(
                report.skill_gaps, job=job, resume=resume
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except SkillGapEnrichmentError as e:
            st.error(f"Could not interpret the AI recommendations: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error during skill-gap enrichment")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    apply_enrichment(report, enrichment)
    st.session_state["skill_gap_report"] = report
    st.success("AI learning recommendations applied.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_summary_metrics(report: SkillGapReport) -> None:
    section_header("Summary", report.summary)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        with st.container(border=True):
            st.metric("Total Skill Gaps", report.total_gaps)
    with col_b:
        with st.container(border=True):
            st.metric("High Priority", report.high_priority_count)
    with col_c:
        with st.container(border=True):
            st.metric("Medium Priority", report.medium_priority_count)
    with col_d:
        with st.container(border=True):
            st.metric("Low Priority", report.low_priority_count)
    if report.enriched:
        st.caption("AI learning recommendations: applied.")
    else:
        st.caption("AI learning recommendations: not yet generated.")


def _render_skill_gaps(report: SkillGapReport) -> None:
    section_header("Skill gaps", "One card per gap, in priority order.")
    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    for gap in sorted(report.skill_gaps, key=lambda g: order.get(g.priority, 9)):
        _render_gap_card(gap)


def _render_gap_card(gap) -> None:
    with st.container(border=True):
        header_col, badge_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"### {gap.skill}")
        with badge_col:
            _priority_badge(gap.priority)

        meta_a, meta_b = st.columns(2)
        with meta_a:
            st.caption("Current evidence")
            st.write(gap.current_evidence)
            st.caption("Current level")
            st.write(gap.current_level or "No evidence found in resume")
        with meta_b:
            st.caption("Target proficiency")
            st.write(gap.target_level)
            st.caption("Gap reason")
            st.write(gap.gap_reason)

        if gap.why_it_matters or gap.learning_objectives or gap.learning_path:
            st.caption("Why it matters")
            st.write(gap.why_it_matters or "—")
            _render_list("Learning objectives", gap.learning_objectives)
            _render_list("Learning path", gap.learning_path)
            if gap.practice_project:
                st.caption("Practice project")
                st.write(gap.practice_project)
            if gap.estimated_effort:
                st.caption("Estimated effort")
                st.write(gap.estimated_effort)
        elif gap.priority is not Priority.LOW:
            st.caption(
                "AI learning recommendations not yet generated for this gap."
            )


def _priority_badge(priority: Priority) -> None:
    label = priority.value.upper()
    if priority is Priority.HIGH:
        st.error(label)
    elif priority is Priority.MEDIUM:
        st.warning(label)
    else:
        st.info(label)


def _render_list(title: str, items: list[str]) -> None:
    if not items:
        return
    st.caption(title)
    for item in items:
        st.markdown(f"- {item}")
