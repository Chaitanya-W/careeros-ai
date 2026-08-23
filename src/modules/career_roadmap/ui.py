"""Career Roadmap UI workflow.

Flow:
1. Require resume_analysis + job_analysis + match_analysis +
   skill_gap_report in session state. If any is missing, show a friendly
   prerequisite message + a navigation button. Gemini is NEVER called when
   prerequisites are missing.
2. Build the deterministic :class:`CareerRoadmap` (no AI). Cache it in
   session_state keyed to the current skill-gap signature.
3. AI enrichment is OPTIONAL and user-triggered:
   - If no Gemini key is configured, show the deterministic roadmap + a
     clear message that AI enrichment requires Gemini.
   - If a key is configured, a button generates/regenerates AI wording.
4. Render the roadmap: target role, readiness, total effort, phase count,
   then each phase in order.

State keys (see ``src/core/state.py``): ``career_roadmap``. The resume /
job / match / skill-gap keys are reused, never overwritten.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.career_roadmap.builder import build_career_roadmap
from src.modules.career_roadmap.enricher import (
    RoadmapEnrichmentError,
    apply_enrichment,
    enrich_roadmap,
)
from src.modules.career_roadmap.model import CareerRoadmap
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import SkillGapReport
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while generating AI roadmap enrichment. "
    "Please try again."
)


def render() -> None:
    """Render the Career Roadmap workflow."""
    section_header(
        "Career Roadmap",
        "A personalized learning sequence based on your target role and "
        "identified skill gaps.",
    )

    resume, job, match, skill_gap_report = _load_prerequisites()
    if (
        resume is None
        or job is None
        or match is None
        or skill_gap_report is None
    ):
        _render_prerequisite_state(resume, job, match, skill_gap_report)
        return

    roadmap = _ensure_roadmap(skill_gap_report, match, job)

    if not roadmap.phases:
        empty_state_panel(
            title="No skill gaps to roadmap.",
            hint=(
                "Your resume currently covers the identified requirements. "
                "No major skill gaps were detected for this target role."
            ),
        )
        return

    _render_summary(roadmap)
    _render_enrichment_actions(roadmap, job, resume)
    _render_phases(roadmap)


# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #


def _load_prerequisites():
    resume = st.session_state.get("resume_analysis")
    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")
    report = st.session_state.get("skill_gap_report")
    if not isinstance(resume, ResumeAnalysis):
        resume = None
    if not isinstance(job, JobAnalysis):
        job = None
    if match is None or not hasattr(match, "overall_match_score"):
        match = None
    if not isinstance(report, SkillGapReport):
        report = None
    return resume, job, match, report


def _render_prerequisite_state(resume, job, match, report) -> None:
    if resume is None:
        empty_state_panel(
            title="Please analyze your resume first.",
            hint="Career Roadmap builds on your analyzed resume, job match, "
            "and skill-gap report. Head to Resume Intelligence to upload "
            "and analyze your resume.",
        )
        if st.button(
            "Go to Resume Intelligence",
            type="primary",
            key="roadmap_goto_resume",
        ):
            navigate_to("resume_intelligence")
        return
    if job is None or match is None:
        empty_state_panel(
            title="Analyze a target job first.",
            hint="Career Roadmap needs your Job Match results. Head to Job "
            "Match, paste a job description, and run the analysis.",
        )
        if st.button(
            "Go to Job Match",
            type="primary",
            key="roadmap_goto_jobmatch",
        ):
            navigate_to("jd_analysis")
        return
    if report is None:
        empty_state_panel(
            title="Analyze your skill gaps first.",
            hint="Career Roadmap is built from your Skill Gap report. Head "
            "to Skill Gap Analysis to generate it.",
        )
        if st.button(
            "Go to Skill Gap",
            type="primary",
            key="roadmap_goto_skillgap",
        ):
            navigate_to("skill_gap")
        return


# --------------------------------------------------------------------------- #
# Roadmap build / cache
# --------------------------------------------------------------------------- #


def _ensure_roadmap(
    report: SkillGapReport,
    match,
    job: JobAnalysis,
) -> CareerRoadmap:
    """Return a cached roadmap for the current inputs, or build a new one."""
    signature = _signature(report)
    cached: Optional[CareerRoadmap] = st.session_state.get("career_roadmap")
    if (
        isinstance(cached, CareerRoadmap)
        and getattr(cached, "_source_signature", None) == signature
    ):
        return cached
    roadmap = build_career_roadmap(report, match, job)
    object.__setattr__(roadmap, "_source_signature", signature)
    st.session_state["career_roadmap"] = roadmap
    return roadmap


def _signature(report: SkillGapReport) -> tuple:
    skills = tuple(g.skill for g in report.skill_gaps)
    role = report.target_role
    return (skills, role)


# --------------------------------------------------------------------------- #
# AI enrichment actions
# --------------------------------------------------------------------------- #


def _render_enrichment_actions(
    roadmap: CareerRoadmap,
    job: JobAnalysis,
    resume: ResumeAnalysis,
) -> None:
    section_header(
        "AI enrichment",
        "Optional: refine roadmap wording with AI-generated guidance.",
    )
    key_available = _gemini_key_available()
    if not key_available:
        st.info(
            "AI roadmap enrichment requires Gemini configuration. The "
            "deterministic roadmap above is fully available without a key. "
            "Add a `[gemini]` API key to `.streamlit/secrets.toml` to unlock "
            "AI wording refinement."
        )
        return
    if roadmap.enriched:
        if st.button(
            "Regenerate AI enrichment",
            type="secondary",
            key="roadmap_regen_enrichment",
        ):
            _run_enrichment(roadmap, job, resume)
    else:
        if st.button(
            "Generate AI roadmap enrichment",
            type="primary",
            key="roadmap_gen_enrichment",
        ):
            _run_enrichment(roadmap, job, resume)


def _gemini_key_available() -> bool:
    try:
        from src.services.gemini import get_api_key

        get_api_key()
        return True
    except GeminiConfigError:
        return False
    except Exception:
        _logger.exception("Unexpected error while checking Gemini key")
        return False


def _run_enrichment(roadmap, job, resume) -> None:
    with st.spinner("Generating AI roadmap enrichment…"):
        try:
            enrichment = enrich_roadmap(roadmap, job=job, resume=resume)
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except RoadmapEnrichmentError as e:
            st.error(f"Could not apply AI enrichment: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error during roadmap enrichment")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    apply_enrichment(roadmap, enrichment)
    st.session_state["career_roadmap"] = roadmap
    st.success("AI roadmap enrichment applied.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_summary(roadmap: CareerRoadmap) -> None:
    section_header("Roadmap summary", None)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        with st.container(border=True):
            st.caption("Target role")
            st.markdown(f"**{roadmap.target_role}**")
    with col_b:
        with st.container(border=True):
            st.metric("Current readiness", f"{roadmap.current_readiness}%")
    with col_c:
        with st.container(border=True):
            st.metric("Phases", roadmap.phase_count)
    with col_d:
        with st.container(border=True):
            st.caption("Estimated effort")
            st.write(roadmap.estimated_total_effort)
    st.progress(
        roadmap.current_readiness / 100.0,
        text=f"Readiness {roadmap.current_readiness}%",
    )
    if roadmap.enriched:
        st.caption("AI enrichment: applied.")
    else:
        st.caption("AI enrichment: not yet generated.")


def _render_phases(roadmap: CareerRoadmap) -> None:
    section_header("Learning phases", "Ordered, dependency-aware skill sequence.")
    for phase in roadmap.phases:
        _render_phase(phase)


def _render_phase(phase) -> None:
    with st.container(border=True):
        st.markdown(f"### {phase.title}")
        st.caption(phase.duration)
        st.write(phase.objective)

        skill_col, prereq_col = st.columns(2)
        with skill_col:
            st.caption("Skills")
            for s in phase.skills:
                st.markdown(f"- {s}")
        with prereq_col:
            st.caption("Prerequisites")
            if phase.prerequisites:
                for p in phase.prerequisites:
                    st.markdown(f"- {p}")
            else:
                st.write("None — start here.")

        if phase.milestones:
            st.caption("Milestones")
            for ms in phase.milestones:
                _render_milestone(ms)

        if phase.practice_project:
            st.caption("Practice project")
            st.write(phase.practice_project)


def _render_milestone(ms) -> None:
    with st.container(border=True):
        st.markdown(f"**{ms.title}**")
        if ms.skill:
            st.caption(f"Skill: {ms.skill}")
        if ms.description:
            st.write(ms.description)
        if ms.completion_criteria:
            st.caption("Completion criteria")
            st.write(ms.completion_criteria)
