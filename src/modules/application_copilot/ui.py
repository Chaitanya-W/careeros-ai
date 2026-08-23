"""Application Copilot UI workflow.

Tabs: Resume Tailor / Cover Letter / Bullet Optimizer.

Each tab:
- Shows the deterministic, evidence-grounded output (always available).
- Offers an "Generate with AI" / "Regenerate" button when a Gemini key is
  configured; AI output is ALWAYS evidence-validated before display.
- FAIL-validated AI output is shown clearly as untrusted (with the
  unsupported claims listed) — never presented as trusted.

Prerequisites: resume_analysis + job_analysis + match_analysis (required).
skill_gap_report is used when available but not required.

State keys: evidence_items / application_drafts / application_copilot.
The resume / job / match / skill-gap / roadmap keys are never overwritten.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.application_copilot.copilot import (
    generate_cover_letter,
    optimize_bullet,
    tailor_resume,
)
from src.modules.application_copilot.evidence import extract_evidence
from src.modules.application_copilot.model import (
    BulletSuggestion,
    CoverLetterDraft,
    EvidenceItem,
    TailoredResumeSuggestion,
    ValidationStatus,
)
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while generating application material. "
    "Please try again."
)


def render() -> None:
    """Render the Application Copilot workflow."""
    section_header(
        "Application Copilot",
        "Create job-specific application material grounded in your actual "
        "resume evidence.",
    )

    resume, job, match = _load_prerequisites()
    if resume is None or job is None or match is None:
        _render_prerequisite_state(resume, job, match)
        return

    evidence = _ensure_evidence(resume)
    _render_status_bar(resume, job, match, evidence)

    tab_tailor, tab_cover, tab_bullet = st.tabs(
        ["Resume Tailor", "Cover Letter", "Bullet Optimizer"]
    )
    with tab_tailor:
        _render_tailor_tab(resume, job, match, evidence)
    with tab_cover:
        _render_cover_letter_tab(resume, job, match, evidence)
    with tab_bullet:
        _render_bullet_tab(resume, job, match, evidence)


# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #


def _load_prerequisites():
    resume = st.session_state.get("resume_analysis")
    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")
    if not isinstance(resume, ResumeAnalysis):
        resume = None
    if not isinstance(job, JobAnalysis):
        job = None
    if match is None or not hasattr(match, "matching_skills"):
        match = None
    return resume, job, match


def _render_prerequisite_state(resume, job, match) -> None:
    if resume is None:
        empty_state_panel(
            title="Please analyze your resume first.",
            hint="Application Copilot grounds every suggestion in your "
            "analyzed resume. Head to Resume Intelligence to upload and "
            "analyze your resume.",
        )
        if st.button(
            "Go to Resume Intelligence",
            type="primary",
            key="copilot_goto_resume",
        ):
            navigate_to("resume_intelligence")
        return
    if job is None or match is None:
        empty_state_panel(
            title="Analyze a target job first to personalize your application.",
            hint="Application Copilot personalizes material against a target "
            "job. Head to Job Match, paste a job description, and run the "
            "analysis.",
        )
        if st.button(
            "Go to Job Match",
            type="primary",
            key="copilot_goto_jobmatch",
        ):
            navigate_to("jd_analysis")
        return


# --------------------------------------------------------------------------- #
# Status bar + evidence cache
# --------------------------------------------------------------------------- #


def _ensure_evidence(resume: ResumeAnalysis) -> list[EvidenceItem]:
    cached = st.session_state.get("evidence_items")
    if isinstance(cached, list) and cached and all(isinstance(e, EvidenceItem) for e in cached):
        return cached
    evidence = extract_evidence(resume)
    st.session_state["evidence_items"] = evidence
    return evidence


def _render_status_bar(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> None:
    match_score = getattr(match, "overall_match_score", 0)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.container(border=True):
            st.caption("Target role")
            st.markdown(f"**{_target_role(job)}**")
    with col_b:
        with st.container(border=True):
            st.metric("Match score", f"{match_score}%")
    with col_c:
        with st.container(border=True):
            st.metric("Evidence items", len(evidence))
    _gemini_key_available()  # warm-check; rendered per-tab below.


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


# --------------------------------------------------------------------------- #
# Resume Tailor tab
# --------------------------------------------------------------------------- #


def _render_tailor_tab(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> None:
    section_header("Resume Tailor", "Evidence-grounded suggestions to emphasize relevant content.")

    drafts = st.session_state.get("application_drafts") or {}
    cached = drafts.get("tailor") if isinstance(drafts, dict) else None
    suggestions: Optional[list] = cached if isinstance(cached, list) else None

    key_available = _gemini_key_available()
    if not key_available:
        st.info(
            "AI wording refinement requires Gemini configuration. "
            "Deterministic, evidence-grounded suggestions are available below."
        )
        if st.button("Generate deterministic suggestions", key="copilot_tailor_det"):
            suggestions = tailor_resume(resume, job, match, evidence)
            _store_draft("tailor", suggestions)
            st.rerun()
    else:
        if suggestions is None:
            if st.button("Generate with AI", type="primary", key="copilot_tailor_ai"):
                suggestions = _run_tailor_ai(resume, job, match, evidence)
                if suggestions is not None:
                    _store_draft("tailor", suggestions)
                    st.rerun()
        else:
            if st.button("Regenerate", key="copilot_tailor_regen"):
                suggestions = _run_tailor_ai(resume, job, match, evidence)
                if suggestions is not None:
                    _store_draft("tailor", suggestions)
                    st.rerun()

    if not suggestions:
        empty_state_panel(
            title="No suggestions yet",
            hint="Generate tailored suggestions to emphasize your relevant skills and experience.",
        )
        return
    for s in suggestions:
        _render_suggestion(s, evidence)


def _run_tailor_ai(resume, job, match, evidence):
    with st.spinner("Generating tailored suggestions…"):
        try:
            from src.services.gemini import GeminiClient

            return tailor_resume(resume, job, match, evidence, client=GeminiClient())
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return None
        except Exception:
            _logger.exception("Unexpected error during resume tailoring")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return None


# --------------------------------------------------------------------------- #
# Cover Letter tab
# --------------------------------------------------------------------------- #


def _render_cover_letter_tab(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> None:
    section_header("Cover Letter", "A concise, evidence-grounded cover letter for the target role.")

    drafts = st.session_state.get("application_drafts") or {}
    cached = drafts.get("cover_letter") if isinstance(drafts, dict) else None
    draft: Optional[CoverLetterDraft] = cached if isinstance(cached, CoverLetterDraft) else None

    key_available = _gemini_key_available()
    if not key_available:
        st.info("AI cover-letter generation requires Gemini configuration.")
        if st.button("Show status", key="copilot_cover_det"):
            draft = generate_cover_letter(resume, job, match, evidence)
            _store_draft("cover_letter", draft)
            st.rerun()
    else:
        if draft is None or not draft.cover_letter:
            if st.button("Generate cover letter", type="primary", key="copilot_cover_ai"):
                draft = _run_cover_ai(resume, job, match, evidence)
                if draft is not None:
                    _store_draft("cover_letter", draft)
                    st.rerun()
        else:
            if st.button("Regenerate", key="copilot_cover_regen"):
                draft = _run_cover_ai(resume, job, match, evidence)
                if draft is not None:
                    _store_draft("cover_letter", draft)
                    st.rerun()

    if draft is None:
        empty_state_panel(
            title="No cover letter yet",
            hint="Generate an evidence-grounded cover letter for the target role.",
        )
        return
    _render_cover_letter(draft, evidence)


def _run_cover_ai(resume, job, match, evidence):
    with st.spinner("Generating cover letter…"):
        try:
            from src.services.gemini import GeminiClient

            return generate_cover_letter(resume, job, match, evidence, client=GeminiClient())
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return None
        except Exception:
            _logger.exception("Unexpected error during cover-letter generation")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return None


def _render_cover_letter(draft: CoverLetterDraft, evidence: list[EvidenceItem]) -> None:
    if draft.warnings:
        for w in draft.warnings:
            st.warning(w)
    if draft.cover_letter:
        st.write(draft.cover_letter)
    _render_validation(draft.evidence_validation, evidence, draft.evidence_ids)


# --------------------------------------------------------------------------- #
# Bullet Optimizer tab
# --------------------------------------------------------------------------- #


def _render_bullet_tab(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> None:
    section_header("Bullet Optimizer", "Optimize a resume bullet for the target role (preserving facts).")

    bullet = st.text_area(
        "Paste a resume bullet to optimize",
        height=80,
        key="copilot_bullet_input",
    )
    drafts = st.session_state.get("application_drafts") or {}
    cached = drafts.get("bullet") if isinstance(drafts, dict) else None
    suggestion: Optional[BulletSuggestion] = cached if isinstance(cached, BulletSuggestion) else None

    key_available = _gemini_key_available()
    can_run = bool(bullet and bullet.strip())
    if not key_available:
        st.info("AI bullet optimization requires Gemini configuration. Deterministic guidance is available.")
        if st.button("Get guidance", key="copilot_bullet_det", disabled=not can_run):
            suggestion = optimize_bullet(bullet, resume, job, match, evidence)
            _store_draft("bullet", suggestion)
            st.rerun()
    else:
        if suggestion is None or not suggestion.optimized_bullet:
            if st.button("Optimize with AI", type="primary", key="copilot_bullet_ai", disabled=not can_run):
                suggestion = _run_bullet_ai(bullet, resume, job, match, evidence)
                if suggestion is not None:
                    _store_draft("bullet", suggestion)
                    st.rerun()
        else:
            if st.button("Regenerate", key="copilot_bullet_regen"):
                suggestion = _run_bullet_ai(bullet, resume, job, match, evidence)
                if suggestion is not None:
                    _store_draft("bullet", suggestion)
                    st.rerun()

    if suggestion is None:
        empty_state_panel(
            title="No optimization yet",
            hint="Paste a bullet above and generate to see structured guidance or an AI-optimized version.",
        )
        return
    _render_bullet(suggestion, evidence)


def _run_bullet_ai(bullet, resume, job, match, evidence):
    with st.spinner("Optimizing bullet…"):
        try:
            from src.services.gemini import GeminiClient

            return optimize_bullet(bullet, resume, job, match, evidence, client=GeminiClient())
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return None
        except Exception:
            _logger.exception("Unexpected error during bullet optimization")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return None


def _render_bullet(suggestion: BulletSuggestion, evidence: list[EvidenceItem]) -> None:
    st.caption("Original bullet")
    st.write(suggestion.original_bullet)
    if suggestion.optimized_bullet:
        st.caption("Optimized bullet")
        st.write(suggestion.optimized_bullet)
    if suggestion.reason:
        st.caption("Reason")
        st.write(suggestion.reason)
    _render_validation(suggestion.validation, evidence, suggestion.evidence_ids)


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #


def _render_suggestion(s: TailoredResumeSuggestion, evidence: list[EvidenceItem]) -> None:
    with st.container(border=True):
        st.markdown(f"### {s.section.title()}")
        if s.original_content:
            st.caption("Original")
            st.write(s.original_content)
        if s.suggested_content:
            st.caption("Suggested")
            st.write(s.suggested_content)
        if s.reason:
            st.caption("Reason")
            st.write(s.reason)
        _render_validation(s.validation, evidence, s.evidence_ids)


def _render_validation(
    validation,
    evidence: list[EvidenceItem],
    evidence_ids: list[str],
) -> None:
    if evidence_ids:
        st.caption("Supporting evidence")
        by_id = {e.evidence_id: e for e in evidence}
        for eid in evidence_ids:
            ev = by_id.get(eid)
            if ev:
                st.markdown(f"- **{eid}** [{ev.source_type}]: {ev.source_text}")
            else:
                st.markdown(f"- **{eid}** (evidence not found)")
    if validation is None:
        return
    status = validation.status
    if status is ValidationStatus.PASS:
        st.success(f"Evidence status: PASS")
    elif status is ValidationStatus.WARNING:
        st.warning("Evidence status: WARNING — some claims could not be grounded.")
        for w in validation.warnings:
            st.markdown(f"- {w}")
    else:  # FAIL
        st.error("Evidence status: FAIL — unsupported claims detected (output is untrusted).")
        for u in validation.unsupported_claims:
            st.markdown(f"- {u}")
    if validation.supported_claims:
        with st.expander("Supported claims", expanded=False):
            for s in validation.supported_claims:
                st.markdown(f"- {s}")


def _store_draft(key: str, value) -> None:
    drafts = st.session_state.get("application_drafts") or {}
    if not isinstance(drafts, dict):
        drafts = {}
    drafts[key] = value
    st.session_state["application_drafts"] = drafts


def _target_role(job: JobAnalysis) -> str:
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")
