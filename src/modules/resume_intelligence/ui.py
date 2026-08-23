"""Resume Intelligence UI workflow.

Flow: upload -> validate -> extract -> analyze -> results / empty state.

Extraction happens locally on every upload; the extracted text is cached
in ``session_state["resume_text"]`` so it survives reruns and is available
to other CareerOS modules. The successful :class:`ResumeAnalysis` is
cached in ``session_state["resume_analysis"]`` for the same reason.

The UI never imports the Gemini SDK directly — it calls the
:func:`analyze_resume` service and maps typed errors to friendly messages.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src.modules.resume_intelligence.analyzer import (
    EmptyResumeError,
    analyze_resume,
)
from src.modules.resume_intelligence.extractor import (
    ExtractionError,
    extract_text,
)
from src.modules.resume_intelligence.model import (
    ResumeAnalysis,
    ResumeAnalysisError,
)
from src.services.gemini import GeminiError
from src.ui.components.cards import empty_state_panel, section_header

_UPLOAD_KEY: str = "resume_uploader"
_PREVIEW_LIMIT: int = 1000


def render() -> None:
    """Render the Resume Intelligence workflow."""
    _render_uploader()
    _render_extracted_preview()
    _render_analyze_action()
    _render_results_or_empty_state()


def _render_uploader() -> None:
    section_header("Upload your resume", "PDF, DOCX, or TXT — extracted locally.")
    uploaded = st.file_uploader(
        "Choose a resume file",
        type=["pdf", "docx", "txt"],
        key=_UPLOAD_KEY,
        help=(
            "We extract text locally on your machine before sending it to "
            "Gemini for analysis."
        ),
    )
    if uploaded is None:
        # User cleared the upload — drop any stale cached text/analysis.
        if st.session_state.get("resume_text"):
            st.session_state["resume_text"] = None
            st.session_state["resume_analysis"] = None
            st.session_state["_resume_file_sig"] = None
        return

    # Re-extract only when the uploaded file actually changes, so reruns
    # (e.g. clicking Analyze) do not re-parse the file every time.
    signature = (uploaded.name, getattr(uploaded, "size", 0))
    if (
        st.session_state.get("_resume_file_sig") == signature
        and st.session_state.get("resume_text")
    ):
        return

    try:
        text = extract_text(uploaded.name, uploaded.getvalue())
    except ExtractionError as e:
        st.session_state["resume_text"] = None
        st.session_state["resume_analysis"] = None
        st.session_state["_resume_file_sig"] = None
        st.error(str(e))
        return

    st.session_state["resume_text"] = text
    st.session_state["resume_analysis"] = None  # invalidate prior result
    st.session_state["_resume_file_sig"] = signature
    st.success(f"Extracted {len(text):,} characters from `{uploaded.name}`.")


def _render_extracted_preview() -> None:
    text: Optional[str] = st.session_state.get("resume_text")
    if not text:
        return
    with st.expander("Extracted text preview", expanded=False):
        preview = text[:_PREVIEW_LIMIT]
        st.text(preview + ("…" if len(text) > _PREVIEW_LIMIT else ""))
        st.caption(
            f"Showing first {len(preview):,} of {len(text):,} characters."
        )


def _render_analyze_action() -> None:
    if not st.session_state.get("resume_text"):
        return
    section_header("Analyze", "Run the AI analysis on your extracted resume.")
    if st.button("Analyze Resume", type="primary", key="analyze_resume_btn"):
        _run_analysis()


def _run_analysis() -> None:
    text: Optional[str] = st.session_state.get("resume_text")
    if not text:
        st.error("No resume text to analyze. Please upload a resume first.")
        return
    with st.spinner("Analyzing your resume with Gemini…"):
        try:
            analysis: ResumeAnalysis = analyze_resume(text)
        except EmptyResumeError as e:
            st.error(str(e))
            return
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except ResumeAnalysisError as e:
            st.error(f"Could not interpret the analysis: {e}")
            return
        except Exception as e:  # last-resort guard, never show a traceback
            st.error(f"Unexpected error during analysis: {e}")
            return
    st.session_state["resume_analysis"] = analysis
    st.success("Resume analysis complete.")
    st.rerun()


def _render_results_or_empty_state() -> None:
    analysis: Optional[ResumeAnalysis] = st.session_state.get("resume_analysis")
    if analysis is None:
        empty_state_panel(
            title="No resume analyzed yet",
            hint=(
                "Upload a PDF, DOCX, or TXT resume above, then click "
                "Analyze to get a structured AI analysis."
            ),
        )
        return
    _render_results(analysis)


def _render_results(analysis: ResumeAnalysis) -> None:
    section_header("Analysis Results", "Structured breakdown of your resume.")
    _score_card(analysis.overall_score)

    _render_summary(analysis.professional_summary)
    _render_string_list("Skills", analysis.skills)
    _render_experience(analysis.experience)
    _render_education(analysis.education)
    _render_projects(analysis.projects)
    _render_certifications(analysis.certifications)
    _render_string_list("Strengths", analysis.strengths)
    _render_string_list("Areas for Improvement", analysis.weaknesses)
    _render_string_list("Improvement Suggestions", analysis.improvement_suggestions)
    _render_string_list("Recommended Skills", analysis.recommended_skills)


def _score_card(score: int) -> None:
    """Overall score with a numeric card + progress bar."""
    col_metric, col_bar = st.columns([1, 3])
    with col_metric:
        with st.container(border=True):
            st.caption("Overall Resume Score")
            st.markdown(f"## {score} / 100")
    with col_bar:
        st.progress(score / 100.0, text=f"{score}%")


def _render_summary(summary: Optional[str]) -> None:
    st.subheader("Professional Summary")
    if summary:
        st.write(summary)
    else:
        st.caption("No summary could be derived from the resume.")


def _render_string_list(title: str, items: list[str]) -> None:
    st.subheader(title)
    if not items:
        st.caption("No items found.")
        return
    for item in items:
        st.markdown(f"- {item}")


def _render_experience(items: list) -> None:
    st.subheader("Experience")
    if not items:
        st.caption("No experience entries found.")
        return
    for idx, item in enumerate(items, start=1):
        with st.container(border=True):
            header = " · ".join(
                p for p in (item.title, item.company, item.duration) if p
            ) or f"Entry {idx}"
            st.markdown(f"**{header}**")
            if item.description:
                st.write(item.description)


def _render_education(items: list) -> None:
    st.subheader("Education")
    if not items:
        st.caption("No education entries found.")
        return
    for item in items:
        with st.container(border=True):
            header = " · ".join(
                p for p in (item.degree, item.institution, item.year) if p
            )
            st.markdown(f"**{header or 'Education entry'}**")
            if item.details:
                st.write(item.details)


def _render_projects(items: list) -> None:
    st.subheader("Projects")
    if not items:
        st.caption("No projects found.")
        return
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.name or 'Untitled project'}**")
            if item.description:
                st.write(item.description)
            if item.technologies:
                st.caption(f"Technologies: {item.technologies}")


def _render_certifications(items: list) -> None:
    st.subheader("Certifications")
    if not items:
        st.caption("No certifications found.")
        return
    for item in items:
        header = " · ".join(
            p for p in (item.name, item.issuer, item.year) if p
        )
        st.markdown(f"- {header}")
