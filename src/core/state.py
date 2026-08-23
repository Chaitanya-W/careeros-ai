"""Centralized Streamlit session-state management.

Thin wrapper around ``st.session_state`` so that state access is
predictable, type-friendly and refactor-proof as the app grows. All
default keys are declared in one place (:data:`_DEFAULTS`).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import MODULES

# Default values for the application's session-state keys.
# Add new keys here as the application evolves.
_DEFAULTS: dict[str, Any] = {
    "selected_module": MODULES[0]["key"],
    # Resume Intelligence — cached analysis + extracted text so other
    # modules can reuse them within the session.
    "resume_analysis": None,
    "resume_text": None,
    # Job Match — cached job description + job analysis + match analysis.
    # resume_analysis is NOT overwritten by this module.
    "job_description": None,
    "job_analysis": None,
    "match_analysis": None,
    # Skill Gap — cached report (built deterministically; AI-enriched on
    # demand). The keys above are never overwritten by this module.
    "skill_gap_report": None,
    # Career Roadmap — built deterministically from the SkillGapReport;
    # AI-enriched on demand. The keys above are never overwritten.
    "career_roadmap": None,
    # Application Copilot — cached evidence items, generated drafts, and a
    # general copilot state object. The keys above are never overwritten.
    "evidence_items": None,
    "application_drafts": None,
    "application_copilot": None,
    # RAG Career Assistant — indexed documents, chunks, vector index,
    # conversation messages, and the last answer. The keys above are never
    # overwritten.
    "rag_documents": None,  # dict: document_id -> RAGDocument
    "rag_chunks": None,  # dict: document_id -> list[DocumentChunk]
    "rag_index": None,  # VectorIndex instance (or None)
    "rag_messages": None,  # list of (role, text) tuples
    "rag_last_answer": None,  # last RAGAnswer
    # AI Interview Coach — session, questions, answers, evaluations, report.
    # The keys above are never overwritten.
    "interview_session": None,
    "interview_questions": None,
    "interview_answers": None,
    "interview_evaluations": None,
    "interview_report": None,
    # Salary Negotiator — cached benchmark, script, counter-offer.
    # The keys above are never overwritten.
    "salary_benchmark": None,
    "salary_script": None,
    "salary_counter_offer": None,
    # AI Evaluation Dashboard — metrics, trends, report, insights.
    # The keys above are never overwritten.
    "evaluation_metrics": None,
    "evaluation_trends": None,
    "evaluation_report": None,
    "evaluation_insights": None,
}


def init_state() -> None:
    """Populate session state with defaults for any missing keys."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get(key: str, default: Any = None) -> Any:
    """Return a session-state value, falling back to ``default``."""
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    """Set a session-state value."""
    st.session_state[key] = value
