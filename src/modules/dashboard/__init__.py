"""Dashboard module — the CareerOS AI home view.

Renders the professional SaaS-style landing dashboard:
branding hero, welcome, score cards (placeholders), a career-progress
indicator, a Getting-Started checklist, a recent-activity empty state,
and clear calls to action.

No fake data is shown — every metric without a real source renders the
``--`` placeholder, and progress is honestly reported as 0% until the
user completes onboarding steps.
"""

from __future__ import annotations

import streamlit as st

from src.ui.components.cards import (
    empty_state_panel,
    metric_card,
    section_header,
    step_row,
)
from src.ui.components.cta import nav_cta
from src.ui.components.branding import brand_hero

MODULE_NAME: str = "Dashboard"
MODULE_KEY: str = "dashboard"
MODULE_DESCRIPTION: str = "Your career command center."

# Onboarding steps shown on the dashboard. Each maps to a module the user
# can jump into directly from the "Start" button.
_ONBOARDING_STEPS: list[tuple[str, str]] = [
    ("Upload your resume", "resume_intelligence"),
    ("Match a job description", "jd_analysis"),
    ("Assess your skill gaps", "skill_gap"),
    ("Plan your career roadmap", "career_roadmap"),
    ("Practice a mock interview", "voice_interview"),
]

_TOTAL_STEPS: int = len(_ONBOARDING_STEPS)


def render() -> None:
    """Render the CareerOS AI dashboard."""
    brand_hero()
    _render_welcome()
    _render_score_cards()
    _render_career_progress()
    _render_onboarding_and_activity()
    _render_quick_actions()


def _render_welcome() -> None:
    st.write(
        "Welcome to your career command center. Use the sidebar to move "
        "between modules — your scores, readiness, and next steps update "
        "here as you add data across CareerOS."
    )


def _render_score_cards() -> None:
    section_header("Career Snapshot", "Scores appear once you add the related data.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        metric_card("Resume Score", "--", hint="Upload a resume to score it.")
    with col_b:
        metric_card("Job Match", "--", hint="Add a target role to measure match.")
    with col_c:
        metric_card("Career Readiness", "--", hint="Complete onboarding to score.")


def _render_career_progress() -> None:
    section_header("Career Progress", "Based on completed onboarding steps.")
    # No onboarding step has been completed yet — report 0% honestly.
    st.progress(0.0, text=f"0 of {_TOTAL_STEPS} onboarding steps complete")
    st.caption("Finish onboarding to start building your progress.")


def _render_onboarding_and_activity() -> None:
    left, right = st.columns(2)
    with left:
        section_header("Getting Started", "Recommended first steps.")
        for index, (label, target) in enumerate(_ONBOARDING_STEPS, start=1):
            step_row(index, label, target)
    with right:
        section_header("Recent Activity", "Your latest actions across CareerOS.")
        empty_state_panel(
            title="No activity yet",
            hint="As you use CareerOS, your recent actions will appear here.",
        )


def _render_quick_actions() -> None:
    section_header("Quick Actions", "Jump straight into a module.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        nav_cta("Upload Resume", "resume_intelligence", key="cta_resume")
    with col_b:
        nav_cta("Match a Job", "jd_analysis", key="cta_jd")
    with col_c:
        nav_cta("Practice Interview", "voice_interview", key="cta_interview")
