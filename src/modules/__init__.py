"""Feature modules package with a render() dispatcher.

Each subpackage exposes a ``render()`` function that draws the module's
Streamlit view. :func:`render_selected` maps a module key (as stored in
``session_state["selected_module"]``) to its render function.

Adding a module is a two-step change:
1. Add an entry to ``src/config/settings.py::MODULES``.
2. Import the module's ``render`` here and register it in ``_REGISTRY``.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from src.modules.application_copilot import render as _render_application_copilot
from src.modules.career_roadmap import render as _render_career_roadmap
from src.modules.dashboard import render as _render_dashboard
from src.modules.evaluation_dashboard import render as _render_evaluation_dashboard
from src.modules.jd_analysis import render as _render_jd_analysis
from src.modules.rag import render as _render_rag
from src.modules.resume_intelligence import render as _render_resume_intelligence
from src.modules.salary_negotiation import render as _render_salary_negotiation
from src.modules.skill_gap import render as _render_skill_gap
from src.modules.voice_interview import render as _render_voice_interview

RenderFn = Callable[[], None]

# Module key -> render function. Keys mirror the MODULES registry in
# ``src/config/settings.py``.
_REGISTRY: dict[str, RenderFn] = {
    "dashboard": _render_dashboard,
    "resume_intelligence": _render_resume_intelligence,
    "jd_analysis": _render_jd_analysis,
    "skill_gap": _render_skill_gap,
    "career_roadmap": _render_career_roadmap,
    "application_copilot": _render_application_copilot,
    "rag": _render_rag,
    "voice_interview": _render_voice_interview,
    "salary_negotiation": _render_salary_negotiation,
    "evaluation_dashboard": _render_evaluation_dashboard,
}


def render_selected(key: str) -> None:
    """Render the module registered under ``key``.

    Falls back to a friendly warning if the key is unknown (defensive —
    the sidebar only ever sends registered keys).
    """
    fn = _REGISTRY.get(key)
    if fn is None:
        st.warning(f"Unknown module: {key!r}. Returning to the dashboard.")
        _REGISTRY["dashboard"]()
        return
    fn()
