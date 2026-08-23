"""Job Match module.

Public API: :func:`render` draws the job-description input -> job analysis
-> deterministic match -> results workflow. Internally delegates to
:mod:`src.modules.jd_analysis.ui`.

Submodules
----------
- ``model``     : :class:`JobAnalysis` + :class:`MatchAnalysis` schemas.
- ``prompt``     : Gemini system prompt + JSON response schema.
- ``analyzer``  : :func:`analyze_job_description` service (reuses GeminiClient).
- ``matcher``   : :func:`match_jobs` deterministic matcher (no LLM).
- ``ui``        : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.jd_analysis.ui import render

MODULE_NAME: str = "Job Match"
MODULE_KEY: str = "jd_analysis"
MODULE_DESCRIPTION: str = (
    "Analyze job descriptions and measure match against your profile."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
