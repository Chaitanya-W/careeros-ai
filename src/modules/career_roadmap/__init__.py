"""Career Roadmap module.

Public API: :func:`render` draws the prerequisite-guarded, deterministic
roadmap + optional AI enrichment workflow. Internally delegates to
:mod:`src.modules.career_roadmap.ui`.

Submodules
----------
- ``model``     : :class:`CareerRoadmap`, :class:`RoadmapPhase`, :class:`RoadmapMilestone`.
- ``builder``   : :func:`build_career_roadmap` deterministic builder (no AI).
- ``prompt``     : Gemini enrichment system prompt + JSON response schema.
- ``enricher``  : :func:`enrich_roadmap` + :func:`apply_enrichment` (reuses GeminiClient).
- ``ui``        : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.career_roadmap.ui import render

MODULE_NAME: str = "Career Roadmap"
MODULE_KEY: str = "career_roadmap"
MODULE_DESCRIPTION: str = (
    "Generate tailored, milestone-based career growth plans with "
    "learning resources."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
