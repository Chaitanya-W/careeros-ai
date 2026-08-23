"""Skill Gap module.

Public API: :func:`render` draws the prerequisite-guarded, deterministic
report + optional AI enrichment workflow. Internally delegates to
:mod:`src.modules.skill_gap.ui`.

Submodules
----------
- ``model``     : :class:`Priority` enum, :class:`SkillGap`, :class:`SkillGapReport`.
- ``report``    : :func:`build_skill_gap_report` deterministic builder (no AI).
- ``prompt``     : Gemini enrichment system prompt + JSON response schema.
- ``enricher``  : :func:`enrich_skill_gaps` + :func:`apply_enrichment` (reuses GeminiClient).
- ``ui``        : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.skill_gap.ui import render

MODULE_NAME: str = "Skill Gap"
MODULE_KEY: str = "skill_gap"
MODULE_DESCRIPTION: str = (
    "Compare your skills against target roles to surface actionable gaps."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
