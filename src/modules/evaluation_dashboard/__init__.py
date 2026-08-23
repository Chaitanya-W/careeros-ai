"""AI Evaluation Dashboard module.

Public API: :func:`render` draws the metrics → trends → answer quality →
insights → exports workflow. Internally delegates to
:mod:`src.modules.evaluation_dashboard.ui`.

Submodules
----------
- ``model``      : EvaluationMetric, AnswerQuality, EvaluationSummary, TrendPoint, EvaluationReport.
- ``metrics``    : deterministic metric extraction from session state (no Gemini).
- ``trends``     : deterministic trend calculation from evaluation history.
- ``insights``   : deterministic insights + optional Gemini wording enrichment.
- ``prompt``     : Gemini insight-enrichment system prompt + JSON schema.
- ``exporter``   : JSON + CSV export of deterministic values.
- ``ui``         : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.evaluation_dashboard.ui import render

MODULE_NAME: str = "Evaluation"
MODULE_KEY: str = "evaluation_dashboard"
MODULE_DESCRIPTION: str = (
    "Track your progress, AI answer quality, and engagement over time."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
