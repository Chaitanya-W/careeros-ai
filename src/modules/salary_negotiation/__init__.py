"""Salary Negotiator module.

Public API: :func:`render` draws the benchmark → leverage → script →
counter-offer workflow. Internally delegates to
:mod:`src.modules.salary_negotiation.ui`.

Submodules
----------
- ``model``      : SalaryBenchmark, NegotiationPoint, NegotiationScript, CounterOffer.
- ``benchmark``  : deterministic heuristic benchmark (no Gemini, no paid API).
- ``script``     : deterministic script + counter-offer builder (grounded in evidence).
- ``prompt``     : Gemini enrichment system prompt + JSON schema.
- ``enricher``   : optional Gemini wording enrichment (reuses GeminiClient, validated).
- ``ui``         : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.salary_negotiation.ui import render

MODULE_NAME: str = "Salary Negotiator"
MODULE_KEY: str = "salary_negotiation"
MODULE_DESCRIPTION: str = (
    "Benchmark compensation and rehearse data-backed negotiation scripts."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
