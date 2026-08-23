"""Deterministic insights + optional Gemini wording enrichment.

Insights are generated deterministically from metrics + trends. Gemini may
optionally improve the wording — but it MUST NOT change scores,
percentages, trend directions, or factual values. The enricher validates
that Gemini's output contains no changed numbers.

Reuses the existing :class:`GeminiClient` (Step 3) — no second client.
No raw ``resume_text`` is sent to Gemini.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.modules.evaluation_dashboard.model import (
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)
from src.modules.evaluation_dashboard.prompt import (
    INSIGHT_SCHEMA,
    INSIGHT_SYSTEM_PROMPT,
)
from src.services.gemini import GeminiClient, GeminiError

_logger = logging.getLogger(__name__)

__all__ = ["generate_insights", "enrich_insights"]


def generate_insights(
    summary: EvaluationSummary,
    trends: list[TrendPoint],
) -> list[str]:
    """Generate deterministic insights from metrics + trends.

    These are template-based, grounded in the actual computed values.
    """
    insights: list[str] = []

    # --- Readiness trend ---
    readiness_trend = _find_trend(trends, "Interview Readiness")
    if readiness_trend and readiness_trend.direction is not Direction.INSUFFICIENT_DATA:
        if readiness_trend.direction is Direction.IMPROVING:
            insights.append(
                f"Interview readiness improved from {readiness_trend.previous:.0f}% "
                f"to {readiness_trend.current:.0f}%."
            )
        elif readiness_trend.direction is Direction.DECLINING:
            insights.append(
                f"Interview readiness declined from {readiness_trend.previous:.0f}% "
                f"to {readiness_trend.current:.0f}%. Focus on your weakest areas."
            )
        else:
            insights.append(
                f"Interview readiness is stable at {readiness_trend.current:.0f}%."
            )

    # --- Lowest-scoring dimension ---
    aq = summary.answer_quality
    if aq.total_evaluated > 0:
        scores = {
            "Technical": aq.technical_avg,
            "Relevance": aq.relevance_avg,
            "Specificity": aq.specificity_avg,
            "Communication": aq.communication_avg,
            "Completeness": aq.completeness_avg,
        }
        lowest = min(scores, key=scores.get)
        highest = max(scores, key=scores.get)
        insights.append(
            f"{lowest} is currently your lowest-scoring answer dimension "
            f"({scores[lowest]:.1f}/10)."
        )
        insights.append(
            f"{highest} is your strongest dimension ({scores[highest]:.1f}/10)."
        )

    # --- Unsupported claims ---
    if aq.unsupported_claim_count > 0:
        insights.append(
            f"{aq.unsupported_claim_count} unsupported claim(s) detected across "
            f"interview answers. Ground your responses in real experience."
        )

    # --- Trend insights ---
    for tp in trends:
        if tp.direction is Direction.IMPROVING and tp.label != "Interview Readiness":
            insights.append(f"{tp.label} improved across sessions.")
        elif tp.direction is Direction.DECLINING and tp.label != "Interview Readiness":
            insights.append(f"{tp.label} declined — focus practice here.")

    # --- Data availability ---
    if summary.job_match_score is not None:
        insights.append(
            f"Current job match score: {summary.job_match_score:.0f}%."
        )
    if summary.skill_readiness is not None:
        insights.append(
            f"Skill readiness: {summary.skill_readiness:.0f}%."
        )
    if summary.negotiation_prepared is False:
        insights.append(
            "Negotiation preparation is incomplete — generate a salary "
            "benchmark and negotiation script."
        )

    if not insights:
        insights.append(
            "Complete an interview session to unlock detailed evaluation insights."
        )

    return insights


def enrich_insights(
    insights: list[str],
    summary: EvaluationSummary,
    trends: list[TrendPoint],
    *,
    client: Optional[GeminiClient] = None,
) -> list[str]:
    """Optionally enrich insight wording via Gemini.

    Gemini may ONLY improve wording. If Gemini changes, removes, adds,
    or moves any number within or across insights, the enrichment for
    that insight is rejected and the deterministic version is kept.
    Validation is **per-insight** (numeric multiset match), not global.
    """
    if client is None or not insights:
        return insights

    user_text = _build_context(insights, summary, trends)
    try:
        raw = client.generate_json(
            system_prompt=INSIGHT_SYSTEM_PROMPT,
            user_text=user_text,
            response_schema=INSIGHT_SCHEMA,
        )
    except GeminiError:
        raise
    except Exception:
        _logger.exception("Unexpected error during insight enrichment")
        return insights

    return _validate_insights(raw, insights)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# Alias for readability.
Direction = TrendDirection


def _find_trend(trends: list[TrendPoint], label: str) -> Optional[TrendPoint]:
    for tp in trends:
        if tp.label == label:
            return tp
    return None


def _build_context(insights: list[str], summary: EvaluationSummary, trends: list[TrendPoint]) -> str:
    """Build a redacted context (no raw resume_text)."""
    lines = ["Deterministic insights (improve wording ONLY — do NOT change", "numbers, trend directions, or factual values):", ""]
    for i, insight in enumerate(insights, start=1):
        lines.append(f"[{i}] {insight}")
    lines.append("")
    lines.append(f"Readiness: {summary.readiness_score}")
    lines.append(f"Answer quality overall: {summary.answer_quality.overall_avg}")
    lines.append(f"Job match: {summary.job_match_score}")
    lines.append(f"Skill readiness: {summary.skill_readiness}")
    for tp in trends:
        lines.append(f"Trend {tp.label}: {tp.direction.value} (delta={tp.delta})")
    return "\n".join(lines)


def _validate_insights(raw: str, deterministic: list[str]) -> list[str]:
    """Parse + validate Gemini's insight enrichment.

    Returns the deterministic insights if validation fails. Only accepts
    Gemini wording that **per-insight** preserves the exact numeric
    multiset of the corresponding deterministic insight.

    Gemini must NOT:
    - change a number within an insight
    - remove a deterministic number
    - introduce a new number
    - move a number from another insight
    - change a percentage, score, session count, or trend value

    Numbers are compared as a **per-insight multiset** (sorted list), not
    a global set — so swapping numbers between insights is rejected.
    """
    if not raw or not raw.strip():
        return deterministic
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return deterministic
    if not isinstance(data, dict):
        return deterministic
    gemini_insights = data.get("insights")
    if not isinstance(gemini_insights, list) or len(gemini_insights) != len(deterministic):
        return deterministic

    out: list[str] = []
    for i, gem_insight in enumerate(gemini_insights):
        if not isinstance(gem_insight, str):
            out.append(deterministic[i])
            continue
        gem_insight = gem_insight.strip()
        # PER-INSIGHT numeric immutability: the numeric multiset of the
        # Gemini version must EXACTLY match the deterministic version for
        # THIS insight (not a global set). This prevents number swapping
        # between insights.
        # The regex r"\d+(?:\.\d+)?" matches decimals like "6.5" but
        # NOT trailing sentence periods (e.g. "72." → "72").
        _num_re = r"\d+(?:\.\d+)?"
        det_nums = sorted(re.findall(_num_re, deterministic[i]))
        gem_nums = sorted(re.findall(_num_re, gem_insight))
        if det_nums != gem_nums:
            # Numbers changed, removed, added, or moved → reject.
            out.append(deterministic[i])
            continue
        out.append(gem_insight)
    return out
