"""Negotiation script Gemini enrichment service.

Reuses the existing :class:`GeminiClient` (Step 3) — no second client.

Gemini may ONLY improve the wording of talking points. It must NOT change:
- salary numbers / benchmark / counter-offer
- **currency or currency symbol** (e.g. $ → ₹, or €/£/ISO prefixes)
- evidence IDs / point IDs
- candidate facts
- structural metadata (titles, point count)

The enricher validates:
1. The number of returned talking points matches the supplied script.
2. Each point_id matches a supplied point_id.
3. The wording is evidence-grounded (via ``validate_claims``).
4. No new salary numbers are introduced that differ from the deterministic ones.
5. **Currency immutability**: the deterministic script's currency symbol is
   extracted; Gemini wording that introduces a DIFFERENT currency symbol
   ($, ₹, €, £, ¥, etc.) or an ISO currency prefix (USD, INR, EUR, GBP,
   etc. followed by a number) is rejected.

If any check fails → the enrichment is rejected and the deterministic
wording is kept.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.modules.application_copilot.evidence import EvidenceItem
from src.modules.application_copilot.validator import validate_claims
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.salary_negotiation.model import (
    NegotiationScript,
    ValidationStatus,
)
from src.modules.salary_negotiation.prompt import (
    NEGOTIATION_ENRICHMENT_SCHEMA,
    NEGOTIATION_ENRICHMENT_SYSTEM_PROMPT,
)
from src.services.gemini import GeminiClient, GeminiError

_logger = logging.getLogger(__name__)

__all__ = ["enrich_script", "apply_enrichment"]


def enrich_script(
    script: NegotiationScript,
    *,
    resume: ResumeAnalysis,
    job: Optional[JobAnalysis] = None,
    match=None,
    evidence: list[EvidenceItem],
    client: Optional[GeminiClient] = None,
) -> dict:
    """Call Gemini to enrich talking-point wording.

    Returns a mapping ``{point_id: improved_wording}``. If Gemini's output
    fails validation (invented facts, changed structure, wrong point count),
    an empty dict is returned (caller keeps deterministic wording).
    """
    if client is None or not script.talking_points:
        return {}

    user_text = _build_context(script)
    try:
        raw = client.generate_json(
            system_prompt=NEGOTIATION_ENRICHMENT_SYSTEM_PROMPT,
            user_text=user_text,
            response_schema=NEGOTIATION_ENRICHMENT_SCHEMA,
        )
    except GeminiError:
        raise
    except Exception:
        _logger.exception("Unexpected error during negotiation enrichment")
        return {}

    return _validate_enrichment(raw, script, evidence, resume, job, match)


def apply_enrichment(
    script: NegotiationScript,
    enrichment: dict,
) -> None:
    """Merge ``enrichment`` into the script in place.

    Only wording is updated. Point IDs, evidence IDs, titles, and all
    structural metadata are NEVER changed. Sets ``script.enriched = True``
    if any wording was updated.
    """
    if not enrichment:
        return
    touched = False
    for tp in script.talking_points:
        improved = enrichment.get(tp.point_id)
        if improved and isinstance(improved, str) and improved.strip():
            tp.wording = improved.strip()
            touched = True
    if touched:
        script.enriched = True


# --------------------------------------------------------------------------- #
# Context + validation
# --------------------------------------------------------------------------- #


def _build_context(script: NegotiationScript) -> str:
    """Build a redacted context (no raw resume_text)."""
    lines = [
        "Negotiation script talking points (improve wording ONLY — do NOT",
        "change numbers, point IDs, evidence IDs, or invent candidate facts):",
        "",
    ]
    for tp in script.talking_points:
        lines.append(f"[{tp.point_id}] {tp.title}: {tp.wording}")
    return "\n".join(lines)


def _validate_enrichment(
    raw: str,
    script: NegotiationScript,
    evidence: list[EvidenceItem],
    resume: ResumeAnalysis,
    job: Optional[JobAnalysis],
    match,
) -> dict:
    """Parse + validate the Gemini response.

    Returns ``{point_id: improved_wording}`` for valid points only.
    Returns ``{}`` if the response is malformed or validation fails.
    """
    if not raw or not raw.strip():
        return {}
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    tps_raw = data.get("talking_points")
    if not isinstance(tps_raw, list):
        return {}
    if len(tps_raw) != len(script.talking_points):
        return {}  # point count mismatch → reject

    supplied_ids = {tp.point_id for tp in script.talking_points}

    # --- Extract the deterministic script's currency symbol(s) ---
    # The script uses one currency symbol ($ or ₹) throughout. We collect
    # all symbols actually present so we know what's "allowed".
    _all_currency_re = r"[\$₹€£¥][\d,]+"
    _iso_currency_re = r"\b(?:USD|INR|EUR|GBP|JPY|CNY|AUD|CAD|CHF)\s*[\d,]+"
    det_numbers = set()
    det_currency_symbols: set[str] = set()
    for tp in script.talking_points:
        det_numbers |= set(re.findall(_all_currency_re, tp.wording))
        det_numbers |= set(re.findall(_iso_currency_re, tp.wording))
        # Collect which currency symbols the deterministic script uses.
        for sym_match in re.finditer(r"[\$₹€£¥]", tp.wording):
            det_currency_symbols.add(sym_match.group(0))

    out: dict = {}
    for item in tps_raw:
        if not isinstance(item, dict):
            return {}
        pid = (item.get("point_id") or "").strip()
        wording = (item.get("wording") or "").strip()
        if pid not in supplied_ids:
            return {}  # unknown point_id → reject

        # --- Currency immutability check ---
        # Collect all currency symbols in the Gemini wording.
        gem_currency_symbols: set[str] = set()
        for sym_match in re.finditer(r"[\$₹€£¥]", wording):
            gem_currency_symbols.add(sym_match.group(0))
        # Any symbol NOT in the deterministic set → reject this point.
        foreign_symbols = gem_currency_symbols - det_currency_symbols
        if foreign_symbols:
            continue

        # Check for ISO currency prefixes (USD/INR/EUR/GBP + number).
        gem_iso = set(re.findall(_iso_currency_re, wording))
        det_iso = set(re.findall(_iso_currency_re, " ".join(
            tp.wording for tp in script.talking_points
        )))
        if gem_iso and not gem_iso.issubset(det_iso):
            continue

        # Evidence validation (no invented facts).
        v = validate_claims(wording, evidence, resume=resume, job=job, match=match)
        # Filter out false-positive proper-noun warnings for common English
        # pronouns ("I'm", "I've", "I'll", "I'd") — these are NOT invented
        # candidate entities.
        real_warnings = [w for w in v.warnings if not _is_pronoun_false_positive(w)]
        if v.unsupported_claims or real_warnings:
            continue

        # Verify salary numbers weren't changed (using the correct currency
        # symbol set for this script).
        gem_numbers = set(re.findall(_all_currency_re, wording))
        gem_numbers |= set(re.findall(_iso_currency_re, wording))
        if gem_numbers and not gem_numbers.issubset(det_numbers):
            continue

        out[pid] = wording
    return out


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t


# Common English pronoun contractions that are NOT proper-noun entities.
# These are false positives from the proper-noun extractor (capitalized,
# not at sentence start, not in the evidence corpus).
_PRONOUN_CONTRACTIONS = frozenset({
    "i'm", "i've", "i'll", "i'd", "i're",
})


def _is_pronoun_false_positive(warning: str) -> bool:
    """True if the warning is about a common pronoun (false positive)."""
    w_lower = warning.lower()
    for pron in _PRONOUN_CONTRACTIONS:
        if f"'{pron}'" in w_lower or f" {pron} " in w_lower:
            return True
    return False
