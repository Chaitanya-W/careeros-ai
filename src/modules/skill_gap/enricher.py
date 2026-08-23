"""Skill Gap enrichment service — bridge between the report and Gemini.

Reuses the existing :class:`GeminiClient` (Step 3) — there is no second
Gemini implementation.

The enricher receives the deterministic :class:`SkillGap` list + a
redacted job/resume context and returns per-skill learning guidance. Gemini
is explicitly instructed NOT to add new skills, and the parsed response is
**validated**: any skill not in the supplied list is rejected (never
corrupting the gap list). If Gemini omits a supplied skill, that skill's
enrichment stays empty (deterministic defaults remain).

No full resume text or full job description is sent — only structured,
redacted context (skill names, target role, requirement phrases, evidence
summaries). API keys are never logged.
"""

from __future__ import annotations

import json
from typing import Optional

from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import SkillGap, SkillGapParseError
from src.modules.skill_gap.prompt import (
    SKILL_GAP_ENRICHMENT_SCHEMA,
    SKILL_GAP_ENRICHMENT_SYSTEM_PROMPT,
)
from src.services.gemini import GeminiClient, GeminiError

__all__ = [
    "SkillGapEnrichmentError",
    "enrich_skill_gaps",
    "apply_enrichment",
]


class SkillGapEnrichmentError(Exception):
    """Raised when the enrichment response is invalid / cannot be applied."""


def enrich_skill_gaps(
    skill_gaps: list[SkillGap],
    *,
    job: JobAnalysis,
    resume: ResumeAnalysis,
    client: Optional[GeminiClient] = None,
) -> dict[str, dict]:
    """Call Gemini to enrich the supplied skill gaps with learning guidance.

    Returns a mapping ``{canonical_skill_name: {enrichment fields}}``. Any
    skill the model invents (not in ``skill_gaps``) is **rejected**. Skills
    the model omits simply have no entry (their deterministic defaults
    remain).

    Raises:
        GeminiError: missing key / SDK problem / API failure.
        SkillGapEnrichmentError: response could not be parsed / validated.
    """
    if not skill_gaps:
        return {}

    supplied = [g.skill for g in skill_gaps]
    user_text = _build_context(supplied, job, resume)

    gemini = client or GeminiClient()
    raw = gemini.generate_json(
        system_prompt=SKILL_GAP_ENRICHMENT_SYSTEM_PROMPT,
        user_text=user_text,
        response_schema=SKILL_GAP_ENRICHMENT_SCHEMA,
    )
    return _validate_enrichment(raw, supplied)


def apply_enrichment(
    report,  # SkillGapReport
    enrichment: dict[str, dict],
) -> None:
    """Merge ``enrichment`` into the report's skill gaps in place.

    Only skills already present in the report can be enriched — this never
    adds skills. Sets ``report.enriched = True`` if any field was filled.
    """
    from src.modules.skill_gap.report import _normalize  # local helper

    # Map normalized skill -> report SkillGap (case-insensitive match).
    by_norm: dict[str, SkillGap] = {}
    for g in report.skill_gaps:
        n = _normalize(g.skill)
        if n:
            by_norm.setdefault(n, g)

    touched = False
    for canonical, fields in enrichment.items():
        n = _normalize(canonical)
        gap = by_norm.get(n)
        if gap is None:
            continue  # not a supplied skill — ignore
        gap.why_it_matters = fields.get("why_it_matters", "") or ""
        gap.learning_objectives = fields.get("learning_objectives", []) or []
        gap.learning_path = fields.get("learning_path", []) or []
        gap.practice_project = fields.get("practice_project", "") or ""
        gap.estimated_effort = fields.get("estimated_effort", "") or ""
        touched = True
    if touched:
        report.enriched = True


# --------------------------------------------------------------------------- #
# Context + validation
# --------------------------------------------------------------------------- #


def _build_context(
    skills: list[str],
    job: JobAnalysis,
    resume: ResumeAnalysis,
) -> str:
    """Build a redacted, structured prompt payload (no full JD/resume text)."""
    role = " · ".join(p for p in (job.job_title, job.domain) if p) or "the target role"
    required = ", ".join(job.required_skills) if job.required_skills else "(none listed)"
    preferred = ", ".join(job.preferred_skills) if job.preferred_skills else "(none listed)"
    experience_req = job.experience_requirements or "(not specified)"
    evidence_summary = _summarize_evidence(skills, resume)

    return (
        f"Target role: {role}\n"
        f"Required skills: {required}\n"
        f"Preferred skills: {preferred}\n"
        f"Experience requirement: {experience_req}\n\n"
        f"Missing skills to enrich (enrich ONLY these — do NOT add new skills):\n"
        + "\n".join(f"- {s}" for s in skills)
        + "\n\n"
        f"Candidate's existing resume evidence per missing skill:\n"
        + evidence_summary
    )


def _summarize_evidence(
    skills: list[str], resume: ResumeAnalysis
) -> str:
    """One line per skill: whether explicit / related / no evidence.

    Redacted — never includes full resume text or section contents.
    """
    from src.modules.skill_gap.report import _detect_evidence

    lines = []
    for s in skills:
        evidence, _level = _detect_evidence(s, resume)
        lines.append(f"- {s}: {evidence}")
    return "\n".join(lines)


def _validate_enrichment(
    raw: str,
    supplied: list[str],
) -> dict[str, dict]:
    """Parse + validate the Gemini response; reject AI-added skills.

    Returns ``{canonical_skill: {fields}}`` for supplied skills only.
    """
    if not raw or not raw.strip():
        raise SkillGapEnrichmentError("Gemini returned an empty response.")
    text = _strip_code_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SkillGapEnrichmentError(
            f"Gemini response was not valid JSON "
            f"({e.msg} at line {e.lineno} col {e.colno})."
        )
    if not isinstance(data, dict):
        raise SkillGapEnrichmentError(
            "Gemini response was not a JSON object."
        )

    skills_field = data.get("skills")
    if isinstance(skills_field, dict):
        items = list(skills_field.items())
    elif isinstance(skills_field, list):
        items = []
        for item in skills_field:
            if isinstance(item, dict) and item.get("skill"):
                items.append((str(item.get("skill")), item))
    else:
        raise SkillGapEnrichmentError(
            "Gemini response did not contain a 'skills' list."
        )

    supplied_norm = {_normalize(s): s for s in supplied}
    out: dict[str, dict] = {}
    for key, val in items:
        if not isinstance(val, dict):
            continue
        norm = _normalize(str(key))
        if norm not in supplied_norm:
            # AI added a skill not in the supplied list -> REJECT.
            continue
        canonical = supplied_norm[norm]
        out[canonical] = {
            "why_it_matters": _clean_str(val.get("why_it_matters")),
            "learning_objectives": _coerce_str_list(
                val.get("learning_objectives")
            ),
            "learning_path": _coerce_str_list(val.get("learning_path")),
            "practice_project": _clean_str(val.get("practice_project")),
            "estimated_effort": _clean_str(val.get("estimated_effort")),
        }
    return out


# --------------------------------------------------------------------------- #
# Local helpers (kept independent to avoid cross-module private imports).
# --------------------------------------------------------------------------- #


def _normalize(skill: str) -> str:
    import re

    if not skill:
        return ""
    s = skill.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b\d+(\.\d+)*\+?\b", "", s)
    s = re.sub(r"[^a-z0-9+#./\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_str_list(value) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    items = value if isinstance(value, list) else [value]
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t
