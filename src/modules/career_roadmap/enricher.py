"""Career Roadmap enrichment service — bridge between the roadmap and Gemini.

Reuses the existing :class:`GeminiClient` (Step 3) — there is no second
Gemini implementation.

The enricher receives the deterministic :class:`CareerRoadmap` and returns
wording improvements (objective, duration, practice_project, milestone
wording, total effort, final outcome). Gemini is explicitly instructed NOT
to add/remove/reorder skills, and the parsed response is **validated**:

- Each returned phase's ``phase_number`` must exist in the roadmap.
- Each returned phase's ``skills`` (as an ordered list) must EXACTLY match
  the roadmap phase's skills — any added/removed/reordered skill causes the
  whole enrichment to be rejected.
- The union of returned skills must equal the union of roadmap skills.

If validation fails, :class:`RoadmapEnrichmentError` is raised and the
deterministic roadmap is left untouched (no session-state corruption).

No full resume text or full job description is sent — only a redacted
roadmap context. API keys are never logged.
"""

from __future__ import annotations

import json
from typing import Optional

from src.modules.career_roadmap.model import CareerRoadmap
from src.modules.career_roadmap.prompt import (
    ROADMAP_ENRICHMENT_SCHEMA,
    ROADMAP_ENRICHMENT_SYSTEM_PROMPT,
)
from src.services.gemini import GeminiClient, GeminiError

__all__ = [
    "RoadmapEnrichmentError",
    "enrich_roadmap",
    "apply_enrichment",
]


class RoadmapEnrichmentError(Exception):
    """Raised when the enrichment response is invalid / cannot be applied."""


def enrich_roadmap(
    roadmap: CareerRoadmap,
    *,
    job=None,
    resume=None,
    client: Optional[GeminiClient] = None,
) -> dict:
    """Call Gemini to enrich the roadmap's wording.

    Returns a structured enrichment dict that the caller merges via
    :func:`apply_enrichment`. Never adds/removes/reorders skills. Raises
    :class:`RoadmapEnrichmentError` if the response does not validate
    against the roadmap's skill structure.
    """
    if not roadmap.phases:
        return {}

    user_text = _build_context(roadmap)
    gemini = client or GeminiClient()
    raw = gemini.generate_json(
        system_prompt=ROADMAP_ENRICHMENT_SYSTEM_PROMPT,
        user_text=user_text,
        response_schema=ROADMAP_ENRICHMENT_SCHEMA,
    )
    return _validate_enrichment(raw, roadmap)


def apply_enrichment(roadmap: CareerRoadmap, enrichment: dict) -> None:
    """Merge ``enrichment`` into the roadmap in place.

    Only wording fields are updated. Skills, phase ordering, readiness,
    and phase_number are NEVER changed. Sets ``roadmap.enriched = True``.
    """
    if not enrichment:
        return
    if enrichment.get("estimated_total_effort"):
        roadmap.estimated_total_effort = enrichment["estimated_total_effort"]
    if enrichment.get("final_outcome"):
        roadmap.final_outcome = enrichment["final_outcome"]

    phases_by_num = {p.phase_number: p for p in roadmap.phases}
    for phase_enrich in enrichment.get("phases", []):
        num = phase_enrich.get("phase_number")
        phase = phases_by_num.get(num)
        if phase is None:
            continue
        if phase_enrich.get("objective"):
            phase.objective = phase_enrich["objective"]
        if phase_enrich.get("duration"):
            phase.duration = phase_enrich["duration"]
        if phase_enrich.get("practice_project"):
            phase.practice_project = phase_enrich["practice_project"]
        ms_by_skill = {}
        for ms in phase.milestones:
            n = _normalize(ms.skill)
            if n:
                ms_by_skill.setdefault(n, ms)
        for ms_enrich in phase_enrich.get("milestones", []):
            n = _normalize(ms_enrich.get("skill", ""))
            ms = ms_by_skill.get(n)
            if ms is None:
                continue
            if ms_enrich.get("title"):
                ms.title = ms_enrich["title"]
            if ms_enrich.get("description"):
                ms.description = ms_enrich["description"]
            if ms_enrich.get("completion_criteria"):
                ms.completion_criteria = ms_enrich["completion_criteria"]
    roadmap.enriched = True


# --------------------------------------------------------------------------- #
# Context + validation
# --------------------------------------------------------------------------- #


def _build_context(roadmap: CareerRoadmap) -> str:
    """Build a redacted prompt payload (the roadmap structure only)."""
    lines = [
        f"Target role: {roadmap.target_role}",
        f"Current readiness: {roadmap.current_readiness}/100 "
        "(deterministic — do not change)",
        "",
        "Phases (enrich wording only — do NOT add/remove/reorder skills):",
    ]
    for p in roadmap.phases:
        lines.append(f"- Phase {p.phase_number}: {p.title}")
        lines.append(f"  skills: {', '.join(p.skills)}")
        lines.append(f"  objective: {p.objective}")
        lines.append(f"  duration: {p.duration}")
        lines.append(f"  practice_project: {p.practice_project}")
        for ms in p.milestones:
            lines.append(
                f"  milestone [{ms.skill}]: {ms.title} | "
                f"{ms.description} | criteria: {ms.completion_criteria}"
            )
    return "\n".join(lines)


def _validate_enrichment(raw: str, roadmap: CareerRoadmap) -> dict:
    """Parse + validate the Gemini response.

    Ensures:
    - response is a JSON object with a "phases" list.
    - each returned phase's phase_number exists in the roadmap.
    - each returned phase's "skills" list EXACTLY matches the roadmap
      phase's skills (same items, same order) -> no add/remove/reorder.
    - the union of returned skills equals the union of roadmap skills.

    Raises :class:`RoadmapEnrichmentError` on any mismatch.
    """
    if not raw or not raw.strip():
        raise RoadmapEnrichmentError("Gemini returned an empty response.")
    text = _strip_code_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RoadmapEnrichmentError(
            f"Gemini response was not valid JSON "
            f"({e.msg} at line {e.lineno} col {e.colno})."
        )
    if not isinstance(data, dict):
        raise RoadmapEnrichmentError("Response was not a JSON object.")

    phases_raw = data.get("phases")
    if not isinstance(phases_raw, list):
        raise RoadmapEnrichmentError("Response did not contain a 'phases' list.")
    if len(phases_raw) != len(roadmap.phases):
        raise RoadmapEnrichmentError(
            f"Phase count mismatch: response has {len(phases_raw)} "
            f"phase(s), roadmap has {len(roadmap.phases)}."
        )

    roadmap_phases_by_num = {p.phase_number: p for p in roadmap.phases}
    roadmap_skills_union = {_normalize(s) for s in roadmap.all_skills}
    returned_skills_union: set[str] = set()

    for phase_enrich in phases_raw:
        if not isinstance(phase_enrich, dict):
            raise RoadmapEnrichmentError("A phase entry was not an object.")
        num = phase_enrich.get("phase_number")
        roadmap_phase = roadmap_phases_by_num.get(num)
        if roadmap_phase is None:
            raise RoadmapEnrichmentError(
                f"Returned phase_number {num!r} not in roadmap."
            )
        returned_skills = phase_enrich.get("skills")
        if not isinstance(returned_skills, list):
            raise RoadmapEnrichmentError(
                f"Phase {num} 'skills' was not a list."
            )
        # Exact ordered match: no add / remove / reorder.
        if [_normalize(s) for s in returned_skills] != [
            _normalize(s) for s in roadmap_phase.skills
        ]:
            raise RoadmapEnrichmentError(
                f"Phase {num} skills differ from the roadmap "
                f"(added/removed/reordered)."
            )
        for s in returned_skills:
            returned_skills_union.add(_normalize(s))

    if returned_skills_union != roadmap_skills_union:
        raise RoadmapEnrichmentError(
            "Returned skill set does not match the roadmap skill set "
            "(skills added or removed)."
        )

    return data


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #


def _normalize(skill: str) -> str:
    import re

    if not skill:
        return ""
    s = str(skill).lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b\d+(\.\d+)*\+?\b", "", s)
    s = re.sub(r"[^a-z0-9+#./\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t
