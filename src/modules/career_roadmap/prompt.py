"""Gemini enrichment prompt + JSON response schema for the career roadmap."""

from __future__ import annotations

ROADMAP_ENRICHMENT_SYSTEM_PROMPT: str = """\
You are an expert career coach and technical learning mentor.

You will receive a deterministic career roadmap (target role, current
readiness, ordered phases, each phase's skills and milestones). Your task
is to enrich the wording — NOT to restructure the roadmap.

CRITICAL CONSTRAINTS:
- Use ONLY the skills supplied in the roadmap. Do NOT add, invent, or
  suggest any new skill that is not already in the supplied phases.
- Do NOT remove or omit any supplied skill from any phase.
- Do NOT change phase ordering, phase numbers, or which skills belong to
  which phase. Echo back the exact phase_number and skills for each phase.
- Do NOT change the current_readiness score (it is deterministic).
- For each phase, you may improve: objective, duration, practice_project.
- For each milestone (keyed by its skill), you may improve: title,
  description, completion_criteria.
- You may also refine: estimated_total_effort and final_outcome.

If you cannot improve a field, return the original value verbatim — do not
fabricate.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "estimated_total_effort": "",
  "final_outcome": "",
  "phases": [
    {
      "phase_number": 1,
      "skills": ["<exact supplied skills, in the same order>"],
      "objective": "",
      "duration": "",
      "practice_project": "",
      "milestones": [
        {
          "skill": "<exact supplied skill>",
          "title": "",
          "description": "",
          "completion_criteria": ""
        }
      ]
    }
  ]
}
"""

# OpenAPI-subset schema constraining Gemini's JSON output. The per-phase
# "skills" list is REQUIRED so the enricher can validate that no skills
# were added/removed/reordered. Client-side validation
# (enricher._validate_enrichment) is the safety net.
ROADMAP_ENRICHMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "estimated_total_effort": {"type": "string"},
        "final_outcome": {"type": "string"},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phase_number": {"type": "integer"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "objective": {"type": "string"},
                    "duration": {"type": "string"},
                    "practice_project": {"type": "string"},
                    "milestones": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "completion_criteria": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["phase_number", "skills", "milestones"],
            },
        },
    },
    "required": ["phases"],
}
