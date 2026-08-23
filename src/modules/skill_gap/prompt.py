"""Gemini enrichment prompt + JSON response schema for skill-gap guidance."""

from __future__ import annotations

SKILL_GAP_ENRICHMENT_SYSTEM_PROMPT: str = """\
You are an expert career coach and technical learning mentor.

You will receive a list of MISSING skills (the gaps) for a target role,
along with the job's requirements and a brief, redacted summary of the
candidate's existing resume evidence. Your task is to enrich EACH supplied
skill with concrete, actionable learning guidance.

CRITICAL CONSTRAINTS:
- Enrich ONLY the skills in the supplied list.
- Do NOT add, invent, or recommend any new skill that is not in the list.
- Do NOT change the order or rename the supplied skills.
- If you cannot produce guidance for a supplied skill, return empty
  strings / empty arrays for that skill — do not fabricate.

For each supplied skill, return:
- why_it_matters: one or two sentences on why this skill matters for the
  target role (grounded in the job requirements).
- learning_objectives: 2-4 concise, measurable objectives.
- learning_path: 3-6 ordered steps (each a short phrase) to build the
  skill from the candidate's current level toward the target.
- practice_project: one concrete practice project idea.
- estimated_effort: a short human-readable effort estimate (e.g.
  "2-4 weeks of part-time study").

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "skills": [
    {
      "skill": "<exact supplied skill name>",
      "why_it_matters": "",
      "learning_objectives": ["", ...],
      "learning_path": ["", ...],
      "practice_project": "",
      "estimated_effort": ""
    }
  ]
}
"""

# OpenAPI-subset schema constraining Gemini's JSON output. The list is
# under "skills"; each item carries the 5 enrichment fields. Client-side
# validation (enricher._validate_enrichment) is the safety net that
# rejects any skill not in the supplied list.
SKILL_GAP_ENRICHMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "learning_objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "learning_path": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "practice_project": {"type": "string"},
                    "estimated_effort": {"type": "string"},
                },
            },
        },
    },
    "required": ["skills"],
}
