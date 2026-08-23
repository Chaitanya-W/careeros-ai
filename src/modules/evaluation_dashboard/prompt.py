"""Gemini insight enrichment prompt + JSON schema.

Gemini may ONLY improve the wording of insights. It must NOT change
scores, percentages, trend directions, session counts, or any factual
metric. The prompt enforces this.
"""

from __future__ import annotations

INSIGHT_SYSTEM_PROMPT: str = """\
You are an evidence-grounded career coaching assistant.

You receive deterministic evaluation insights with factual metrics. You may
ONLY improve the wording, tone, and clarity of the insights. You must NOT:

- Change any scores, percentages, or numerical values.
- Change trend directions (improving/declining/stable).
- Change session counts or metric availability.
- Invent candidate facts (employers, projects, technologies, etc.).
- Add new insights beyond the supplied ones.

If an insight is already well-worded, return it verbatim.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "insights": ["", ...]
}
"""

INSIGHT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["insights"],
}
