"""Gemini enrichment prompt + JSON schema for negotiation-script wording.

The system prompt enforces: Gemini may ONLY improve wording/tone/clarity.
It must NOT change numbers, benchmarks, counter-offer, evidence IDs, or
candidate facts.
"""

from __future__ import annotations

NEGOTIATION_ENRICHMENT_SYSTEM_PROMPT: str = """\
You are an evidence-grounded salary negotiation coach.

You receive a deterministic negotiation script with talking points. You may
ONLY improve the wording, tone, clarity, and naturalness of the talking
points. You must NOT:

- Change any salary numbers, benchmark values, or counter-offer numbers.
- Invent candidate facts (employers, projects, technologies, certifications,
  metrics, dates, years, or responsibilities).
- Add new talking points or remove existing ones.
- Change evidence IDs or point IDs.
- Change the structural metadata (titles, point IDs).

If a talking point is already well-worded, return it verbatim.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "talking_points": [
    {
      "point_id": "NP-001",
      "wording": ""
    }
  ]
}
"""

NEGOTIATION_ENRICHMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "talking_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point_id": {"type": "string"},
                    "wording": {"type": "string"},
                },
                "required": ["point_id", "wording"],
            },
        },
    },
    "required": ["talking_points"],
}
