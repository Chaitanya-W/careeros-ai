"""Gemini prompts + JSON response schemas for the Application Copilot.

All three system prompts share the SAME evidence-grounding constraints:
Gemini may improve wording only — it must NEVER invent candidate facts
(employers, titles, dates, years, projects, technologies, languages, certs,
degrees, achievements, metrics, responsibilities, awards, publications,
links, credentials).
"""

from __future__ import annotations

_EVIDENCE_GROUNDING_RULES: str = """\
CRITICAL CONSTRAINTS (do not violate any):
- You are an evidence-grounded career application assistant.
- Only use facts present in the supplied evidence. Improve wording rather
  than creating facts.
- Never invent candidate experience, technologies, achievements, metrics,
  dates, employers, certifications, or responsibilities.
- If a claim is unsupported by the supplied evidence, do not generate it.
- Do not add technologies that are not in the candidate's evidence.
- Do not invent quantitative metrics (%, x, k, M) or years that do not
  appear verbatim in the evidence.
- If job-specific detail is not available, write generically rather than
  inventing it.
"""


TAILOR_SYSTEM_PROMPT: str = (
    _EVIDENCE_GROUNDING_RULES
    + """\
Your task: produce resume-tailoring suggestions that make EXISTING resume
evidence more relevant to the target role. For each suggestion you may
rephrase the candidate's existing content for clarity/relevance, but you
must NOT add new facts, technologies, metrics, or employers.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "suggestions": [
    {
      "section": "summary" | "skills" | "experience" | "projects" | "education",
      "original_content": "",
      "suggested_content": "",
      "reason": ""
    }
  ]
}
"""
)

COVER_LETTER_SYSTEM_PROMPT: str = (
    _EVIDENCE_GROUNDING_RULES
    + """\
Your task: write a concise, professional cover letter for the target role
that connects the candidate's GENUINE experience to the job requirements.
Explain interest in the role, connect relevant experience, highlight true
strengths. Do not invent company research, achievements, metrics,
employers, projects, or skills.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "cover_letter": "",
  "evidence_ids": ["EXP-001", "PROJ-001", ...]
}
"""
)

BULLET_SYSTEM_PROMPT: str = (
    _EVIDENCE_GROUNDING_RULES
    + """\
Your task: optimize a single resume bullet for the target role. You may
improve clarity, action verbs, technical specificity, relevance, and
conciseness. You must PRESERVE factual meaning: do not increase numbers,
do not invent outcomes, do not add technologies, do not invent leadership.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "optimized_bullet": "",
  "reason": "",
  "evidence_ids": ["EXP-001", ...]
}
"""
)

# OpenAPI-subset schemas constraining Gemini's JSON output. Client-side
# validation (validator.validate_claims) is the safety net that rejects any
# invented candidate facts.

TAILOR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "original_content": {"type": "string"},
                    "suggested_content": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["section", "suggested_content"],
            },
        },
    },
    "required": ["suggestions"],
}

COVER_LETTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cover_letter": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cover_letter"],
}

BULLET_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "optimized_bullet": {"type": "string"},
        "reason": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["optimized_bullet"],
}
