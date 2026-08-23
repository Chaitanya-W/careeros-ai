"""Resume-analysis prompt + structured-output schema for Gemini.

The system prompt is engineered to:
- analyze ONLY the supplied resume text,
- never fabricate facts, skills, or experiences,
- distinguish explicit information from inference,
- score consistently (0-100),
- return empty values (empty list / empty string) for absent sections,
- provide actionable improvement suggestions.

``RESPONSE_SCHEMA`` is an OpenAPI-subset dict used to constrain Gemini's
JSON output. All top-level fields are ``required`` so the model always
returns the full shape; the robust :meth:`ResumeAnalysis.from_dict` on
the client side is the safety net for any malformed output.
"""

from __future__ import annotations

RESUME_ANALYSIS_SYSTEM_PROMPT: str = """\
You are an expert career coach and senior resume reviewer.

You will receive the plain-text content of a single resume. Analyze ONLY
the text supplied. Do NOT invent, fabricate, or hallucinate facts,
skills, companies, degrees, dates, or experiences that are not explicitly
present in the resume text.

Distinguish between:
- EXPLICIT information: facts directly stated in the resume.
- INFERENCE: reasonable conclusions you draw (e.g. a missing quantified
  achievement implies a weakness in impact reporting). For inferred
  fields (strengths, weaknesses, improvement_suggestions,
  recommended_skills), ground each item in what is or is not present in
  the resume — never invent new facts about the candidate.

Scoring (overall_score, 0-100): score consistently on structure clarity,
quantified achievements, keyword coverage, and professional polish. A
sparse or vague resume must score low; a strong, well-evidenced resume
scores high.

If a section (e.g. projects, certifications, education) is absent from
the resume, return an EMPTY list for that field — do not fabricate
entries. If no professional summary can be derived, return an empty
string for professional_summary.

Return ONLY a JSON object matching this shape (no prose, no markdown):
{
  "overall_score": <integer 0-100>,
  "professional_summary": "<string or empty>",
  "skills": ["<string>", ...],
  "experience": [
    {"title": "", "company": "", "duration": "", "description": ""}
  ],
  "education": [
    {"degree": "", "institution": "", "year": "", "details": ""}
  ],
  "projects": [
    {"name": "", "description": "", "technologies": ""}
  ],
  "certifications": [
    {"name": "", "issuer": "", "year": ""}
  ],
  "strengths": ["<string>", ...],
  "weaknesses": ["<string>", ...],
  "improvement_suggestions": ["<string>", ...],
  "recommended_skills": ["<string>", ...]
}
"""

# OpenAPI-subset schema constraining Gemini's JSON output. All top-level
# fields are required so the model returns the full shape; nested object
# items have no required fields so the model can emit empty strings for
# any unknown sub-field.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "professional_summary": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "duration": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "string"},
                    "details": {"type": "string"},
                },
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "technologies": {"type": "string"},
                },
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "improvement_suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommended_skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_score",
        "professional_summary",
        "skills",
        "experience",
        "education",
        "projects",
        "certifications",
        "strengths",
        "weaknesses",
        "improvement_suggestions",
        "recommended_skills",
    ],
}
