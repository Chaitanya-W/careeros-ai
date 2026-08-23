"""Job-description analysis prompt + structured-output schema for Gemini.

The prompt is engineered to:
- use ONLY the supplied job-description text,
- distinguish REQUIRED vs PREFERRED skills,
- never invent company names, requirements, or skills,
- return empty values for absent information,
- return structured JSON.
"""

from __future__ import annotations

JOB_ANALYSIS_SYSTEM_PROMPT: str = """\
You are an expert technical recruiter and job-description analyst.

You will receive the plain-text content of a single job description.
Analyze ONLY the text supplied. Do NOT invent company names, job titles,
requirements, skills, or qualifications that are not explicitly present
in the job description.

Distinguish between:
- REQUIRED skills: explicitly stated as required / mandatory / must-have.
- PREFERRED skills: described as preferred / nice-to-have / "a plus" /
  "bonus". If the job description does not distinguish, put skills in
  required_skills and leave preferred_skills empty.

For experience_requirements and education_requirements, return a single
concise normalized string each (e.g. "3+ years of backend engineering" or
"B.S. in Computer Science or equivalent"). If absent, return an empty
string.

If any field has no information in the job description, return an empty
string or empty list — do not fabricate content.

Return ONLY a JSON object matching this shape (no prose, no markdown):
{
  "job_title": "",
  "company": "",
  "required_skills": ["", ...],
  "preferred_skills": ["", ...],
  "responsibilities": ["", ...],
  "experience_requirements": "",
  "education_requirements": "",
  "keywords": ["", ...],
  "domain": ""
}
"""

# OpenAPI-subset schema constraining Gemini's JSON output. All top-level
# fields are required so the model returns the full shape; JobAnalysis.
# from_dict is the robustness safety net for any malformed output.
JOB_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "job_title": {"type": "string"},
        "company": {"type": "string"},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "experience_requirements": {"type": "string"},
        "education_requirements": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "domain": {"type": "string"},
    },
    "required": [
        "job_title",
        "company",
        "required_skills",
        "preferred_skills",
        "responsibilities",
        "experience_requirements",
        "education_requirements",
        "keywords",
        "domain",
    ],
}
