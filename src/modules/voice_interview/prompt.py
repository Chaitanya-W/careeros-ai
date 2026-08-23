"""Gemini prompts + JSON schemas for the Interview Coach.

Both system prompts enforce evidence-grounding: Gemini must NOT invent
candidate facts (projects, technologies, employers, certifications,
metrics, dates, responsibilities) and must NOT reward unsupported claims.
"""

from __future__ import annotations

QUESTION_GEN_SYSTEM_PROMPT: str = """\
You are an evidence-grounded interview coach.

Generate interview questions ONLY from the supplied candidate evidence and
job requirements. Do NOT invent candidate experience, projects, technologies,
employers, certifications, metrics, dates, or responsibilities.

For each question:
- If it references a skill the candidate has, frame it around their experience.
- If it references a job-required skill the candidate LACKS, ask how they
  would approach/learn it — do NOT claim the candidate already has it.
- Prefer questions that expose depth, reasoning, tradeoffs, and problem-solving.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "questions": [
    {
      "question": "",
      "category": "technical|behavioral|project|resume|job_specific|skill_gap",
      "difficulty": "easy|medium|hard",
      "target_skill": "",
      "rationale": ""
    }
  ]
}
"""

QUESTION_GEN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "category": {"type": "string"},
                    "difficulty": {"type": "string"},
                    "target_skill": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["question", "category"],
            },
        },
    },
    "required": ["questions"],
}

ANSWER_EVAL_SYSTEM_PROMPT: str = """\
You are an evidence-grounded interview coach evaluating a candidate's answer.

Evaluate the answer against the question, using the supplied job context and
candidate evidence. Score each dimension 0-10 (heuristic coaching scores,
not psychometric).

CRITICAL RULES:
- Distinguish candidate claims from known resume evidence.
- Do NOT reward unsupported claims as established facts.
- Do NOT invent facts about the candidate.
- Provide actionable, specific coaching.
- Return structured JSON.

Return ONLY a JSON object (no prose, no markdown) of this shape:
{
  "relevance_score": 0,
  "technical_score": 0,
  "specificity_score": 0,
  "communication_score": 0,
  "completeness_score": 0,
  "overall_score": 0,
  "strengths": ["", ...],
  "improvements": ["", ...],
  "missing_points": ["", ...],
  "evidence_alignment": "aligned|partial|unsupported",
  "unsupported_claims": ["", ...],
  "recommendation": ""
}
"""

ANSWER_EVAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "relevance_score": {"type": "number"},
        "technical_score": {"type": "number"},
        "specificity_score": {"type": "number"},
        "communication_score": {"type": "number"},
        "completeness_score": {"type": "number"},
        "overall_score": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "evidence_alignment": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
    },
    "required": ["overall_score"],
}
