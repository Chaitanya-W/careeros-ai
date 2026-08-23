"""Deterministic question validator.

Validates interview questions (especially Gemini-generated ones) against
the candidate evidence to ensure they never invent candidate facts.

Reuses the Application Copilot's :func:`validate_claims` (Step 7 evidence
architecture) for metric / year / cert checks — passing ``match=None`` so
the missing-skill check is skipped (handled separately below, phrasing-aware).

A question is INVALID if:
- ``validate_claims`` returns unsupported_claims (invented metric/year/cert),
  OR
- the question references a proper-noun entity not in the resume evidence
  AND not a job-required/preferred skill (job-specific questions may ask
  about a required skill even if the candidate lacks it),
  OR
- the question ASSERTS the candidate has a missing skill (assertion phrasing
  near a missing-skill mention).
"""

from __future__ import annotations

import re
from typing import Optional

from src.modules.application_copilot.evidence import EvidenceItem, evidence_corpus
from src.modules.application_copilot.validator import (
    _extract_proper_nouns,
    _stopword_set,
    validate_claims,
)

# Phrases that ASSERT the candidate has/used a skill. If a missing skill
# appears in a question that also contains one of these, the question is
# (likely) claiming the candidate has the skill -> INVALID.
_ASSERTION_PATTERNS = (
    "you used", "you have used", "your experience with", "you built",
    "you deployed", "you managed", "you implemented", "you led",
    "you developed", "you wrote", "you designed", "you maintained",
    "you operated", "you scaled", "you shipped",
)


def validate_question(
    question_text: str,
    evidence: list[EvidenceItem],
    resume,
    job,
    match,
    skill_gap_report=None,
) -> tuple[bool, list[str]]:
    """Return ``(is_valid, reasons)``.

    ``is_valid`` is True if the question is grounded; False if it invents
    candidate facts or asserts a skill the candidate lacks.
    """
    if not question_text or not question_text.strip():
        return False, ["Empty question."]

    # 1. Metric / year / cert checks via the Copilot validator (match=None
    #    skips the missing-skill check). unsupported_claims -> invalid.
    v = validate_claims(question_text, evidence, resume=resume, job=job, match=None)
    if v.unsupported_claims:
        return False, list(v.unsupported_claims)

    # 2. Proper-noun check that EXCLUDES job required/preferred skills
    #    (a job-specific question may legitimately mention a required skill).
    stopwords = _stopword_set(job)
    if job is not None:
        for s in list(getattr(job, "required_skills", []) or []) + list(
            getattr(job, "preferred_skills", []) or []
        ):
            for tok in re.split(r"[^A-Za-z0-9+#]+", s):
                if tok:
                    stopwords.add(tok.lower())
    corpus = evidence_corpus(evidence)
    for pn in _extract_proper_nouns(question_text):
        key = pn.lower()
        if key in stopwords or key in corpus:
            continue
        return False, [
            f"Question references '{pn}' which is not in resume evidence."
        ]

    # 3. Phrasing-aware lacked-skill assertion check. Considers skills the
    #    candidate lacks (required gaps from match.missing_skills + preferred
    #    gaps from the skill_gap_report).
    candidate_lacks = list(getattr(match, "missing_skills", []) or []) if match else []
    if skill_gap_report is not None:
        for g in getattr(skill_gap_report, "skill_gaps", []) or []:
            s = getattr(g, "skill", "")
            if s and s not in candidate_lacks:
                candidate_lacks.append(s)
    asserted = _asserts_lacked_skill(question_text, candidate_lacks)
    if asserted is not None:
        return False, [
            f"Question asserts the candidate has '{asserted}', which is a "
            f"skill the candidate lacks — the candidate may not have it."
        ]

    return True, []


def _asserts_lacked_skill(question_text: str, lacked_skills: list[str]) -> Optional[str]:
    """Return the lacked skill the question ASSERTS the candidate has, or None."""
    q_lower = question_text.lower()
    has_assertion = any(pat in q_lower for pat in _ASSERTION_PATTERNS)
    if not has_assertion:
        return None
    for skill in lacked_skills:
        s = skill.lower().strip()
        if s and s in q_lower:
            return skill
    return None
