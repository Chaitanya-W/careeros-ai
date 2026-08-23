"""Deterministic resume-vs-job matcher.

Produces a :class:`MatchAnalysis` from a :class:`ResumeAnalysis` and a
:class:`JobAnalysis` using **pure deterministic logic** — no LLM call.

Skill overlap, missing-skill detection, partial-skill handling, and
score calculation are all rule-based so the behavior is fully testable
and reproducible. The only Gemini usage in the Job Match feature is
:class:`~src.modules.jd_analysis.analyzer.analyze_job_description`
(structured JD extraction); this matcher never calls Gemini.

Scoring
-------
The overall match score is a weighted average over **active** component
scores only. A component is active when the job analysis supplies data
for it (e.g. ``required_skills`` is active only if the job lists any).
Weights are tunable and **not** scientifically validated — they are a
documented, opinionated default.

+--------------------+--------+------------------------------------------------+
| Component          | Weight | Score (0-100) derivation                       |
+--------------------+--------+------------------------------------------------+
| required_skills    | 0.45   | 100 * (exact + 0.5*partial) / total_required   |
| preferred_skills   | 0.15   | 100 * (exact + 0.5*partial) / total_preferred  |
| experience         | 0.20   | years-ratio where parseable, else neutral 50   |
| education          | 0.10   | keyword overlap between reqs & resume degrees  |
| keywords           | 0.10   | 100 * job keywords found in resume / total      |
+--------------------+--------+------------------------------------------------+

overall = round( sum(weight_i * score_i for active i) /
                 sum(weight_i  for active i) )        # 0-100

If no component is active, the overall score is 0.

Partial-skill matching
----------------------
A required/preferred skill is a **partial** match when the shorter
*normalized* skill is a complete token of the longer one (split on
non-alphanumeric characters), with a minimum token length of 3. This
captures pairs like ("AWS", "AWS Lambda") and ("React", "React Native")
while avoiding false positives like ("Java", "JavaScript") — because
"java" is NOT a token of "javascript".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.modules.jd_analysis.model import JobAnalysis, MatchAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis

# Tunable weights (sum to 1.0 over all components). NOT scientifically
# validated — see module docstring.
WEIGHTS: dict[str, float] = {
    "required_skills": 0.45,
    "preferred_skills": 0.15,
    "experience": 0.20,
    "education": 0.10,
    "keywords": 0.10,
}

# Minimum normalized length for a token-subset partial match. Avoids
# noisy matches on tiny tokens (e.g. "C", "JS").
_PARTIAL_MIN_TOKEN_LEN: int = 3


@dataclass
class _Component:
    """An intermediate scoring component."""

    name: str
    score: int  # 0-100
    active: bool
    weight: float


def match_jobs(resume: ResumeAnalysis, job: JobAnalysis) -> MatchAnalysis:
    """Deterministically compare ``resume`` against ``job``.

    Returns a fully-populated :class:`MatchAnalysis` including the overall
    score, skill classifications, per-component scores, deterministic
    strengths/gaps, a template explanation, and recommended actions.
    """
    # --- required-skill classification (drives the public lists) ---------
    matching, partial, missing = _classify_skills(
        resume.skills, job.required_skills
    )

    # --- preferred-skill classification (component score only) -----------
    pref_match, pref_partial, _ = _classify_skills(
        resume.skills, job.preferred_skills
    )

    # --- component scores ------------------------------------------------
    required = _Component(
        "required_skills",
        _skill_score(job.required_skills, matching, partial),
        active=bool(job.required_skills),
        weight=WEIGHTS["required_skills"],
    )
    preferred = _Component(
        "preferred_skills",
        _skill_score(job.preferred_skills, pref_match, pref_partial),
        active=bool(job.preferred_skills),
        weight=WEIGHTS["preferred_skills"],
    )
    experience = _Component(
        "experience",
        _experience_score(resume, job),
        active=bool(job.experience_requirements.strip()),
        weight=WEIGHTS["experience"],
    )
    education = _Component(
        "education",
        _education_score(resume, job),
        active=bool(job.education_requirements.strip()),
        weight=WEIGHTS["education"],
    )
    keywords = _Component(
        "keywords",
        _keyword_score(resume, job),
        active=bool(job.keywords),
        weight=WEIGHTS["keywords"],
    )

    overall = _weighted_overall(
        [required, preferred, experience, education, keywords]
    )

    strengths = _build_strengths(
        matching, partial, required, preferred, experience, keywords
    )
    gaps = _build_gaps(missing, required, experience, education, keywords)
    explanation = _build_explanation(
        overall, matching, partial, missing, required, preferred,
        experience, education, keywords,
    )
    actions = _build_recommended_actions(
        missing, partial, pref_match, keywords, education
    )

    return MatchAnalysis(
        overall_match_score=overall,
        matching_skills=matching,
        missing_skills=missing,
        partial_match_skills=partial,
        experience_match=experience.score,
        education_match=education.score,
        keyword_match=keywords.score,
        strengths=strengths,
        gaps=gaps,
        explanation=explanation,
        recommended_actions=actions,
    )


# --------------------------------------------------------------------------- #
# Skill normalization + classification
# --------------------------------------------------------------------------- #


def _normalize_skill(skill: str) -> str:
    """Normalize a skill for comparison.

    Lowercase, strip parenthetical notes and version numbers, collapse
    whitespace, keep alphanumerics and ``+ # . / -``.
    """
    if not skill:
        return ""
    s = skill.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)  # drop (notes)
    s = re.sub(r"\b\d+(\.\d+)*\+?\b", "", s)  # drop version numbers
    s = re.sub(r"[^a-z0-9+#./\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_partial_match(a: str, b: str) -> bool:
    """Token-subset partial match (see module docstring)."""
    a_n = _normalize_skill(a)
    b_n = _normalize_skill(b)
    if not a_n or not b_n or a_n == b_n:
        return False
    shorter, longer = (a_n, b_n) if len(a_n) <= len(b_n) else (b_n, a_n)
    if len(shorter) < _PARTIAL_MIN_TOKEN_LEN:
        return False
    tokens = {t for t in re.split(r"[^a-z0-9+#]+", longer) if t}
    return shorter in tokens


def _classify_skills(
    resume_skills: list[str],
    target_skills: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Classify ``target_skills`` against ``resume_skills``.

    Returns ``(matching, partial, missing)`` — each a list of the
    ORIGINAL (un-normalized) target-skill strings, preserving order.
    """
    matching: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    resume_norm = [_normalize_skill(s) for s in resume_skills]
    resume_norm_set = {n for n in resume_norm if n}
    for target in target_skills:
        t_norm = _normalize_skill(target)
        if not t_norm:
            continue
        if t_norm in resume_norm_set:
            matching.append(target)
        elif any(
            _is_partial_match(target, r) for r in resume_skills
        ):
            partial.append(target)
        else:
            missing.append(target)
    return matching, partial, missing


# --------------------------------------------------------------------------- #
# Component scores
# --------------------------------------------------------------------------- #


def _skill_score(
    total: list[str],
    exact: list[str],
    partial_matches: list[str],
) -> int:
    """Skill-overlap score: exact matches count 1.0, partial count 0.5."""
    if not total:
        return 0
    score = (len(exact) + 0.5 * len(partial_matches)) / len(total)
    return int(round(score * 100))


def _experience_score(resume: ResumeAnalysis, job: JobAnalysis) -> int:
    """Experience score based on parseable years-ratio.

    - Job has no requirement -> inactive (caller handles).
    - Job requires years but resume has no experience entries -> 0.
    - Required years parseable AND resume years parseable -> ratio*100.
    - Otherwise (can't quantify) -> 50 (neutral).
    """
    req = job.experience_requirements.strip()
    if not req:
        return 0
    if not resume.experience:
        return 0
    required_years = _extract_years(req)
    resume_years = _estimate_resume_years(resume)
    if required_years and resume_years:
        ratio = resume_years / required_years
        return max(0, min(100, int(round(ratio * 100))))
    return 50  # has experience on both sides but can't quantify


def _education_score(resume: ResumeAnalysis, job: JobAnalysis) -> int:
    """Education score: keyword overlap between job reqs and resume degrees.

    - Job has no education requirement -> inactive.
    - Requirement present, resume has no education -> 0.
    - Overlap found -> 100; else -> 50 (has education, different field).
    """
    req = job.education_requirements.strip()
    if not req:
        return 0
    if not resume.education:
        return 0
    req_tokens = _tokens(req)
    if not req_tokens:
        return 50
    resume_edu = " ".join(
        " ".join(p for p in (e.degree, e.institution, e.details) if p)
        for e in resume.education
    )
    resume_tokens = _tokens(resume_edu)
    if req_tokens & resume_tokens:
        return 100
    return 50


def _keyword_score(resume: ResumeAnalysis, job: JobAnalysis) -> int:
    """Keyword score: fraction of job keywords present in the resume corpus."""
    if not job.keywords:
        return 0
    corpus = _resume_corpus(resume)
    corpus_lower = corpus.lower()
    found = 0
    for kw in job.keywords:
        kw_norm = kw.strip().lower()
        if kw_norm and kw_norm in corpus_lower:
            found += 1
    return int(round((found / len(job.keywords)) * 100))


# --------------------------------------------------------------------------- #
# Overall score
# --------------------------------------------------------------------------- #


def _weighted_overall(components: list[_Component]) -> int:
    """Weighted average over active components; 0 if none active."""
    active = [c for c in components if c.active]
    if not active:
        return 0
    total_weight = sum(c.weight for c in active)
    if total_weight <= 0:
        return 0
    weighted = sum(c.score * c.weight for c in active)
    return max(0, min(100, int(round(weighted / total_weight))))


# --------------------------------------------------------------------------- #
# Deterministic narrative builders
# --------------------------------------------------------------------------- #


def _build_strengths(
    matching: list[str],
    partial: list[str],
    required: _Component,
    preferred: _Component,
    experience: _Component,
    keywords: _Component,
) -> list[str]:
    strengths: list[str] = []
    if matching:
        strengths.append(
            "Strong required-skill overlap: " + ", ".join(matching[:6])
        )
    if partial:
        strengths.append(
            "Related skills already on the resume: " + ", ".join(partial[:6])
        )
    if experience.active and experience.score >= 70:
        strengths.append("Experience appears to meet the role's expectations.")
    if keywords.active and keywords.score >= 70:
        strengths.append("High keyword alignment with the job description.")
    return strengths


def _build_gaps(
    missing: list[str],
    required: _Component,
    experience: _Component,
    education: _Component,
    keywords: _Component,
) -> list[str]:
    gaps: list[str] = []
    if missing:
        gaps.append("Missing required skills: " + ", ".join(missing[:6]))
    if required.active and required.score < 50:
        gaps.append("Required-skill coverage is below 50%.")
    if experience.active and experience.score < 50:
        gaps.append("Experience may fall short of the role's requirement.")
    if education.active and education.score < 50:
        gaps.append("Education does not clearly match the requirement.")
    if keywords.active and keywords.score < 50:
        gaps.append("Low keyword alignment with the job description.")
    return gaps


def _build_explanation(
    overall: int,
    matching: list[str],
    partial: list[str],
    missing: list[str],
    required: _Component,
    preferred: _Component,
    experience: _Component,
    education: _Component,
    keywords: _Component,
) -> str:
    parts: list[str] = [
        f"Overall match: {overall}%.",
        f"{len(matching)} matching, {len(partial)} partial, "
        f"{len(missing)} missing required skills.",
    ]
    if experience.active:
        parts.append(f"Experience match: {experience.score}%.")
    if education.active:
        parts.append(f"Education match: {education.score}%.")
    if keywords.active:
        parts.append(f"Keyword alignment: {keywords.score}%.")
    return " ".join(parts)


def _build_recommended_actions(
    missing: list[str],
    partial: list[str],
    pref_match: list[str],
    keywords: _Component,
    education: _Component,
) -> list[str]:
    actions: list[str] = []
    if missing:
        top = ", ".join(missing[:3])
        actions.append(f"Learn or highlight these missing skills: {top}.")
    if partial:
        actions.append(
            "Clarify depth in related skills: " + ", ".join(partial[:3]) + "."
        )
    if keywords.active and keywords.score < 70:
        actions.append(
            "Add more job-specific keywords to your resume where truthful."
        )
    if education.active and education.score < 70:
        actions.append(
            "Make your education entries explicit and aligned with the "
            "role's requirement."
        )
    if not actions:
        actions.append(
            "Your resume aligns well with this role — tailor wording to "
            "the job description before applying."
        )
    return actions


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of length >= 3 from ``text``."""
    if not text:
        return set()
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 3}


def _resume_corpus(resume: ResumeAnalysis) -> str:
    """Build a text corpus from the ResumeAnalysis fields (not raw text)."""
    chunks: list[str] = []
    if resume.professional_summary:
        chunks.append(resume.professional_summary)
    chunks.extend(resume.skills)
    for exp in resume.experience:
        chunks.extend(p for p in (exp.title, exp.company, exp.description) if p)
    for edu in resume.education:
        chunks.extend(
            p for p in (edu.degree, edu.institution, edu.details) if p
        )
    for proj in resume.projects:
        chunks.extend(
            p for p in (proj.name, proj.description, proj.technologies) if p
        )
    for cert in resume.certifications:
        chunks.extend(p for p in (cert.name, cert.issuer) if p)
    return " ".join(chunks)


def _extract_years(text: str) -> Optional[float]:
    """Extract a year count from a requirement string.

    Handles "3+ years", "3 years", "5-7 years" (returns 5). Returns None
    if no parseable year count is found.
    """
    if not text:
        return None
    # range like "5-7 years" or "5 to 7 years" -> lower bound
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*years?",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _estimate_resume_years(resume: ResumeAnalysis) -> float:
    """Estimate total years of experience from duration strings.

    Sums explicit "N years" patterns and year-span date ranges. This is
    a conservative heuristic, not an authoritative calculation.
    """
    total = 0.0
    for exp in resume.experience:
        if not exp.duration:
            continue
        years = _extract_years(exp.duration)
        if years:
            total += years
            continue
        # date range like "2021-2023" or "2021-present"
        total += _year_span(exp.duration)
    return total


def _year_span(duration: str) -> float:
    """Estimate years from a date-range duration string."""
    now_year = date.today().year  # dynamic current year for "present" ranges
    m = re.search(
        r"(20\d{2})\s*(?:-|–|to)\s*(present|current|now|20\d{2})",
        duration,
        re.IGNORECASE,
    )
    if not m:
        return 0.0
    start = int(m.group(1))
    end_token = m.group(2).lower()
    if end_token in ("present", "current", "now"):
        end = now_year
    else:
        end = int(end_token)
    span = end - start
    return max(0.0, float(span))
