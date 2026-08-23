"""Deterministic Skill Gap report builder.

Produces a :class:`SkillGapReport` from a :class:`ResumeAnalysis`, a
:class:`JobAnalysis`, and a :class:`MatchAnalysis` using **pure
deterministic logic** — no LLM call.

Single source of truth
----------------------
``MatchAnalysis.missing_skills`` is the single source of truth for REQUIRED
skill gaps. These are included verbatim (deduplicated, case-insensitively)
and are NEVER re-derived by AI. Additionally, PREFERRED-skill gaps are
derived deterministically from ``JobAnalysis.preferred_skills`` (skills not
present in the resume, excluding any already counted as required gaps).
Preferred skills are explicitly part of the job — they are not "unrelated"
skills. No skills beyond required + preferred are ever added; AI enrichment
only fills learning-guidance fields.

Prioritization (deterministic, NOT scientifically validated)
------------------------------------------------------------
+-------------------+----------+--------------------------------------------+
| Condition         | Priority | Rationale                                  |
+-------------------+----------+--------------------------------------------+
| in required_skills| HIGH     | A formal required gap is the most critical. |
| in preferred_skills (not required) | MEDIUM | A preferred gap is secondary.   |
| otherwise          | LOW      | Defensive — shouldn't occur; handled safely.|
+-------------------+----------+--------------------------------------------+

Current evidence
----------------
- Exact match in ``resume.skills`` -> "Explicitly present".
- Substring mention in the resume corpus (experience / projects / summary)
  -> "Related evidence found in resume".
- Otherwise -> "No evidence found in resume" (never claim unsupported
  ability).

Target proficiency
-------------------
Derived conservatively from ``job.experience_requirements`` (job-level,
not per-skill):
- mentions senior/lead/principal/staff -> "Advanced"
- mentions junior/entry/intern -> "Beginner"
- mentions a years figure (mid-level) -> "Intermediate"
- absent / unparseable -> "Not specified"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.skill_gap.model import (
    Priority,
    SkillGap,
    SkillGapReport,
)


def build_skill_gap_report(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,  # MatchAnalysis — imported lazily to avoid a hard cycle
) -> SkillGapReport:
    """Build a deterministic :class:`SkillGapReport`.

    ``match.missing_skills`` is the single source of truth for required
    gaps. Preferred gaps are derived from ``job.preferred_skills``. No AI.
    """
    required_gaps = _dedupe(match.missing_skills)
    preferred_gaps = _preferred_gaps(job, resume, exclude=required_gaps)

    target_role = _target_role(job)
    target_level = _target_level(job)

    skill_gaps: list[SkillGap] = []
    for skill in required_gaps + preferred_gaps:
        priority = _prioritize(skill, job)
        evidence, current_level = _detect_evidence(skill, resume)
        skill_gaps.append(
            SkillGap(
                skill=skill,
                priority=priority,
                current_level=current_level,
                target_level=target_level,
                current_evidence=evidence,
                gap_reason=_gap_reason(skill, job),
            )
        )

    counts = _count_by_priority(skill_gaps)
    total = len(skill_gaps)
    return SkillGapReport(
        target_role=target_role,
        total_gaps=total,
        high_priority_count=counts[Priority.HIGH],
        medium_priority_count=counts[Priority.MEDIUM],
        low_priority_count=counts[Priority.LOW],
        skill_gaps=skill_gaps,
        summary=_build_summary(target_role, total, counts),
        generated_at=datetime.now(timezone.utc).isoformat(),
        enriched=False,
    )


# --------------------------------------------------------------------------- #
# Gap-list construction
# --------------------------------------------------------------------------- #


def _dedupe(skills: list[str]) -> list[str]:
    """Deduplicate case-insensitively, preserving order + canonical names."""
    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        key = _normalize(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _preferred_gaps(
    job: JobAnalysis,
    resume: ResumeAnalysis,
    exclude: list[str],
) -> list[str]:
    """Preferred skills not present in the resume (excluding required gaps).

    A preferred skill is a gap if it is not an exact (normalized) match in
    the resume's skills. Related-but-not-exact evidence is captured in the
    ``current_evidence`` field rather than suppressing the gap.
    """
    resume_norm = {_normalize(s) for s in resume.skills}
    exclude_norm = {_normalize(s) for s in exclude}
    seen: set[str] = set(exclude_norm)
    out: list[str] = []
    for ps in job.preferred_skills:
        n = _normalize(ps)
        if not n or n in seen:
            continue
        if n in resume_norm:
            continue  # resume has it -> not a gap
        seen.add(n)
        out.append(ps)
    return out


# --------------------------------------------------------------------------- #
# Prioritization
# --------------------------------------------------------------------------- #


def _prioritize(skill: str, job: JobAnalysis) -> Priority:
    """Deterministic priority (see module docstring)."""
    s = _normalize(skill)
    if not s:
        return Priority.LOW
    if s in {_normalize(x) for x in job.required_skills}:
        return Priority.HIGH
    if s in {_normalize(x) for x in job.preferred_skills}:
        return Priority.MEDIUM
    return Priority.LOW  # defensive


def _count_by_priority(gaps: list[SkillGap]) -> dict[Priority, int]:
    return {
        Priority.HIGH: sum(1 for g in gaps if g.priority is Priority.HIGH),
        Priority.MEDIUM: sum(1 for g in gaps if g.priority is Priority.MEDIUM),
        Priority.LOW: sum(1 for g in gaps if g.priority is Priority.LOW),
    }


# --------------------------------------------------------------------------- #
# Current evidence
# --------------------------------------------------------------------------- #


def _detect_evidence(
    skill: str, resume: ResumeAnalysis
) -> tuple[str, str]:
    """Return (evidence_text, current_level) for a skill.

    Never claims unsupported ability — defaults to "No evidence found".
    """
    s = _normalize(skill)
    if not s:
        return ("No evidence found in resume", "No evidence found in resume")
    if s in {_normalize(x) for x in resume.skills}:
        return ("Explicitly present in resume skills.", "Explicitly present")
    corpus = _resume_corpus(resume).lower()
    if s and s in corpus:
        return (
            "Related evidence found in resume.",
            "Related evidence found",
        )
    return ("No evidence found in resume", "No evidence found in resume")


def _resume_corpus(resume: ResumeAnalysis) -> str:
    """Build a text corpus from ResumeAnalysis fields (not raw resume text)."""
    chunks: list[str] = []
    if resume.professional_summary:
        chunks.append(resume.professional_summary)
    chunks.extend(resume.skills)
    for exp in resume.experience:
        chunks.extend(p for p in (exp.title, exp.company, exp.description) if p)
    for edu in resume.education:
        chunks.extend(p for p in (edu.degree, edu.institution, edu.details) if p)
    for proj in resume.projects:
        chunks.extend(
            p for p in (proj.name, proj.description, proj.technologies) if p
        )
    for cert in resume.certifications:
        chunks.extend(p for p in (cert.name, cert.issuer) if p)
    return " ".join(chunks)


# --------------------------------------------------------------------------- #
# Target proficiency + role
# --------------------------------------------------------------------------- #


_SENIOR_TOKENS = ("senior", "lead", "principal", "staff", "expert", "architect")
_JUNIOR_TOKENS = ("junior", "entry", "intern", "graduate", "entry-level")


def _target_level(job: JobAnalysis) -> str:
    """Conservative job-level target proficiency."""
    req = (job.experience_requirements or "").lower()
    if not req:
        return "Not specified"
    if any(tok in req for tok in _SENIOR_TOKENS):
        return "Advanced"
    if any(tok in req for tok in _JUNIOR_TOKENS):
        return "Beginner"
    if _extract_years(req) is not None:
        return "Intermediate"
    return "Not specified"


def _target_role(job: JobAnalysis) -> str:
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")


def _gap_reason(skill: str, job: JobAnalysis) -> str:
    s = _normalize(skill)
    if s in {_normalize(x) for x in job.required_skills}:
        return "Listed as a required skill in the job description but not found in your resume."
    if s in {_normalize(x) for x in job.preferred_skills}:
        return "Listed as a preferred skill but not evidenced in your resume."
    return "Identified as a gap relative to the target role."


def _build_summary(
    target_role: str, total: int, counts: dict[Priority, int]
) -> str:
    return (
        f"Identified {total} skill gap(s) for the {target_role} role: "
        f"{counts[Priority.HIGH]} high, {counts[Priority.MEDIUM]} medium, "
        f"{counts[Priority.LOW]} low priority."
    )


# --------------------------------------------------------------------------- #
# Low-level helpers (mirror the matcher's normalization; kept local to
# avoid cross-module private imports).
# --------------------------------------------------------------------------- #


def _normalize(skill: str) -> str:
    if not skill:
        return ""
    s = skill.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b\d+(\.\d+)*\+?\b", "", s)
    s = re.sub(r"[^a-z0-9+#./\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_years(text: str) -> Optional[float]:
    if not text:
        return None
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
