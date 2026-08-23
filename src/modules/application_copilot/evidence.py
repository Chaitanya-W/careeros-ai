"""Deterministic evidence extraction from ResumeAnalysis.

Builds :class:`EvidenceItem` objects with **stable IDs** (EXP-001,
PROJ-001, SKILL-001, EDU-001, CERT-001, SUMMARY-001) from the existing
ResumeAnalysis fields. No Gemini — facts come only from the structured
resume analysis.

If a section is absent from the resume, no evidence is invented for it.
"""

from __future__ import annotations

from typing import Optional

from src.modules.application_copilot.model import EvidenceItem
from src.modules.resume_intelligence.model import ResumeAnalysis


def extract_evidence(resume: ResumeAnalysis) -> list[EvidenceItem]:
    """Build evidence items from ResumeAnalysis fields.

    IDs are stable and deterministic: enumerated in field order with a
    zero-padded index (EXP-001, EXP-002, ...). Re-running on the same
    resume yields the same IDs.
    """
    items: list[EvidenceItem] = []

    for i, exp in enumerate(resume.experience, start=1):
        text = " ".join(
            p for p in (exp.title, exp.company, exp.duration, exp.description) if p
        ).strip()
        if not text:
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"EXP-{i:03d}",
                source_type="experience",
                source_text=text,
                category="experience",
            )
        )

    for i, proj in enumerate(resume.projects, start=1):
        text = " ".join(
            p for p in (proj.name, proj.description, proj.technologies) if p
        ).strip()
        if not text:
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"PROJ-{i:03d}",
                source_type="project",
                source_text=text,
                category="project",
            )
        )

    for i, skill in enumerate(resume.skills, start=1):
        s = skill.strip() if skill else ""
        if not s:
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"SKILL-{i:03d}",
                source_type="skill",
                source_text=s,
                category="skill",
            )
        )

    for i, edu in enumerate(resume.education, start=1):
        text = " ".join(
            p for p in (edu.degree, edu.institution, edu.year, edu.details) if p
        ).strip()
        if not text:
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"EDU-{i:03d}",
                source_type="education",
                source_text=text,
                category="education",
            )
        )

    for i, cert in enumerate(resume.certifications, start=1):
        text = " ".join(
            p for p in (cert.name, cert.issuer, cert.year) if p
        ).strip()
        if not text:
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"CERT-{i:03d}",
                source_type="certification",
                source_text=text,
                category="certification",
            )
        )

    if resume.professional_summary:
        items.append(
            EvidenceItem(
                evidence_id="SUMMARY-001",
                source_type="summary",
                source_text=resume.professional_summary.strip(),
                category="summary",
            )
        )

    return items


def evidence_corpus(evidence: list[EvidenceItem]) -> str:
    """Return the lowercased concatenation of all source_text (for matching)."""
    return " ".join(e.source_text for e in evidence).lower()


def evidence_ids_for_skills(
    skills: list[str], evidence: list[EvidenceItem]
) -> list[str]:
    """Return the SKILL-* evidence IDs whose source_text matches a skill."""
    wanted = {_normalize(s) for s in skills if s}
    out: list[str] = []
    for e in evidence:
        if e.source_type == "skill" and _normalize(e.source_text) in wanted:
            out.append(e.evidence_id)
    return out


def evidence_ids_for_experience_text(
    text: str, resume: ResumeAnalysis, evidence: list[EvidenceItem]
) -> list[str]:
    """Return the EXP-* evidence ID whose experience description overlaps text."""
    text_lower = (text or "").lower()
    out: list[str] = []
    for i, exp in enumerate(resume.experience, start=1):
        if exp.description and exp.description.strip().lower() in text_lower:
            ev_id = f"EXP-{i:03d}"
            if any(e.evidence_id == ev_id for e in evidence):
                out.append(ev_id)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize(s: str) -> str:
    if not s:
        return ""
    import re

    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9+#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
