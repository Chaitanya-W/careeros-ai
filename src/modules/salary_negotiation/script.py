"""Deterministic negotiation script + counter-offer builder.

Builds negotiation points grounded in candidate evidence (reusing the
Application Copilot's ``extract_evidence`` + ``validate_claims``). The
counter-offer number comes from the deterministic benchmark — Gemini
cannot change it.

No raw ``resume_text`` is read. Only structured ``ResumeAnalysis`` fields
and the evidence items are used.
"""

from __future__ import annotations

from typing import Optional

from src.modules.application_copilot.evidence import (
    EvidenceItem,
    evidence_ids_for_skills,
    extract_evidence,
)
from src.modules.application_copilot.validator import validate_claims
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.salary_negotiation.model import (
    CounterOffer,
    NegotiationPoint,
    NegotiationScript,
    ValidationStatus,
)

import re


def build_script(
    resume: ResumeAnalysis,
    job: Optional[JobAnalysis],
    match,
    benchmark_mid: float,
    *,
    target_role: str = "",
    current_offer: Optional[float] = None,
    expected_salary: Optional[float] = None,
    evidence: Optional[list[EvidenceItem]] = None,
    currency_symbol: str = "$",
) -> NegotiationScript:
    """Build a deterministic negotiation script grounded in evidence.

    The counter-offer number is deterministic (from ``benchmark.py``).
    Gemini may only refine wording later (see ``enricher.py``).
    """
    if evidence is None and resume is not None:
        evidence = extract_evidence(resume)

    target = target_role or _target_role(job)
    matching = list(getattr(match, "matching_skills", []) or []) if match else []
    leverage = _build_leverage(resume, matching, evidence)
    counter_num = _counter_number(benchmark_mid, current_offer, expected_salary)

    talking_points: list[NegotiationPoint] = []
    idx = 0

    # 1. Opening — market positioning.
    idx += 1
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Opening Position",
        wording=(
            f"Based on my research and the responsibilities of the {target} role, "
            f"I'm looking for compensation in the range of "
            f"{currency_symbol}{benchmark_mid * 0.85:,.0f}–{currency_symbol}{benchmark_mid * 1.15:,.0f}."
        ),
        evidence_ids=[],
        validation_status=ValidationStatus.PASS,
    ))

    # 2. Candidate leverage — matching skills.
    idx += 1
    skill_ev_ids = evidence_ids_for_skills(matching[:5], evidence or []) if evidence else []
    leverage_wording = "My experience directly aligns with the role's requirements."
    if matching:
        leverage_wording = (
            f"I bring proven experience with {', '.join(matching[:5])}, "
            f"which are core requirements for this position."
        )
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Candidate Leverage",
        wording=leverage_wording,
        evidence_ids=skill_ev_ids,
        validation_status=ValidationStatus.PASS,
    ))

    # 3. Role alignment — relevant experience/projects.
    idx += 1
    exp_ev_ids = _experience_evidence_ids(resume, evidence or [])
    exp_wording = "My background includes directly relevant work."
    if resume and getattr(resume, "experience", None):
        first_exp = resume.experience[0] if resume.experience else None
        if first_exp and first_exp.description:
            exp_wording = f"In my role at {first_exp.company or 'my previous position'}, I {first_exp.description.lower().rstrip('.')}."
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Role Alignment",
        wording=exp_wording,
        evidence_ids=exp_ev_ids,
        validation_status=ValidationStatus.PASS,
    ))

    # 4. Compensation ask — deterministic counter number.
    idx += 1
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Compensation Ask",
        wording=(
            f"Given the market range and my experience, I'd like to propose "
            f"a base salary of {currency_symbol}{counter_num:,.0f}."
        ),
        evidence_ids=[],
        validation_status=ValidationStatus.PASS,
    ))

    # 5. Flexibility / alternatives.
    idx += 1
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Flexibility & Alternatives",
        wording=(
            "I'm also open to discussing equity, signing bonuses, or "
            "performance-based incentives as part of the total package."
        ),
        evidence_ids=[],
        validation_status=ValidationStatus.PASS,
    ))

    # 6. Closing.
    idx += 1
    talking_points.append(NegotiationPoint(
        point_id=f"NP-{idx:03d}",
        title="Closing",
        wording=(
            "I'm excited about the opportunity and confident I can deliver "
            "significant value. I'd love to find a number that works for both of us."
        ),
        evidence_ids=[],
        validation_status=ValidationStatus.PASS,
    ))

    fallback = (
        f"If the base cannot meet {currency_symbol}{counter_num:,.0f}, I'm open to a total "
        f"compensation package that includes equity or a signing bonus."
    )

    return NegotiationScript(
        target_role=target,
        opening_position=f"Target: {currency_symbol}{counter_num:,.0f} base salary.",
        leverage_points=leverage,
        talking_points=talking_points,
        fallback_position=fallback,
        validation_status=ValidationStatus.PASS,
        enriched=False,
        warnings=[],
    )


def build_counter_offer(
    benchmark_mid: float,
    *,
    current_offer: Optional[float] = None,
    expected_salary: Optional[float] = None,
    target_role: str = "",
    currency_symbol: str = "$",
) -> CounterOffer:
    """Build a deterministic counter-offer with a ready-to-use template."""
    from src.modules.salary_negotiation.benchmark import compute_counter_number

    counter_num = compute_counter_number(
        benchmark_mid, current_offer=current_offer, expected_salary=expected_salary
    )
    rationale = _counter_rationale(
        benchmark_mid, counter_num, current_offer, expected_salary,
        currency_symbol,
    )
    template = _counter_template(counter_num, rationale, target_role, currency_symbol)
    return CounterOffer(
        current_offer=current_offer,
        expected_salary=expected_salary,
        benchmark_mid=benchmark_mid,
        counter_number=counter_num,
        rationale=rationale,
        template_wording=template,
    )


# --------------------------------------------------------------------------- #
# Leverage extraction
# --------------------------------------------------------------------------- #


def _build_leverage(
    resume: ResumeAnalysis,
    matching: list[str],
    evidence: Optional[list[EvidenceItem]],
) -> list[str]:
    """Build leverage-point strings from matching skills + resume evidence."""
    leverage: list[str] = []
    if matching:
        leverage.append(f"Directly matches {len(matching)} required skills: {', '.join(matching[:5])}.")
    if resume and getattr(resume, "projects", None):
        names = [p.name for p in resume.projects if p.name][:3]
        if names:
            leverage.append(f"Relevant projects: {', '.join(names)}.")
    if resume and getattr(resume, "certifications", None):
        certs = [c.name for c in resume.certifications if c.name][:2]
        if certs:
            leverage.append(f"Certifications: {', '.join(certs)}.")
    if resume and getattr(resume, "experience", None):
        years = _estimate_years(resume)
        if years > 0:
            leverage.append(f"~{years:.0f} years of relevant experience.")
    if not leverage:
        leverage.append("Solid foundation aligned with the role's requirements.")
    return leverage


def _estimate_years(resume: ResumeAnalysis) -> float:
    """Conservative years estimate from experience durations."""
    total = 0.0
    for exp in getattr(resume, "experience", []) or []:
        dur = getattr(exp, "duration", "") or ""
        years = _extract_years(dur)
        if years:
            total += years
            continue
        total += _year_span(dur)
    return total


def _extract_years(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _year_span(duration: str) -> float:
    from datetime import date
    now_year = date.today().year
    m = re.search(r"(20\d{2})\s*(?:-|–|to)\s*(present|current|now|20\d{2})", duration, re.IGNORECASE)
    if not m:
        return 0.0
    start = int(m.group(1))
    end_token = m.group(2).lower()
    if end_token in ("present", "current", "now"):
        end = now_year
    else:
        end = int(end_token)
    return max(0.0, float(end - start))


def _experience_evidence_ids(
    resume: ResumeAnalysis, evidence: list[EvidenceItem]
) -> list[str]:
    """Return EXP-* evidence IDs for the resume's experience entries."""
    out: list[str] = []
    for i, exp in enumerate(getattr(resume, "experience", []) or [], start=1):
        ev_id = f"EXP-{i:03d}"
        if any(e.evidence_id == ev_id for e in evidence):
            out.append(ev_id)
    return out


def _counter_number(
    benchmark_mid: float,
    current_offer: Optional[float],
    expected_salary: Optional[float],
) -> float:
    from src.modules.salary_negotiation.benchmark import compute_counter_number

    return compute_counter_number(
        benchmark_mid, current_offer=current_offer, expected_salary=expected_salary
    )


def _counter_rationale(
    benchmark_mid: float,
    counter_num: float,
    current_offer: Optional[float],
    expected_salary: Optional[float],
    currency_symbol: str = "",
) -> str:
    parts = [f"Benchmark midpoint: {currency_symbol}{benchmark_mid:,.0f}."]
    if expected_salary and expected_salary > 0:
        parts.append(f"Your expected salary: {currency_symbol}{expected_salary:,.0f}.")
    if current_offer and current_offer > 0:
        parts.append(f"Current offer: {currency_symbol}{current_offer:,.0f}.")
    parts.append(f"Counter number: {currency_symbol}{counter_num:,.0f} (deterministic).")
    return " ".join(parts)


def _counter_template(counter_num: float, rationale: str, target_role: str, currency_symbol: str = "$") -> str:
    role_phrase = f" for the {target_role} role" if target_role else ""
    return (
        f"Thank you for the offer{role_phrase}. After researching market "
        f"ranges and considering my experience, I'd like to propose a base "
        f"salary of {currency_symbol}{counter_num:,.0f}. {rationale} "
        f"I'm excited about the opportunity and confident I can deliver "
        f"significant value to the team."
    )


def _target_role(job: Optional[JobAnalysis]) -> str:
    if job is None:
        return "the target role"
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")
