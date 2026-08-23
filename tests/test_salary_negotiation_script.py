"""Tests for the deterministic negotiation script + counter-offer builder."""

from __future__ import annotations

import pytest

from src.modules.salary_negotiation.benchmark import compute_benchmark
from src.modules.salary_negotiation.script import build_counter_offer, build_script


# ---- script generation --------------------------------------------


def test_build_script_has_talking_points(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence)
    assert len(s.talking_points) >= 6
    assert all(tp.point_id for tp in s.talking_points)
    assert all(tp.wording for tp in s.talking_points)
    assert not s.enriched  # deterministic


def test_build_script_has_leverage_points(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior")
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    assert len(s.leverage_points) >= 1


def test_build_script_evidence_ids_present(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior")
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    # At least one talking point should have evidence IDs.
    assert any(tp.evidence_ids for tp in s.talking_points)


def test_build_script_point_ids_sequential(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark()
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    ids = [tp.point_id for tp in s.talking_points]
    assert ids == [f"NP-{i:03d}" for i in range(1, len(ids) + 1)]


def test_build_script_deterministic(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior")
    a = build_script(resume, job, match, bm.mid, evidence=evidence)
    b = build_script(resume, job, match, bm.mid, evidence=evidence)
    assert [tp.wording for tp in a.talking_points] == [tp.wording for tp in b.talking_points]


def test_build_script_has_fallback(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark()
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    assert s.fallback_position


def test_build_script_no_raw_resume_text(sample_salary_inputs) -> None:
    """The script must not include raw resume_text."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark()
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    # The script only uses structured fields (experience descriptions, skills).
    # There should be no raw resume_text dump.
    all_text = " ".join(tp.wording for tp in s.talking_points)
    assert "Built Python services on AWS serving 100k users" not in all_text


# ---- leverage extraction ------------------------------------------


def test_leverage_includes_matching_skills(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark()
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    joined = " ".join(s.leverage_points)
    assert "Python" in joined or "SQL" in joined or "AWS" in joined


def test_leverage_includes_projects(sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark()
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    joined = " ".join(s.leverage_points)
    assert "CareerOS" in joined


# ---- counter-offer ------------------------------------------------


def test_build_counter_offer_deterministic() -> None:
    a = build_counter_offer(150000, current_offer=100000, expected_salary=140000)
    b = build_counter_offer(150000, current_offer=100000, expected_salary=140000)
    assert a.counter_number == b.counter_number


def test_build_counter_offer_has_template() -> None:
    c = build_counter_offer(150000, target_role="Eng")
    assert c.template_wording
    assert "$" in c.template_wording


def test_build_counter_offer_rationale() -> None:
    c = build_counter_offer(150000, current_offer=120000)
    assert c.rationale
    assert "150,000" in c.rationale or "150000" in c.rationale


def test_build_counter_offer_no_offer_no_expected() -> None:
    c = build_counter_offer(150000)
    assert c.counter_number == 150000


# ---- missing prerequisites ---------------------------------------


def test_build_script_no_job() -> None:
    """Without a job, script still works (role defaults)."""
    from src.modules.resume_intelligence.model import ResumeAnalysis
    from src.modules.application_copilot.evidence import extract_evidence

    resume = ResumeAnalysis(skills=["Python"])
    evidence = extract_evidence(resume)
    s = build_script(resume, None, None, 100000, evidence=evidence)
    assert s.talking_points


def test_build_script_no_match() -> None:
    from src.modules.resume_intelligence.model import ResumeAnalysis
    from src.modules.application_copilot.evidence import extract_evidence

    resume = ResumeAnalysis(skills=["Python"])
    evidence = extract_evidence(resume)
    s = build_script(resume, None, None, 100000, evidence=evidence)
    assert s.talking_points
    # No matching skills → leverage still has a fallback.
    assert s.leverage_points


# ---- INR currency support in script + counter-offer ---------------


def test_build_script_inr_currency(sample_salary_inputs) -> None:
    """Script uses ₹ symbol when currency_symbol='₹'."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote", market="India")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence, currency_symbol="₹")
    all_text = " ".join(tp.wording for tp in s.talking_points)
    assert "₹" in all_text
    assert "$" not in all_text


def test_build_script_usd_currency(sample_salary_inputs) -> None:
    """Script uses $ symbol when currency_symbol='$' (default)."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote", market="United States")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence, currency_symbol="$")
    all_text = " ".join(tp.wording for tp in s.talking_points)
    assert "$" in all_text


def test_build_counter_offer_inr() -> None:
    c = build_counter_offer(800000, target_role="Eng", currency_symbol="₹")
    assert c.counter_number == 800000
    assert "₹" in c.template_wording
    assert "$" not in c.template_wording


def test_build_counter_offer_usd() -> None:
    c = build_counter_offer(150000, target_role="Eng", currency_symbol="$")
    assert c.counter_number == 150000
    assert "$" in c.template_wording


def test_counter_offer_currency_deterministic_inr() -> None:
    a = build_counter_offer(800000, current_offer=600000, expected_salary=750000, currency_symbol="₹")
    b = build_counter_offer(800000, current_offer=600000, expected_salary=750000, currency_symbol="₹")
    assert a.counter_number == b.counter_number


def test_counter_offer_currency_deterministic_usd() -> None:
    a = build_counter_offer(150000, current_offer=100000, expected_salary=140000, currency_symbol="$")
    b = build_counter_offer(150000, current_offer=100000, expected_salary=140000, currency_symbol="$")
    assert a.counter_number == b.counter_number
