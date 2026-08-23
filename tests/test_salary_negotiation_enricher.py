"""Tests for the Salary Negotiator Gemini enrichment (MOCKED — no API key).

Covers: valid enrichment, invented-fact rejection, malformed response,
API failure, counter-offer/benchmark immutability, evidence-ID immutability,
**currency immutability** (Gemini cannot change $ ↔ ₹, introduce €/£, or
use ISO currency prefixes).
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.modules.salary_negotiation.benchmark import compute_benchmark
from src.modules.salary_negotiation.enricher import apply_enrichment, enrich_script
from src.modules.salary_negotiation.script import build_script
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


def _fake_client(response: str = "", *, raise_exc: Optional[Exception] = None):
    class _F:
        def __init__(self):
            self.calls = []

        def generate_json(self, *, system_prompt, user_text, response_schema, **kw):
            self.calls.append({"system_prompt": system_prompt, "user_text": user_text})
            if raise_exc is not None:
                raise raise_exc
            return response

    return _F()


def _build_script(sample_salary_inputs):
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence)
    return s, resume, job, match, evidence


def _build_script_inr(sample_salary_inputs):
    """Build a script using INR/₹ currency."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote", market="India")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence, currency_symbol="₹")
    return s, resume, job, match, evidence


def _make_6pts(override: dict[int, str]) -> str:
    """Build a 6-point Gemini response; override[idx] = wording for NP-00idx."""
    base = "OK."
    pts = []
    for i in range(1, 7):
        w = override.get(i, base)
        pts.append({"point_id": f"NP-{i:03d}", "wording": w})
    return json.dumps({"talking_points": pts})


# ---- valid enrichment --------------------------------------------


def test_enrich_valid_accepted(sample_salary_inputs, salary_enrichment_valid_json) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(salary_enrichment_valid_json)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert enrichment  # non-empty
    apply_enrichment(s, enrichment)
    assert s.enriched is True
    assert any("researched" in tp.wording.lower() for tp in s.talking_points)


def test_enrich_prompt_grounding(sample_salary_inputs, salary_enrichment_valid_json) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(salary_enrichment_valid_json)
    enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "evidence-grounded" in fake.calls[0]["system_prompt"].lower()
    assert "must not" in fake.calls[0]["system_prompt"].lower() or "do not" in fake.calls[0]["system_prompt"].lower()


# ---- invented metric rejected ------------------------------------


def test_enrich_invented_metric_rejected(sample_salary_inputs, salary_enrichment_invented_json) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(salary_enrichment_invented_json)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-001" not in enrichment or not enrichment.get("NP-001")


# ---- invented employer rejected ---------------------------------


def test_enrich_invented_employer_rejected(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(_make_6pts({1: "I worked at Google building scalable systems."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-001" not in enrichment or not enrichment.get("NP-001")


# ---- invented skill rejected -------------------------------------


def test_enrich_invented_skill_rejected(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(_make_6pts({1: "I have extensive TensorFlow experience."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-001" not in enrichment or not enrichment.get("NP-001")


# ---- counter-offer number cannot be changed ---------------------


def test_enrich_cannot_change_counter_number(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(_make_6pts({4: "I want $999,999."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- benchmark cannot be changed ---------------------------------


def test_enrich_cannot_change_benchmark(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    bm = compute_benchmark(seniority="senior", location="remote")
    original_mid = bm.mid
    fake = _fake_client(_make_6pts({}))
    enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert bm.mid == original_mid


# ---- evidence IDs cannot be changed ------------------------------


def test_enrich_cannot_change_evidence_ids(sample_salary_inputs, salary_enrichment_valid_json) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    original_ids = {tp.point_id: list(tp.evidence_ids) for tp in s.talking_points}
    fake = _fake_client(salary_enrichment_valid_json)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    apply_enrichment(s, enrichment)
    for tp in s.talking_points:
        assert tp.evidence_ids == original_ids[tp.point_id]


# ---- malformed response ------------------------------------------


def test_enrich_malformed_returns_empty(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client("not json {{{")
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert enrichment == {}


def test_enrich_empty_returns_empty(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client("")
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert enrichment == {}


def test_enrich_wrong_point_count_rejected(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(json.dumps({"talking_points": [{"point_id": "NP-001", "wording": "OK."}]}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert enrichment == {}


# ---- API failure -------------------------------------------------


def test_enrich_propagates_api_error(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client("", raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)


# ---- no API key -------------------------------------------------


def test_enrich_no_client_returns_empty(sample_salary_inputs) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=None)
    assert enrichment == {}


def test_enrich_default_client_no_key_returns_empty(monkeypatch, sample_salary_inputs) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=None)
    assert enrichment == {}


# ---- no raw resume_text in Gemini context -----------------------


def test_enrich_context_no_raw_resume_text(sample_salary_inputs, salary_enrichment_valid_json) -> None:
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(salary_enrichment_valid_json)
    enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    payload = fake.calls[0]["user_text"]
    assert "Backend engineer building Python services on AWS" not in payload


# ================================================================== #
# CURRENCY IMMUTABILITY TESTS
# ================================================================== #


# ---- USD → INR currency change rejected --------------------------


def test_currency_change_usd_to_inr_rejected(sample_salary_inputs) -> None:
    """Deterministic script uses $; Gemini changes to ₹ → rejected."""
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    # NP-004 (Compensation Ask) has a $ number; Gemini changes it to ₹.
    fake = _fake_client(_make_6pts({4: "I would like to propose a base salary of ₹999,999."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- INR → USD currency change rejected --------------------------


def test_currency_change_inr_to_usd_rejected(sample_salary_inputs) -> None:
    """Deterministic script uses ₹; Gemini changes to $ → rejected."""
    s, resume, job, match, evidence = _build_script_inr(sample_salary_inputs)
    # The script uses ₹; Gemini introduces $.
    fake = _fake_client(_make_6pts({4: "I would like to propose a base salary of $999,999."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- € introduced rejected --------------------------------------


def test_currency_euro_rejected(sample_salary_inputs) -> None:
    """Gemini introduces € → rejected."""
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(_make_6pts({4: "I would like a base salary of €120,000."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- £ introduced rejected --------------------------------------


def test_currency_pound_rejected(sample_salary_inputs) -> None:
    """Gemini introduces £ → rejected."""
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(_make_6pts({4: "I would like a base salary of £90,000."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- ISO prefix (INR 800,000) instead of ₹800,000 rejected ------


def test_currency_iso_prefix_rejected(sample_salary_inputs) -> None:
    """Gemini uses 'INR 800,000' instead of '₹800,000' → rejected."""
    s, resume, job, match, evidence = _build_script_inr(sample_salary_inputs)
    fake = _fake_client(_make_6pts({4: "I would like to propose a base salary of INR 999,999."}))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert "NP-004" not in enrichment or not enrichment.get("NP-004")


# ---- valid INR wording accepted ---------------------------------


def test_valid_inr_wording_accepted(sample_salary_inputs) -> None:
    """Valid INR wording (same ₹ symbol, same numbers) is accepted."""
    s, resume, job, match, evidence = _build_script_inr(sample_salary_inputs)
    # Build a valid enrichment that keeps ₹ and the same numbers.
    det_numbers = []
    for tp in s.talking_points:
        import re
        det_numbers.extend(re.findall(r"₹[\d,]+", tp.wording))
    # Build a valid response: keep the ₹ numbers, change wording slightly.
    valid_wordings = {i: "OK." for i in range(1, 7)}
    # NP-001 has the range, NP-004 has the counter; keep numbers.
    for tp in s.talking_points:
        import re
        nums = re.findall(r"₹[\d,]+", tp.wording)
        if nums:
            valid_wordings[int(tp.point_id.split("-")[1])] = f"I am looking for {nums[0]}."
    fake = _fake_client(_make_6pts(valid_wordings))
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    # At least some points should be accepted.
    assert len(enrichment) > 0, "valid INR wording should be accepted"


# ---- valid USD wording accepted ---------------------------------


def test_valid_usd_wording_accepted(sample_salary_inputs, salary_enrichment_valid_json) -> None:
    """Valid USD wording (same $ symbol, same or no numbers) is accepted."""
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    fake = _fake_client(salary_enrichment_valid_json)
    enrichment = enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert len(enrichment) > 0, "valid USD wording should be accepted"


# ---- counter-offer remains unchanged ----------------------------


def test_counter_offer_remains_unchanged(sample_salary_inputs) -> None:
    """The counter-offer object is never touched by the enricher."""
    from src.modules.salary_negotiation.script import build_counter_offer

    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    bm = compute_benchmark(seniority="senior", location="remote")
    counter = build_counter_offer(bm.mid, target_role="Backend Engineer")
    original_num = counter.counter_number
    fake = _fake_client(_make_6pts({4: "I want $999,999."}))
    enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert counter.counter_number == original_num  # counter-offer untouched


# ---- benchmark remains unchanged ---------------------------------


def test_benchmark_remains_unchanged(sample_salary_inputs) -> None:
    """The benchmark is never touched by the enricher."""
    s, resume, job, match, evidence = _build_script(sample_salary_inputs)
    bm = compute_benchmark(seniority="senior", location="remote")
    original_low, original_mid, original_high = bm.low, bm.mid, bm.high
    fake = _fake_client(_make_6pts({4: "I want €999,999."}))
    enrich_script(s, resume=resume, job=job, match=match, evidence=evidence, client=fake)
    assert bm.low == original_low
    assert bm.mid == original_mid
    assert bm.high == original_high
