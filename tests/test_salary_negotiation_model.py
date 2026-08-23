"""Tests for the Salary Negotiator data models."""

from __future__ import annotations

import pytest

from src.modules.salary_negotiation.model import (
    CounterOffer,
    NegotiationPoint,
    NegotiationScript,
    SalaryBenchmark,
    SalaryError,
    SalaryParseError,
    ValidationStatus,
)


# ---- ValidationStatus enum -----------------------------------------


def test_validation_status_members() -> None:
    assert {s.value for s in ValidationStatus} == {"pass", "warning", "fail"}


def test_validation_status_from_value() -> None:
    assert ValidationStatus.from_value("FAIL") is ValidationStatus.FAIL
    assert ValidationStatus.from_value(None) is ValidationStatus.PASS
    assert ValidationStatus.from_value("bogus") is ValidationStatus.PASS


# ---- SalaryBenchmark -----------------------------------------------


def test_salary_benchmark_defaults() -> None:
    b = SalaryBenchmark()
    assert b.low == 0.0
    assert b.mid == 0.0
    assert b.high == 0.0
    assert b.currency == "USD"
    assert b.currency_symbol == "$"
    assert b.disclaimer == ""


def test_salary_benchmark_from_dict() -> None:
    b = SalaryBenchmark.from_dict(
        {"role": "Eng", "low": 100, "mid": 120, "high": 140, "currency": "USD", "disclaimer": "x"}
    )
    assert b.role == "Eng"
    assert b.low == 100
    assert b.disclaimer == "x"


def test_salary_benchmark_from_dict_non_dict_raises() -> None:
    with pytest.raises(SalaryParseError):
        SalaryBenchmark.from_dict("nope")  # type: ignore[arg-type]


# ---- NegotiationPoint ---------------------------------------------


def test_negotiation_point_defaults() -> None:
    p = NegotiationPoint()
    assert p.point_id == ""
    assert p.evidence_ids == []
    assert p.validation_status is ValidationStatus.PASS


def test_negotiation_point_from_dict() -> None:
    p = NegotiationPoint.from_dict(
        {"point_id": "NP-001", "title": "Ask", "wording": "w", "evidence_ids": ["EXP-001"], "validation_status": "fail"}
    )
    assert p.point_id == "NP-001"
    assert p.evidence_ids == ["EXP-001"]
    assert p.validation_status is ValidationStatus.FAIL


# ---- NegotiationScript --------------------------------------------


def test_negotiation_script_defaults() -> None:
    s = NegotiationScript()
    assert s.talking_points == []
    assert s.enriched is False


def test_negotiation_script_from_dict() -> None:
    s = NegotiationScript.from_dict(
        {"target_role": "Eng", "talking_points": [{"point_id": "NP-001"}], "leverage_points": ["x"], "enriched": True}
    )
    assert s.target_role == "Eng"
    assert len(s.talking_points) == 1
    assert s.enriched is True


def test_negotiation_script_to_dict_round_trip() -> None:
    s = NegotiationScript(
        target_role="Eng",
        talking_points=[NegotiationPoint(point_id="NP-001", title="x")],
        enriched=True,
    )
    d = s.to_dict()
    again = NegotiationScript.from_dict(d)
    assert again.target_role == "Eng"
    assert again.enriched is True
    assert again.talking_points[0].point_id == "NP-001"


# ---- CounterOffer -------------------------------------------------


def test_counter_offer_defaults() -> None:
    c = CounterOffer()
    assert c.counter_number == 0.0
    assert c.current_offer is None
    assert c.expected_salary is None


def test_counter_offer_from_dict() -> None:
    c = CounterOffer.from_dict(
        {"counter_number": 150000, "benchmark_mid": 140000, "current_offer": 120000, "rationale": "r", "template_wording": "t"}
    )
    assert c.counter_number == 150000
    assert c.current_offer == 120000
    assert c.rationale == "r"


def test_error_hierarchy() -> None:
    assert issubclass(SalaryParseError, SalaryError)
