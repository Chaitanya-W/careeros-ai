"""Tests for the deterministic salary benchmark (multi-market, multi-currency)."""

from __future__ import annotations

import pytest

from src.modules.salary_negotiation.benchmark import (
    DEFAULT_LOCATION,
    DEFAULT_MARKET,
    DEFAULT_SENIORITY,
    DISCLAIMER,
    HIGH_FACTOR,
    LOW_FACTOR,
    MARKET_CONFIG,
    compute_benchmark,
    compute_counter_number,
    get_currency_code,
    get_currency_symbol,
    infer_seniority,
)


# ---- market configuration -----------------------------------------


def test_market_config_has_us_and_india() -> None:
    assert "United States" in MARKET_CONFIG
    assert "India" in MARKET_CONFIG


def test_us_market_is_usd() -> None:
    assert MARKET_CONFIG["United States"]["currency"] == "USD"
    assert MARKET_CONFIG["United States"]["symbol"] == "$"


def test_india_market_is_inr() -> None:
    assert MARKET_CONFIG["India"]["currency"] == "INR"
    assert MARKET_CONFIG["India"]["symbol"] == "₹"


def test_get_currency_symbol_us() -> None:
    assert get_currency_symbol("United States") == "$"


def test_get_currency_symbol_india() -> None:
    assert get_currency_symbol("India") == "₹"


def test_get_currency_symbol_default() -> None:
    assert get_currency_symbol("") == "$"
    assert get_currency_symbol("Unknown") == "$"


def test_get_currency_code_us() -> None:
    assert get_currency_code("United States") == "USD"


def test_get_currency_code_india() -> None:
    assert get_currency_code("India") == "INR"


# ---- benchmark determinism ----------------------------------------


def test_benchmark_deterministic_us() -> None:
    a = compute_benchmark(role="Eng", location="San Francisco", seniority="senior", market="United States")
    b = compute_benchmark(role="Eng", location="San Francisco", seniority="senior", market="United States")
    assert a.low == b.low
    assert a.mid == b.mid
    assert a.high == b.high


def test_benchmark_deterministic_india() -> None:
    a = compute_benchmark(role="Eng", location="Bangalore", seniority="senior", market="India")
    b = compute_benchmark(role="Eng", location="Bangalore", seniority="senior", market="India")
    assert a.low == b.low
    assert a.mid == b.mid
    assert a.high == b.high


# ---- low <= mid <= high -------------------------------------------


def test_low_le_mid_le_high_us() -> None:
    bm = compute_benchmark(role="Eng", location="Remote", seniority="mid", market="United States")
    assert bm.low <= bm.mid <= bm.high


def test_low_le_mid_le_high_inr() -> None:
    bm = compute_benchmark(role="Eng", location="Remote", seniority="mid", market="India")
    assert bm.low <= bm.mid <= bm.high


def test_low_le_mid_le_high_all_markets_and_seniorities() -> None:
    for market in ["United States", "India"]:
        for sen in ["entry", "junior", "mid", "senior", "lead", "staff", "principal"]:
            bm = compute_benchmark(seniority=sen, market=market)
            assert bm.low <= bm.mid <= bm.high, f"failed for {market}/{sen}"


# ---- currency stored in benchmark --------------------------------


def test_benchmark_currency_us() -> None:
    bm = compute_benchmark(market="United States")
    assert bm.currency == "USD"
    assert bm.currency_symbol == "$"


def test_benchmark_currency_india() -> None:
    bm = compute_benchmark(market="India")
    assert bm.currency == "INR"
    assert bm.currency_symbol == "₹"


def test_benchmark_currency_default() -> None:
    bm = compute_benchmark()
    assert bm.currency == "USD"
    assert bm.currency_symbol == "$"


# ---- seniority multiplier ----------------------------------------


def test_seniority_affects_mid_us() -> None:
    entry = compute_benchmark(seniority="entry", market="United States")
    senior = compute_benchmark(seniority="senior", market="United States")
    assert senior.mid > entry.mid


def test_seniority_affects_mid_india() -> None:
    entry = compute_benchmark(seniority="entry", market="India")
    senior = compute_benchmark(seniority="senior", market="India")
    assert senior.mid > entry.mid


def test_seniority_unknown_uses_default() -> None:
    bm = compute_benchmark(seniority="bogus")
    default_bm = compute_benchmark(seniority="mid")
    assert bm.mid == default_bm.mid


# ---- location multiplier -----------------------------------------


def test_location_affects_mid_us() -> None:
    remote = compute_benchmark(location="remote", market="United States")
    sf = compute_benchmark(location="san francisco", market="United States")
    assert sf.mid > remote.mid


def test_location_affects_mid_india() -> None:
    remote = compute_benchmark(location="remote", market="India")
    blr = compute_benchmark(location="bangalore", market="India")
    assert blr.mid > remote.mid


def test_location_unknown_uses_default() -> None:
    bm = compute_benchmark(location="Nowhere")
    default_bm = compute_benchmark(location="remote")
    assert bm.mid == default_bm.mid


def test_location_partial_match() -> None:
    bm = compute_benchmark(location="San Francisco Bay Area")
    sf = compute_benchmark(location="san francisco")
    assert bm.mid == sf.mid


# ---- clamping -----------------------------------------------------


def test_clamped_to_min_us() -> None:
    bm = compute_benchmark(seniority="entry", location="global", market="United States")
    assert bm.mid >= MARKET_CONFIG["United States"]["min_salary"]


def test_clamped_to_max_us() -> None:
    bm = compute_benchmark(seniority="principal", location="san francisco", market="United States")
    assert bm.mid <= MARKET_CONFIG["United States"]["max_salary"]


def test_clamped_to_min_india() -> None:
    bm = compute_benchmark(seniority="entry", location="global", market="India")
    assert bm.mid >= MARKET_CONFIG["India"]["min_salary"]


def test_clamped_to_max_india() -> None:
    bm = compute_benchmark(seniority="principal", location="bangalore", market="India")
    assert bm.mid <= MARKET_CONFIG["India"]["max_salary"]


# ---- disclaimer ---------------------------------------------------


def test_disclaimer_present() -> None:
    bm = compute_benchmark()
    assert bm.disclaimer == DISCLAIMER
    assert "paid salary API" in bm.disclaimer


def test_methodology_present() -> None:
    bm = compute_benchmark(role="Eng", location="Remote", seniority="mid", market="India")
    assert bm.methodology
    assert "base=" in bm.methodology
    assert "INR" in bm.methodology or "₹" in bm.methodology


# ---- infer_seniority ---------------------------------------------


def test_infer_seniority_senior() -> None:
    assert infer_seniority("Senior engineer, 5+ years") == "senior"


def test_infer_seniority_entry() -> None:
    assert infer_seniority("Junior, entry-level") == "entry"


def test_infer_seniority_mid() -> None:
    assert infer_seniority("3+ years experience") == "mid"


def test_infer_seniority_empty() -> None:
    assert infer_seniority("") == "mid"


# ---- counter-offer number ----------------------------------------


def test_counter_number_default() -> None:
    assert compute_counter_number(100000) == 100000


def test_counter_number_with_expected() -> None:
    assert compute_counter_number(100000, expected_salary=120000) == 120000


def test_counter_number_with_current_offer() -> None:
    num = compute_counter_number(100000, current_offer=80000)
    assert 80000 < num < 100000 * HIGH_FACTOR


def test_counter_number_zero_benchmark() -> None:
    assert compute_counter_number(0) == 0


def test_counter_number_deterministic() -> None:
    a = compute_counter_number(150000, current_offer=100000, expected_salary=140000)
    b = compute_counter_number(150000, current_offer=100000, expected_salary=140000)
    assert a == b


# ---- India vs US base salaries differ ----------------------------


def test_india_base_higher_than_us_in_absolute_terms() -> None:
    """India base salary (INR) is numerically larger than US (USD) because
    INR values are in rupees — this is expected and documented."""
    us_bm = compute_benchmark(seniority="entry", location="remote", market="United States")
    in_bm = compute_benchmark(seniority="entry", location="remote", market="India")
    assert in_bm.mid > us_bm.mid  # INR numbers are larger
    assert in_bm.currency == "INR"
    assert us_bm.currency == "USD"
