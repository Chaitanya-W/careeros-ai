"""Deterministic salary benchmark — NO Gemini, NO paid API, NO internet.

This is a deterministic heuristic benchmark for demonstration/planning
purposes. It is not live market compensation data and does not use a paid
salary API.

Supported markets/currencies:
    - "United States" → USD ($), base salary $80,000
    - "India"         → INR (₹), base salary ₹800,000

Uses a documented heuristic formula:

    estimated_mid = market_base_salary × seniority_multiplier × location_multiplier

Then:
    low  = mid × LOW_FACTOR  (0.85)
    mid  = estimated_mid (clamped to market-specific bounds)
    high = mid × HIGH_FACTOR  (1.15)

The calculation is:
- deterministic (same inputs → same output)
- repeatable
- testable
- clamped to sensible bounds
- independent of Gemini
- Gemini MUST NOT influence low/mid/high/currency/counter-offer number
"""

from __future__ import annotations

from typing import Final

from src.modules.salary_negotiation.model import SalaryBenchmark

# --- Market configuration (deterministic, local, no external API) ---------- #
# Each market has its own base salary, currency, symbol, and clamp bounds.
# Values are heuristic estimates — NOT real market data.

MARKET_CONFIG: Final[dict[str, dict]] = {
    "United States": {
        "currency": "USD",
        "symbol": "$",
        "base_salary": 80_000.0,
        "min_salary": 20_000.0,
        "max_salary": 500_000.0,
    },
    "India": {
        "currency": "INR",
        "symbol": "₹",
        "base_salary": 800_000.0,
        "min_salary": 100_000.0,
        "max_salary": 5_000_000.0,
    },
}

DEFAULT_MARKET: Final[str] = "United States"

# --- Seniority multipliers (applied to market base salary) ---------------- #

SENIORITY_MULTIPLIERS: Final[dict[str, float]] = {
    "entry": 1.0,
    "junior": 1.0,
    "mid": 1.35,
    "intermediate": 1.35,
    "senior": 1.75,
    "lead": 2.10,
    "staff": 2.40,
    "principal": 2.75,
}

# --- Location multipliers (cost-of-living adjustment within a market) ------ #

LOCATION_MULTIPLIERS: Final[dict[str, float]] = {
    "san francisco": 1.60,
    "new york": 1.50,
    "seattle": 1.40,
    "boston": 1.30,
    "los angeles": 1.35,
    "washington dc": 1.30,
    "chicago": 1.10,
    "austin": 1.15,
    "denver": 1.10,
    "atlanta": 1.05,
    "remote": 1.00,
    "global": 0.80,
    "bangalore": 1.30,
    "mumbai": 1.25,
    "delhi": 1.20,
    "hyderabad": 1.15,
    "pune": 1.10,
    "chennai": 1.10,
}

DEFAULT_SENIORITY: Final[str] = "mid"
DEFAULT_LOCATION: Final[str] = "remote"

# Range factors (low/mid/high spread).
LOW_FACTOR: Final[float] = 0.85
HIGH_FACTOR: Final[float] = 1.15

DISCLAIMER: Final[str] = (
    "Heuristic estimate based on role, seniority, location, and market "
    "multipliers. Not from a paid salary API. Verify against real market data."
)


def compute_benchmark(
    role: str = "",
    location: str = "",
    seniority: str = "",
    market: str = "",
) -> SalaryBenchmark:
    """Compute a deterministic salary benchmark.

    Args:
        role: target role title (informational, not used in the formula).
        location: location string (matched case-insensitively to the
            location multiplier table; unknown → DEFAULT_LOCATION).
        seniority: seniority string (matched case-insensitively to the
            seniority multiplier table; unknown → DEFAULT_SENIORITY).
        market: market key ("United States" or "India");
            unknown → DEFAULT_MARKET.

    Returns:
        A :class:`SalaryBenchmark` with low/mid/high, currency, symbol,
        and the disclaimer.
    """
    cfg = _get_market_config(market)
    base_salary = cfg["base_salary"]
    currency = cfg["currency"]
    symbol = cfg["symbol"]
    min_sal = cfg["min_salary"]
    max_sal = cfg["max_salary"]

    sen_mult = _lookup_multiplier(seniority, SENIORITY_MULTIPLIERS, DEFAULT_SENIORITY)
    loc_mult = _lookup_multiplier(location, LOCATION_MULTIPLIERS, DEFAULT_LOCATION)

    mid = base_salary * sen_mult * loc_mult
    mid = _clamp(mid, min_sal, max_sal)

    low = _clamp(mid * LOW_FACTOR, min_sal, max_sal)
    high = _clamp(mid * HIGH_FACTOR, min_sal, max_sal)

    if low > mid:
        low = mid
    if high < mid:
        high = mid

    methodology = (
        f"base={symbol}{base_salary:,.0f} ({currency}) "
        f"× seniority={sen_mult} ({seniority or DEFAULT_SENIORITY})"
        f" × location={loc_mult} ({location or DEFAULT_LOCATION})"
        f" × market={market or DEFAULT_MARKET}"
        f" → mid={symbol}{mid:,.0f}; "
        f"low={LOW_FACTOR}×mid; high={HIGH_FACTOR}×mid"
    )

    return SalaryBenchmark(
        role=role,
        location=location,
        seniority=seniority,
        low=round(low),
        mid=round(mid),
        high=round(high),
        currency=currency,
        currency_symbol=symbol,
        methodology=methodology,
        disclaimer=DISCLAIMER,
    )


def get_currency_symbol(market: str = "") -> str:
    """Return the currency symbol for a market (default: $)."""
    cfg = _get_market_config(market)
    return cfg["symbol"]


def get_currency_code(market: str = "") -> str:
    """Return the ISO currency code for a market (default: USD)."""
    cfg = _get_market_config(market)
    return cfg["currency"]


def infer_seniority(experience_requirements: str = "") -> str:
    """Infer a seniority level from a job's experience_requirements string."""
    text = (experience_requirements or "").lower()
    if any(w in text for w in ("senior", "lead", "principal", "staff", "architect")):
        return "senior"
    if any(w in text for w in ("junior", "entry", "intern", "graduate")):
        return "entry"
    if any(w in text for w in ("5+", "7+", "10+", "8+")):
        return "senior"
    if any(w in text for w in ("3+", "2+", "1+")):
        return "mid"
    return DEFAULT_SENIORITY


def compute_counter_number(
    benchmark_mid: float,
    current_offer: float | None = None,
    expected_salary: float | None = None,
) -> float:
    """Deterministically compute the counter-offer number.

    If the candidate has an expected salary, use the higher of (benchmark
    mid, expected salary) — but never below the benchmark mid. If there's a
    current offer, counter slightly above it (toward the benchmark high).

    The number is ALWAYS deterministic — Gemini cannot change it.
    """
    if benchmark_mid <= 0:
        return 0.0

    if expected_salary is not None and expected_salary > 0:
        return round(max(expected_salary, benchmark_mid))

    if current_offer is not None and current_offer > 0:
        return round((current_offer + benchmark_mid * HIGH_FACTOR) / 2)

    return round(benchmark_mid)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _get_market_config(market: str) -> dict:
    """Return the market config dict for ``market`` (default: United States)."""
    m = (market or "").strip()
    if m in MARKET_CONFIG:
        return MARKET_CONFIG[m]
    # Case-insensitive match.
    m_lower = m.lower()
    for key, cfg in MARKET_CONFIG.items():
        if key.lower() == m_lower:
            return cfg
    return MARKET_CONFIG[DEFAULT_MARKET]


def _lookup_multiplier(
    key: str, table: dict[str, float], default_key: str
) -> float:
    """Case-insensitive lookup in a multiplier table."""
    k = (key or "").strip().lower()
    if k in table:
        return table[k]
    for table_key, mult in table.items():
        if table_key in k or k in table_key:
            return mult
    return table[default_key]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
