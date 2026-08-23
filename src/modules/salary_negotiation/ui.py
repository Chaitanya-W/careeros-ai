"""Salary Negotiator UI workflow.

Flow: prerequisite guard → config (role/location/seniority/salary) →
benchmark card → leverage points → negotiation script → counter-offer.

Prerequisites: ResumeAnalysis required (for leverage). JobAnalysis is
optional (role can be user-entered).

State keys: salary_benchmark / salary_script / salary_counter_offer.
All existing state is preserved.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.application_copilot.evidence import extract_evidence
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.modules.salary_negotiation.benchmark import (
    compute_benchmark,
    infer_seniority,
)
from src.modules.salary_negotiation.enricher import apply_enrichment, enrich_script
from src.modules.salary_negotiation.model import (
    CounterOffer,
    NegotiationScript,
    SalaryBenchmark,
)
from src.modules.salary_negotiation.script import build_counter_offer, build_script
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while preparing your negotiation. Please try again."
)


def render() -> None:
    """Render the Salary Negotiator workflow."""
    section_header(
        "Salary Negotiator",
        "Benchmark compensation and rehearse data-backed negotiation scripts.",
    )

    resume = st.session_state.get("resume_analysis")
    if not isinstance(resume, ResumeAnalysis):
        _render_no_resume_state()
        return

    job = st.session_state.get("job_analysis")
    match = st.session_state.get("match_analysis")

    _render_config(resume, job, match)


# --------------------------------------------------------------------------- #
# Prerequisite
# --------------------------------------------------------------------------- #


def _render_no_resume_state() -> None:
    empty_state_panel(
        title="Analyze your resume first.",
        hint="The Salary Negotiator grounds negotiation leverage in your "
        "analyzed resume. Head to Resume Intelligence to upload and "
        "analyze your resume.",
    )
    if st.button(
        "Go to Resume Intelligence",
        type="primary",
        key="salary_goto_resume",
    ):
        navigate_to("resume_intelligence")


# --------------------------------------------------------------------------- #
# Configuration + compute
# --------------------------------------------------------------------------- #


def _render_config(resume, job, match) -> None:
    section_header("Configuration", "Enter your target role, location, and salary details.")

    default_role = ""
    default_seniority = "mid"
    if isinstance(job, JobAnalysis):
        default_role = job.job_title or ""
        default_seniority = infer_seniority(job.experience_requirements or "")

    market = st.selectbox(
        "Market",
        ["United States", "India"],
        index=0,
        key="salary_market",
        help="Select your market for currency and base salary benchmarking.",
    )
    from src.modules.salary_negotiation.benchmark import get_currency_symbol
    symbol = get_currency_symbol(market)

    col_left, col_right = st.columns(2)
    with col_left:
        target_role = st.text_input(
            "Target role", value=default_role, key="salary_target_role",
            help="The role you're negotiating for.",
        )
        location = st.text_input(
            "Location", value="", key="salary_location",
            placeholder="e.g. San Francisco, Remote, Bangalore",
        )
    with col_right:
        seniority = st.selectbox(
            "Experience / seniority",
            ["entry", "junior", "mid", "senior", "lead", "staff", "principal"],
            index=["entry", "junior", "mid", "senior", "lead", "staff", "principal"].index(default_seniority),
            key="salary_seniority",
        )
        current_offer = st.number_input(
            f"Current offer ({symbol}, optional)", min_value=0, value=0, step=5000, key="salary_current_offer",
        )
        expected_salary = st.number_input(
            f"Expected salary ({symbol}, optional)", min_value=0, value=0, step=5000, key="salary_expected",
        )

    if st.button("Compute benchmark & script", type="primary", key="salary_compute"):
        _compute(
            resume, job, match, target_role, location, seniority, market,
            current_offer or None, expected_salary or None,
        )

    # Show cached results if available.
    benchmark = st.session_state.get("salary_benchmark")
    script = st.session_state.get("salary_script")
    counter = st.session_state.get("salary_counter_offer")

    if benchmark is not None and script is not None and counter is not None:
        _render_benchmark(benchmark)
        _render_leverage(script)
        _render_script(script, resume, job, match)
        _render_counter_offer(counter)


def _compute(resume, job, match, target_role, location, seniority, market, current_offer, expected_salary):
    with st.spinner("Computing benchmark & script…"):
        try:
            benchmark = compute_benchmark(
                role=target_role, location=location, seniority=seniority,
                market=market,
            )
            evidence = extract_evidence(resume)
            script = build_script(
                resume, job, match, benchmark.mid,
                target_role=target_role,
                current_offer=current_offer,
                expected_salary=expected_salary,
                evidence=evidence,
                currency_symbol=benchmark.currency_symbol,
            )
            counter = build_counter_offer(
                benchmark.mid,
                current_offer=current_offer,
                expected_salary=expected_salary,
                target_role=target_role,
                currency_symbol=benchmark.currency_symbol,
            )
        except Exception:
            _logger.exception("Unexpected error during salary negotiation")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    st.session_state["salary_benchmark"] = benchmark
    st.session_state["salary_script"] = script
    st.session_state["salary_counter_offer"] = counter
    st.success("Benchmark & negotiation script ready.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_benchmark(benchmark: SalaryBenchmark) -> None:
    section_header("Benchmark", "Heuristic estimate — not from a paid salary API.")
    sym = benchmark.currency_symbol or "$"
    col_low, col_mid, col_high = st.columns(3)
    with col_low:
        with st.container(border=True):
            st.metric("Low", f"{sym}{benchmark.low:,.0f}")
    with col_mid:
        with st.container(border=True):
            st.metric("Mid", f"{sym}{benchmark.mid:,.0f}")
    with col_high:
        with st.container(border=True):
            st.metric("High", f"{sym}{benchmark.high:,.0f}")
    st.caption(f"Role: {benchmark.role or '—'} · Location: {benchmark.location or '—'} · Seniority: {benchmark.seniority or '—'}")
    st.caption(f"Methodology: {benchmark.methodology}")
    st.warning(benchmark.disclaimer)


def _render_leverage(script: NegotiationScript) -> None:
    if not script.leverage_points:
        return
    section_header("Leverage points", "Grounded in your resume evidence.")
    for lp in script.leverage_points:
        st.markdown(f"- {lp}")


def _render_script(script: NegotiationScript, resume, job, match) -> None:
    section_header("Negotiation script", "Talking points grounded in your evidence.")
    if script.warnings:
        for w in script.warnings:
            st.warning(w)

    for tp in script.talking_points:
        with st.container(border=True):
            st.markdown(f"**{tp.point_id} — {tp.title}**")
            st.write(tp.wording)
            if tp.evidence_ids:
                st.caption(f"Evidence: {', '.join(tp.evidence_ids)}")

    # AI enrichment (optional).
    key_available = _gemini_key_available()
    if not key_available:
        st.info(
            "AI script refinement requires Gemini configuration. "
            "The deterministic script above is fully usable without a key."
        )
    else:
        if script.enriched:
            st.caption("AI enrichment: applied.")
            if st.button("Regenerate AI refinement", key="salary_regen"):
                _run_enrichment(script, resume, job, match)
        else:
            if st.button("Refine with AI", type="primary", key="salary_enrich"):
                _run_enrichment(script, resume, job, match)


def _run_enrichment(script, resume, job, match):
    from src.modules.application_copilot.evidence import extract_evidence

    with st.spinner("Refining script wording…"):
        try:
            from src.services.gemini import GeminiClient

            evidence = extract_evidence(resume)
            enrichment = enrich_script(
                script, resume=resume, job=job, match=match,
                evidence=evidence, client=GeminiClient(),
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error during script enrichment")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    apply_enrichment(script, enrichment)
    st.session_state["salary_script"] = script
    st.success("AI refinement applied.")
    st.rerun()


def _render_counter_offer(counter: CounterOffer) -> None:
    section_header("Counter-offer", "Deterministic number — not AI-invented.")
    sym = "$"  # Default; overridden below if benchmark has a symbol.
    bm = st.session_state.get("salary_benchmark")
    if bm is not None and hasattr(bm, "currency_symbol"):
        sym = bm.currency_symbol or "$"
    col_num, col_mid = st.columns(2)
    with col_num:
        with st.container(border=True):
            st.metric("Counter number", f"{sym}{counter.counter_number:,.0f}")
    with col_mid:
        with st.container(border=True):
            st.metric("Benchmark mid", f"{sym}{counter.benchmark_mid:,.0f}")
    if counter.current_offer:
        st.caption(f"Current offer: {sym}{counter.current_offer:,.0f}")
    if counter.expected_salary:
        st.caption(f"Expected salary: {sym}{counter.expected_salary:,.0f}")
    st.caption("Rationale")
    st.write(counter.rationale)
    st.caption("Ready-to-use template")
    with st.container(border=True):
        st.write(counter.template_wording)


def _gemini_key_available() -> bool:
    try:
        from src.services.gemini import get_api_key
        get_api_key()
        return True
    except GeminiConfigError:
        return False
    except Exception:
        _logger.exception("Unexpected error checking Gemini key")
        return False
