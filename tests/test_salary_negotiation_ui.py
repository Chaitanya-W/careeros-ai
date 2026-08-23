"""Tests for the Salary Negotiator UI (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no enrich button click).
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.salary_negotiation.benchmark import compute_benchmark
from src.modules.salary_negotiation.model import CounterOffer, NegotiationScript, SalaryBenchmark
from src.modules.salary_negotiation.script import build_counter_offer, build_script
from src.modules.application_copilot.evidence import extract_evidence


# ---- UI empty state (no resume) -----------------------------------


def test_salary_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.run()

    assert not at.exception
    assert any("analyze your resume first" in m.value.lower() for m in at.markdown)
    assert any("Resume Intelligence" in b.label for b in at.button)


# ---- UI config with resume ---------------------------------------


def test_salary_renders_config_with_resume(app_path: str, sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception
    assert any("Compute" in b.label for b in at.button), "compute button missing"


# ---- UI result rendering ----------------------------------------


def test_salary_renders_results(app_path: str, sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence)
    c = build_counter_offer(bm.mid, target_role="Backend Engineer")
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["salary_benchmark"] = bm
    at.session_state["salary_script"] = s
    at.session_state["salary_counter_offer"] = c
    at.run()

    assert not at.exception
    # Benchmark section + numbers.
    assert any("Benchmark" in sub.value for sub in at.subheader)
    assert any(f"{bm.mid:,.0f}" in m.value for m in at.markdown) or any(
        str(int(bm.mid)) in str(v) for v in [m.value for m in at.markdown] + [str(s.value) for s in at.metric]
    )
    # Negotiation script section.
    assert any("Negotiation script" in sub.value for sub in at.subheader)
    # Counter-offer section.
    assert any("Counter" in sub.value for sub in at.subheader)


def test_salary_renders_india_inr(app_path: str, sample_salary_inputs) -> None:
    """India market → ₹ symbol in benchmark and no hardcoded (USD) labels."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote", market="India")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence, currency_symbol="₹")
    c = build_counter_offer(bm.mid, target_role="Backend Engineer", currency_symbol="₹")
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["salary_benchmark"] = bm
    at.session_state["salary_script"] = s
    at.session_state["salary_counter_offer"] = c
    at.run()

    assert not at.exception
    # Benchmark renders ₹ symbol.
    all_md = " ".join(m.value for m in at.markdown) + " ".join(str(s.value) for s in at.metric)
    assert "₹" in all_md, "₹ symbol not rendered for India market"
    # Disclaimer remains visible.
    assert any("paid salary API" in w.value for w in at.warning)


def test_salary_renders_usd(app_path: str, sample_salary_inputs) -> None:
    """US market → $ symbol in benchmark."""
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior", location="remote", market="United States")
    s = build_script(resume, job, match, bm.mid, target_role="Backend Engineer", evidence=evidence, currency_symbol="$")
    c = build_counter_offer(bm.mid, target_role="Backend Engineer", currency_symbol="$")
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["salary_benchmark"] = bm
    at.session_state["salary_script"] = s
    at.session_state["salary_counter_offer"] = c
    at.run()

    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown) + " ".join(str(s.value) for s in at.metric)
    assert "$" in all_md, "$ symbol not rendered for US market"


def test_salary_ui_has_market_selector(app_path: str, sample_salary_inputs) -> None:
    """The config UI should have a Market selectbox."""
    resume, job, match, evidence = sample_salary_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()
    assert not at.exception
    # Market selector present.
    assert any("Market" in s.label for s in at.selectbox), "market selector missing"


def test_salary_ui_no_hardcoded_usd_labels(app_path: str, sample_salary_inputs) -> None:
    """Salary input labels should NOT contain hardcoded '(USD)'."""
    resume, job, match, evidence = sample_salary_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()
    assert not at.exception
    labels = " ".join(n.label for n in at.number_input)
    assert "(USD)" not in labels, "hardcoded (USD) label found"


# ---- UI no-key message ------------------------------------------


def test_salary_no_key_message(app_path: str, sample_salary_inputs, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, evidence = sample_salary_inputs
    bm = compute_benchmark(seniority="senior")
    s = build_script(resume, job, match, bm.mid, evidence=evidence)
    c = build_counter_offer(bm.mid)
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["salary_benchmark"] = bm
    at.session_state["salary_script"] = s
    at.session_state["salary_counter_offer"] = c
    at.run()
    assert not at.exception
    assert any("requires Gemini configuration" in i.value for i in at.info)


# ---- session-state preservation ---------------------------------


def test_salary_does_not_overwrite_existing_state(app_path: str, sample_salary_inputs) -> None:
    resume, job, match, evidence = sample_salary_inputs
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.session_state["resume_analysis"] = resume
    at.session_state["resume_text"] = "PRESERVED"
    at.run()
    assert not at.exception
    assert at.session_state["resume_analysis"] is resume
    assert at.session_state["resume_text"] == "PRESERVED"


def test_salary_initializes_state(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "salary_negotiation"
    at.run()
    assert not at.exception
    for key in ("salary_benchmark", "salary_script", "salary_counter_offer"):
        assert key in at.session_state
