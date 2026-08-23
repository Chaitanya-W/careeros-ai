"""Tests for the job-description analyzer (Gemini is MOCKED — no API key)."""

from __future__ import annotations

import json

import pytest

from src.modules.jd_analysis.analyzer import (
    EmptyJobDescriptionError,
    MAX_JD_LENGTH,
    MIN_JD_LENGTH,
    analyze_job_description,
)
from src.modules.jd_analysis.model import JobAnalysis, JobAnalysisParseError
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError

# A job description long enough to pass the MIN_JD_LENGTH (50) check so the
# tests reach the (mocked) client instead of being rejected up front.
_VALID_JD: str = (
    "Senior Backend Engineer at Acme Corp requiring Python, SQL, AWS, "
    "Docker, and distributed systems experience."
)


# ---- (14) mocked Gemini job analysis (happy path) ----------------------


def test_analyze_job_description_success(
    sample_job_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_job_analysis_json)
    jd = "Senior Backend Engineer at Acme Corp. Requires Python, SQL, AWS."
    result = analyze_job_description(jd, client=fake)
    assert isinstance(result, JobAnalysis)
    assert result.job_title == "Senior Backend Engineer"
    assert "Python" in result.required_skills
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["user_text"] == jd.strip()
    assert call["response_schema"] is not None
    assert "recruiter" in call["system_prompt"].lower()


def test_analyze_job_description_strips_whitespace(
    sample_job_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_job_analysis_json)
    # Surround a valid JD with whitespace; the service must strip before
    # sending to the client.
    padded = "   \n" + _VALID_JD + "\n   "
    analyze_job_description(padded, client=fake)
    assert fake.calls[0]["user_text"] == _VALID_JD


def test_analyze_job_description_passes_schema(
    sample_job_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_job_analysis_json)
    analyze_job_description(_VALID_JD, client=fake)
    schema = fake.calls[0]["response_schema"]
    assert schema["type"] == "object"
    assert "required_skills" in schema["required"]


# ---- (3) empty job description / (4) invalid job description -----------
# NOTE: these input-validation failures happen BEFORE any Gemini call.


def test_analyze_job_description_none_raises_empty() -> None:
    with pytest.raises(EmptyJobDescriptionError):
        analyze_job_description(None)  # type: ignore[arg-type]


def test_analyze_job_description_whitespace_only_raises_empty() -> None:
    with pytest.raises(EmptyJobDescriptionError):
        analyze_job_description("   \n\t  ")


def test_analyze_job_description_too_short_raises_empty(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="{}")
    # Below MIN_JD_LENGTH -> rejected before calling the client.
    with pytest.raises(EmptyJobDescriptionError):
        analyze_job_description("short jd", client=fake)
    assert fake.calls == []  # Gemini never invoked


def test_analyze_job_description_too_long_raises_empty(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="{}")
    huge = "x" * (MAX_JD_LENGTH + 1)
    with pytest.raises(EmptyJobDescriptionError):
        analyze_job_description(huge, client=fake)
    assert fake.calls == []


def test_min_max_bounds_are_sane() -> None:
    assert MIN_JD_LENGTH >= 10
    assert MAX_JD_LENGTH > MIN_JD_LENGTH


# ---- (15) invalid Gemini JSON ------------------------------------------


def test_analyze_job_description_invalid_json_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="not json {{{")
    with pytest.raises(JobAnalysisParseError):
        analyze_job_description(
            _VALID_JD, client=fake
        )


def test_analyze_job_description_empty_response_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="")
    with pytest.raises(JobAnalysisParseError):
        analyze_job_description(
            _VALID_JD, client=fake
        )


def test_analyze_job_description_non_object_json_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=json.dumps(["not", "object"]))
    with pytest.raises(JobAnalysisParseError):
        analyze_job_description(
            _VALID_JD, client=fake
        )


# ---- API / config errors propagate as GeminiError ----------------------


def test_analyze_job_description_propagates_api_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        analyze_job_description(
            _VALID_JD, client=fake
        )


def test_analyze_job_description_propagates_config_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiConfigError("no key"))
    with pytest.raises(GeminiError):
        analyze_job_description(
            _VALID_JD, client=fake
        )


# ---- (13) session-state-style storage (result is reusable) -------------


def test_job_analysis_result_can_be_stored_and_reused(
    sample_job_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_job_analysis_json)
    result = analyze_job_description(
        _VALID_JD, client=fake
    )
    stored = result.to_dict()
    restored = JobAnalysis.from_dict(stored)
    assert restored.company == result.company
    assert restored.required_skills == result.required_skills


# ---- no real API key required: default client raises GeminiConfigError -


def test_default_client_without_key_raises_config(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError):
        analyze_job_description(_VALID_JD)
