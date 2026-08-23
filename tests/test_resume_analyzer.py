"""Tests for the resume analyzer service (Gemini is MOCKED — no API key)."""

from __future__ import annotations

import pytest

from src.modules.resume_intelligence.analyzer import (
    EmptyResumeError,
    analyze_resume,
)
from src.modules.resume_intelligence.model import (
    ResumeAnalysis,
    ResumeAnalysisError,
    ResumeAnalysisParseError,
)
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


# ---- (9) Gemini response parsing (happy path) ----------------------------


def test_analyze_resume_success(
    sample_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_analysis_json)
    result = analyze_resume("Some real resume text content here.", client=fake)
    assert isinstance(result, ResumeAnalysis)
    assert result.overall_score == 78
    assert "Python" in result.skills
    assert len(result.experience) == 1
    assert result.experience[0].company == "Acme Corp"
    # The fake client must have been called exactly once with the prompt.
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["user_text"] == "Some real resume text content here."
    assert call["response_schema"] is not None
    assert "expert career coach" in call["system_prompt"].lower()


def test_analyze_resume_strips_whitespace_before_analysis(
    sample_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_analysis_json)
    # Input is long enough to pass the minimum-length check; strip() must
    # remove leading/trailing whitespace before the text reaches Gemini.
    analyze_resume("   \n a sufficiently long resume body text \n", client=fake)
    assert fake.calls[0]["user_text"] == "a sufficiently long resume body text"


def test_analyze_resume_passes_schema_to_client(
    sample_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_analysis_json)
    analyze_resume("real resume text body", client=fake)
    schema = fake.calls[0]["response_schema"]
    assert schema["type"] == "object"
    assert "overall_score" in schema["required"]


# ---- empty / too-short resume text ---------------------------------------


def test_analyze_resume_none_text_raises_empty() -> None:
    with pytest.raises(EmptyResumeError):
        analyze_resume(None)  # type: ignore[arg-type]


def test_analyze_resume_empty_text_raises_empty() -> None:
    with pytest.raises(EmptyResumeError):
        analyze_resume("   ")


def test_analyze_resume_short_text_raises_empty() -> None:
    with pytest.raises(EmptyResumeError):
        analyze_resume("too short")


# ---- (10) invalid Gemini JSON --------------------------------------------


def test_analyze_resume_invalid_json_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="not json at all {{{")
    with pytest.raises(ResumeAnalysisParseError):
        analyze_resume("a real resume body text", client=fake)


def test_analyze_resume_empty_response_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response="")
    with pytest.raises(ResumeAnalysisParseError):
        analyze_resume("a real resume body text", client=fake)


def test_analyze_resume_non_object_json_raises_parse_error(
    fake_gemini_client_factory,
) -> None:
    import json

    fake = fake_gemini_client_factory(response=json.dumps(["not", "object"]))
    with pytest.raises(ResumeAnalysisParseError):
        analyze_resume("a real resume body text", client=fake)


# ---- API / config errors propagate as GeminiError ------------------------


def test_analyze_resume_propagates_api_error(fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        analyze_resume("a real resume body text", client=fake)


def test_analyze_resume_propagates_config_error(fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiConfigError("no key"))
    with pytest.raises(GeminiError):
        analyze_resume("a real resume body text", client=fake)


# ---- (11) session-state-style storage (result is serializable/reusable) --


def test_analysis_result_can_be_stored_and_reused(
    sample_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    """The returned analysis must be a stable object suitable for caching."""
    fake = fake_gemini_client_factory(response=sample_analysis_json)
    result = analyze_resume("real resume body text", client=fake)

    # Simulate session-state storage: round-trip via to_dict/from_dict.
    stored = result.to_dict()
    restored = ResumeAnalysis.from_dict(stored)
    assert restored.overall_score == result.overall_score
    assert restored.skills == result.skills
    assert len(restored.experience) == len(result.experience)


def test_default_client_creation_without_key_raises_config(
    monkeypatch,
    sample_analysis_json: str,  # not used — raised before any call
) -> None:
    """Without a key/env, the default client must raise GeminiConfigError,
    NOT silently proceed or hit the network."""
    # Ensure no key anywhere.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError):
        analyze_resume("a real resume body text")


def test_get_api_key_tolerates_missing_secrets_file(monkeypatch) -> None:
    """Regression: with no secrets.toml AND no env key, the secrets-reader
    must return None (not raise StreamlitSecretNotFoundError) so the app
    degrades to a friendly GeminiConfigError instead of a traceback."""
    from src.services.gemini import (
        GeminiConfigError,
        _from_streamlit_secrets,
        get_api_key,
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Must not raise even though no secrets.toml exists in the sandbox.
    assert _from_streamlit_secrets() is None
    with pytest.raises(GeminiConfigError):
        get_api_key()


def test_analyze_resume_never_returns_none_on_success(
    sample_analysis_json: str,
    fake_gemini_client_factory,
) -> None:
    result = analyze_resume("real resume body text", client=fake_gemini_client_factory(response=sample_analysis_json))
    assert result is not None
    assert isinstance(result, ResumeAnalysis)


def test_resume_analysis_error_is_geminin_error_independent() -> None:
    """ResumeAnalysisError is NOT a GeminiError — UI catches them separately."""
    assert not issubclass(ResumeAnalysisError, GeminiError)
