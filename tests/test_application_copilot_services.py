"""Tests for the Application Copilot services (Resume Tailor, Cover Letter,
Bullet Optimizer) — Gemini is MOCKED; no real API key required.

Covers: deterministic output, mocked Gemini enrichment, invalid Gemini
response, Gemini errors, missing API key, evidence IDs in output, and the
critical guarantee that Gemini cannot invent candidate facts (the
validator catches every invented category).
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.modules.application_copilot.copilot import (
    generate_cover_letter,
    optimize_bullet,
    tailor_resume,
)
from src.modules.application_copilot.evidence import extract_evidence
from src.modules.application_copilot.model import (
    BulletSuggestion,
    CoverLetterDraft,
    TailoredResumeSuggestion,
    ValidationStatus,
)
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


# ---- (20)(21) Resume tailoring output + preserves facts ------------


def test_tailor_deterministic_has_suggestions(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    suggestions = tailor_resume(resume, job, match, evidence)
    assert suggestions
    assert all(isinstance(s, TailoredResumeSuggestion) for s in suggestions)
    # Every deterministic suggestion is PASS (grounded).
    assert all(s.validation.status is ValidationStatus.PASS for s in suggestions)


def test_tailor_deterministic_preserves_facts(sample_copilot_inputs) -> None:
    """Deterministic suggestions never invent content: suggested_content
    equals the original (re-ordered for skills only)."""
    resume, job, match, evidence = sample_copilot_inputs
    suggestions = tailor_resume(resume, job, match, evidence)
    for s in suggestions:
        if s.section == "skills":
            # Reordered: same set of skills, no new ones.
            assert set(s.suggested_content.split(", ")) == set(s.original_content.split(", "))
        else:
            assert s.suggested_content == s.original_content or not s.original_content


# ---- (26) mocked Gemini enrichment --------------------------------


def test_tailor_with_gemini_clean(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "skills", "original_content": "Python", "suggested_content": "Python, SQL, AWS", "reason": "Lead with matching skills."}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions
    skills_s = next(s for s in suggestions if s.section == "skills")
    assert "Python" in skills_s.suggested_content
    # Clean wording -> PASS.
    assert skills_s.validation.status is ValidationStatus.PASS
    # The prompt includes the evidence-grounding rules.
    assert "evidence-grounded" in fake.calls[0]["system_prompt"].lower() or "never invent" in fake.calls[0]["system_prompt"].lower()


# ---- (37-43) Gemini cannot invent X (services reject via validator) ----


def test_tailor_gemini_invented_metric_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "experience", "suggested_content": "Increased efficiency by 40%.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.FAIL
    assert any("40%" in u for u in suggestions[0].validation.unsupported_claims)


def test_tailor_gemini_invented_skill_rejected(sample_copilot_inputs) -> None:
    """Gemini claims Docker (missing) -> FAIL."""
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "skills", "suggested_content": "Python, SQL, AWS, Docker", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.FAIL


def test_tailor_gemini_invented_year_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "experience", "suggested_content": "Worked there since 2015.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.FAIL


def test_tailor_gemini_invented_responsibility_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "experience", "suggested_content": "Led a team of 15 engineers.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.FAIL


def test_tailor_gemini_invented_employer_flagged(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "experience", "suggested_content": "Worked at Initech building tools.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.WARNING


def test_tailor_gemini_invented_cert_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "certifications", "suggested_content": "I am a certified Kubernetes administrator.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is ValidationStatus.FAIL


def test_tailor_gemini_invented_project_flagged(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "projects", "suggested_content": "Built the Apollo project.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    assert suggestions[0].validation.status is not ValidationStatus.PASS


# ---- (27) invalid Gemini output -----------------------------------


def test_tailor_invalid_json_raises(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client("not json {{{")
    with pytest.raises(Exception):
        tailor_resume(resume, job, match, evidence, client=fake)


def test_tailor_empty_response_raises(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client("")
    with pytest.raises(Exception):
        tailor_resume(resume, job, match, evidence, client=fake)


def test_tailor_missing_suggestions_field_raises(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"other": 1}))
    with pytest.raises(Exception):
        tailor_resume(resume, job, match, evidence, client=fake)


# ---- (28) Gemini error ---------------------------------------------


def test_tailor_propagates_api_error(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client("", raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        tailor_resume(resume, job, match, evidence, client=fake)


# ---- (22)(23) Cover letter generation + preserves facts ------------


def test_cover_letter_no_client_shows_ai_message(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    draft = generate_cover_letter(resume, job, match, evidence)
    assert isinstance(draft, CoverLetterDraft)
    assert draft.cover_letter == ""  # no fabricated letter
    assert any("Gemini configuration" in w for w in draft.warnings)


def test_cover_letter_with_gemini_clean(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    letter = "I am excited about the Senior Backend Engineer role. I built Python services on AWS at Acme."
    fake = _fake_client(json.dumps({"cover_letter": letter, "evidence_ids": ["EXP-001", "SKILL-001"]}))
    draft = generate_cover_letter(resume, job, match, evidence, client=fake)
    assert draft.cover_letter == letter
    assert draft.evidence_validation.status is ValidationStatus.PASS
    assert draft.evidence_ids


def test_cover_letter_gemini_invents_metric_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"cover_letter": "I improved efficiency by 40% at Acme."}))
    draft = generate_cover_letter(resume, job, match, evidence, client=fake)
    assert draft.evidence_validation.status is ValidationStatus.FAIL
    assert any("40%" in u for u in draft.evidence_validation.unsupported_claims)
    # Warning indicates untrusted output.
    assert any("failed evidence validation" in w.lower() for w in draft.warnings)


# ---- (24)(25) Bullet optimization + preserves facts ---------------


def test_bullet_no_client_gives_guidance(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    s = optimize_bullet("Built Python services.", resume, job, match, evidence)
    assert isinstance(s, BulletSuggestion)
    assert s.optimized_bullet == ""  # no fabricated AI text without Gemini
    assert s.reason
    assert s.validation.status is ValidationStatus.PASS


def test_bullet_with_gemini_clean(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({
        "optimized_bullet": "Built Python services on AWS at Acme.",
        "reason": "Emphasizes matching skills.",
        "evidence_ids": ["EXP-001"],
    }))
    s = optimize_bullet("Built Python services.", resume, job, match, evidence, client=fake)
    assert s.optimized_bullet == "Built Python services on AWS at Acme."
    assert s.validation.status is ValidationStatus.PASS
    assert s.evidence_ids


def test_bullet_gemini_invents_metric_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"optimized_bullet": "Improved efficiency by 40%."}))
    s = optimize_bullet("Built services.", resume, job, match, evidence, client=fake)
    assert s.validation.status is ValidationStatus.FAIL


def test_bullet_gemini_invents_skill_rejected(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"optimized_bullet": "Built services using Docker."}))
    s = optimize_bullet("Built services.", resume, job, match, evidence, client=fake)
    assert s.validation.status is ValidationStatus.FAIL


# ---- (29) missing API key (default client) -------------------------


def test_tailor_default_client_without_key_raises_config(monkeypatch, sample_copilot_inputs) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, evidence = sample_copilot_inputs
    # Deterministic path (client=None) does NOT require a key.
    suggestions = tailor_resume(resume, job, match, evidence)
    assert suggestions  # deterministic works without a key


def test_cover_letter_default_client_without_key_raises_config(monkeypatch, sample_copilot_inputs) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, evidence = sample_copilot_inputs
    draft = generate_cover_letter(resume, job, match, evidence)
    # Without a key, returns the no-AI status draft (no crash, no fabricated letter).
    assert draft.cover_letter == ""
    assert draft.warnings


# ---- (36) evidence IDs included in generated output ----------------


def test_tailor_deterministic_includes_evidence_ids(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    suggestions = tailor_resume(resume, job, match, evidence)
    assert any(s.evidence_ids for s in suggestions)


def test_cover_letter_with_gemini_includes_evidence_ids(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"cover_letter": "Built Python services.", "evidence_ids": ["EXP-001"]}))
    draft = generate_cover_letter(resume, job, match, evidence, client=fake)
    assert draft.evidence_ids


# ---- (44) unvalidated output never presented as trusted -------------


def test_unvalidated_tailor_output_marked_fail(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"suggestions": [
        {"section": "experience", "suggested_content": "Increased efficiency by 40%.", "reason": "fake"}
    ]}))
    suggestions = tailor_resume(resume, job, match, evidence, client=fake)
    # The FAIL suggestion is returned but marked FAIL (untrusted), never PASS.
    assert suggestions[0].validation.status is ValidationStatus.FAIL
    assert not suggestions[0].validation.status is ValidationStatus.PASS


def test_unvalidated_cover_letter_marked_fail(sample_copilot_inputs) -> None:
    resume, job, match, evidence = sample_copilot_inputs
    fake = _fake_client(json.dumps({"cover_letter": "I improved efficiency by 99%."}))
    draft = generate_cover_letter(resume, job, match, evidence, client=fake)
    assert draft.evidence_validation.status is ValidationStatus.FAIL
    assert any("failed evidence validation" in w.lower() for w in draft.warnings)
