"""Tests for the question validator (adversarial A, B, C + grounding)."""

from __future__ import annotations

from src.modules.voice_interview.validator import validate_question


# ---- TEST C: missing job skill — valid learning question vs invalid assertion


def test_valid_missing_skill_question(sample_interview_inputs) -> None:
    """'The role requires Kubernetes. How would you approach learning it?' -> VALID."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "The role requires Kubernetes. How would you approach learning and using it?",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is True, f"valid question rejected: {reasons}"


def test_invalid_asserts_missing_skill(sample_interview_inputs) -> None:
    """'You used Kubernetes in production. Explain how.' -> INVALID."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "You used Kubernetes in production. Explain how.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False
    assert any("Kubernetes" in r for r in reasons)


# ---- TEST A: invented project -------------------------------------


def test_invented_project_rejected(sample_interview_inputs) -> None:
    """Gemini: 'Tell me how you deployed your AWS Kubernetes platform.' -> rejected."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "Tell me how you deployed your AWS Kubernetes platform.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False, f"invented project accepted: {reasons}"


# ---- TEST B: invented skill ---------------------------------------


def test_invented_skill_rejected(sample_interview_inputs) -> None:
    """'You used TensorFlow extensively...' (TensorFlow not in resume) -> rejected."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "You used TensorFlow extensively in your projects. Explain.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False
    assert any("TensorFlow" in r for r in reasons)


# ---- invented employer / cert / metric / date ---------------------


def test_invented_employer_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "Tell me about your time at Google building scalable systems.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False
    assert any("Google" in r for r in reasons)


def test_invented_cert_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "How did you earn your Certified Kubernetes Administrator certification?",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False
    assert any("Certification" in r or "certification" in r.lower() for r in reasons)


def test_invented_metric_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "How did you achieve a 40% improvement in efficiency?",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False


def test_invented_date_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question(
        "Tell me about your work at Acme in 2015.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False  # 2015 not in resume evidence


# ---- valid grounded questions ------------------------------------


def test_valid_resume_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, _ = validate_question(
        "Walk me through your role at Acme and your key contributions.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is True


def test_valid_project_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, _ = validate_question(
        "Explain the architecture and your contribution to CareerOS.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is True


def test_valid_technical_question(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, _ = validate_question(
        "What role did Python play in your work, and why did you choose it?",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is True


def test_empty_question_invalid(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    is_valid, reasons = validate_question("", evidence, resume, job, match)
    assert is_valid is False
