"""Tests for the deterministic Evidence Validator (PASS/WARNING/FAIL)."""

from __future__ import annotations

from types import SimpleNamespace

from src.modules.application_copilot.evidence import extract_evidence
from src.modules.application_copilot.model import ValidationStatus
from src.modules.application_copilot.validator import validate_claims
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis


def _resume_with(text: str) -> ResumeAnalysis:
    """A minimal resume whose evidence corpus contains ``text``."""
    return ResumeAnalysis(
        skills=["Python", "SQL", "AWS"],
        experience=[],
        professional_summary=text,
    )


def _match(missing: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        overall_match_score=50,
        missing_skills=missing,
        matching_skills=["Python", "SQL", "AWS"],
    )


# ---- (11) unsupported technology detection ---------------------------


def test_unsupported_technology_detected() -> None:
    """A missing required skill claimed in text -> FAIL."""
    resume = _resume_with("I know Python.")
    job = JobAnalysis(job_title="Eng", required_skills=["Python", "Docker"])
    match = _match(["Docker"])
    evidence = extract_evidence(resume)
    v = validate_claims("I have Docker experience.", evidence, resume=resume, job=job, match=match)
    assert v.status is ValidationStatus.FAIL
    assert any("Docker" in u for u in v.unsupported_claims)


# ---- (12) unsupported metric detection -------------------------------


def test_unsupported_metric_detected() -> None:
    resume = _resume_with("Built services.")
    evidence = extract_evidence(resume)
    v = validate_claims("Increased efficiency by 40%.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL
    assert any("40%" in u for u in v.unsupported_claims)


def test_supported_metric_passes() -> None:
    resume = _resume_with("Serving 100k users at 99% uptime.")
    evidence = extract_evidence(resume)
    v = validate_claims("Served 100k users at 99% uptime.", evidence, resume=resume)
    assert v.status is ValidationStatus.PASS


# ---- (13) unsupported employer detection ----------------------------


def test_unsupported_employer_warns() -> None:
    resume = _resume_with("Worked at Acme using Python.")
    evidence = extract_evidence(resume)
    v = validate_claims("Previously worked at Google building tools.", evidence, resume=resume)
    # Google is a proper noun not in evidence -> warning (status WARNING).
    assert v.status is ValidationStatus.WARNING
    assert any("Google" in w for w in v.warnings)


def test_supported_employer_passes() -> None:
    resume = _resume_with("Worked at Acme using Python.")
    evidence = extract_evidence(resume)
    v = validate_claims("Worked at Acme building Python tools.", evidence, resume=resume)
    assert v.status is ValidationStatus.PASS


# ---- (14) unsupported certification detection -----------------------


def test_unsupported_certification_detected() -> None:
    resume = _resume_with("Python engineer.")  # no certs
    evidence = extract_evidence(resume)
    v = validate_claims("I am AWS certified.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL
    assert any("Certification" in u for u in v.unsupported_claims)


def test_supported_certification_passes(sample_copilot_resume) -> None:
    evidence = extract_evidence(sample_copilot_resume)
    v = validate_claims("I hold the AWS Certified Developer certification.", evidence, resume=sample_copilot_resume)
    assert v.status is ValidationStatus.PASS


# ---- (15) unsupported project claim detection -----------------------


def test_unsupported_project_name_warns() -> None:
    resume = _resume_with("Built CareerOS using Python.")
    evidence = extract_evidence(resume)
    v = validate_claims("Led the Apollo project at Acme.", evidence, resume=resume)
    # "Apollo" is a proper noun not in evidence -> warning.
    assert v.status is ValidationStatus.WARNING
    assert any("Apollo" in w for w in v.warnings)


# ---- (16) unsupported date/year detection --------------------------


def test_unsupported_year_detected() -> None:
    resume = _resume_with("Worked from 2019 to present.")
    evidence = extract_evidence(resume)
    v = validate_claims("Worked there since 2021.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL
    assert any("2021" in u for u in v.unsupported_claims)


def test_supported_year_passes() -> None:
    resume = _resume_with("Graduated in 2019.")
    evidence = extract_evidence(resume)
    v = validate_claims("Graduated in 2019.", evidence, resume=resume)
    assert v.status is ValidationStatus.PASS


# ---- (17) PASS validation --------------------------------------------


def test_pass_clean_text(sample_copilot_resume, sample_copilot_job, sample_copilot_match) -> None:
    evidence = extract_evidence(sample_copilot_resume)
    text = "Built Python services on AWS at Acme, graduated in 2019."
    v = validate_claims(text, evidence, resume=sample_copilot_resume, job=sample_copilot_job, match=sample_copilot_match)
    assert v.status is ValidationStatus.PASS
    assert not v.unsupported_claims


def test_pass_empty_text() -> None:
    v = validate_claims("", [], resume=ResumeAnalysis())
    assert v.status is ValidationStatus.PASS


# ---- (18) WARNING validation ----------------------------------------


def test_warning_only_proper_noun_issues() -> None:
    resume = _resume_with("Worked at Acme.")
    evidence = extract_evidence(resume)
    v = validate_claims("Worked at Acme, also contributed at Initech.", evidence, resume=resume)
    assert v.status is ValidationStatus.WARNING
    assert not v.unsupported_claims
    assert any("Initech" in w for w in v.warnings)


# ---- (19) FAIL validation -------------------------------------------


def test_fail_invented_metric() -> None:
    resume = _resume_with("Built services.")
    evidence = extract_evidence(resume)
    v = validate_claims("Grew revenue 5x and saved 40%.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL


def test_fail_invented_responsibility_count() -> None:
    """Invented team-size/responsibility metric -> FAIL."""
    resume = _resume_with("Worked as an engineer.")
    evidence = extract_evidence(resume)
    v = validate_claims("Led a team of 15 engineers.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL
    assert any("15 engineers" in u for u in v.unsupported_claims)


# ---- (37-43) Gemini cannot invent X (validator catches each) -------


def test_validator_catches_invented_technology() -> None:
    resume = _resume_with("Python.")
    job = JobAnalysis(job_title="Eng", required_skills=["Python", "Kubernetes"])
    match = _match(["Kubernetes"])
    evidence = extract_evidence(resume)
    v = validate_claims("I have Kubernetes experience.", evidence, resume=resume, job=job, match=match)
    assert v.status is ValidationStatus.FAIL


def test_validator_catches_invented_metric() -> None:
    resume = _resume_with("Built services.")
    evidence = extract_evidence(resume)
    v = validate_claims("Improved performance by 99%.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL


def test_validator_catches_invented_employer() -> None:
    resume = _resume_with("Worked at Acme.")
    evidence = extract_evidence(resume)
    v = validate_claims("Worked at Amazon.", evidence, resume=resume)
    assert v.status is not ValidationStatus.PASS


def test_validator_catches_invented_certification() -> None:
    resume = _resume_with("Engineer.")
    evidence = extract_evidence(resume)
    v = validate_claims("Certified Kubernetes Administrator certification.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL


def test_validator_catches_invented_project() -> None:
    resume = _resume_with("Built CareerOS.")
    evidence = extract_evidence(resume)
    v = validate_claims("Built the Apollo project.", evidence, resume=resume)
    assert v.status is not ValidationStatus.PASS


def test_validator_catches_invented_date() -> None:
    resume = _resume_with("Started in 2019.")
    evidence = extract_evidence(resume)
    v = validate_claims("Started in 2015.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL


def test_validator_catches_invented_responsibility() -> None:
    resume = _resume_with("Engineer.")
    evidence = extract_evidence(resume)
    v = validate_claims("Managed 20 engineers across three teams.", evidence, resume=resume)
    assert v.status is ValidationStatus.FAIL


# ---- conservative behavior documented --------------------------------


def test_validator_never_silently_approves_unsupported() -> None:
    """Any unsupported claim -> at least WARNING (usually FAIL)."""
    resume = _resume_with("Python.")
    evidence = extract_evidence(resume)
    v = validate_claims("Increased revenue by 50%.", evidence, resume=resume)
    assert v.status is not ValidationStatus.PASS
    assert v.unsupported_claims


def test_job_supplied_tokens_not_flagged(sample_copilot_resume, sample_copilot_job, sample_copilot_match) -> None:
    """The target role/company are job-supplied; mentioning them is fine."""
    evidence = extract_evidence(sample_copilot_resume)
    text = f"Interested in the {sample_copilot_job.job_title} role at {sample_copilot_job.company}."
    v = validate_claims(text, evidence, resume=sample_copilot_resume, job=sample_copilot_job, match=sample_copilot_match)
    # No unsupported claims about the job's own title/company.
    assert v.status is ValidationStatus.PASS
