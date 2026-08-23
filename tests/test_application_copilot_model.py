"""Tests for the Application Copilot data models."""

from __future__ import annotations

import pytest

from src.modules.application_copilot.model import (
    ApplicationCopilotError,
    ApplicationCopilotParseError,
    ApplicationDraft,
    BulletSuggestion,
    CoverLetterDraft,
    EvidenceItem,
    EvidenceValidation,
    TailoredResumeSuggestion,
    ValidationStatus,
)


# ---- ValidationStatus enum ---------------------------------------------


def test_validation_status_members() -> None:
    assert {s.value for s in ValidationStatus} == {"pass", "warning", "fail"}


def test_validation_status_from_value() -> None:
    assert ValidationStatus.from_value("pass") is ValidationStatus.PASS
    assert ValidationStatus.from_value("WARNING") is ValidationStatus.WARNING
    assert ValidationStatus.from_value(ValidationStatus.FAIL) is ValidationStatus.FAIL


def test_validation_status_unknown_defaults_pass() -> None:
    assert ValidationStatus.from_value("bogus") is ValidationStatus.PASS
    assert ValidationStatus.from_value(None) is ValidationStatus.PASS


def test_validation_status_is_str_enum() -> None:
    assert ValidationStatus.PASS == "pass"


# ---- (1) EvidenceItem model -------------------------------------------


def test_evidence_item_defaults() -> None:
    e = EvidenceItem()
    assert e.evidence_id == ""
    assert e.source_type == ""
    assert e.source_text == ""
    assert e.category == ""


def test_evidence_item_from_dict() -> None:
    e = EvidenceItem.from_dict(
        {"evidence_id": "EXP-001", "source_type": "experience", "source_text": "x", "category": "experience"}
    )
    assert e.evidence_id == "EXP-001"
    assert e.source_type == "experience"


def test_evidence_item_from_dict_non_dict_raises() -> None:
    with pytest.raises(ApplicationCopilotParseError):
        EvidenceItem.from_dict("nope")  # type: ignore[arg-type]


# ---- (2) EvidenceValidation model -------------------------------------


def test_evidence_validation_defaults() -> None:
    v = EvidenceValidation()
    assert v.status is ValidationStatus.PASS
    assert v.supported_claims == []
    assert v.unsupported_claims == []
    assert v.warnings == []


def test_evidence_validation_from_dict() -> None:
    v = EvidenceValidation.from_dict(
        {"status": "fail", "unsupported_claims": ["x"], "warnings": ["y"]}
    )
    assert v.status is ValidationStatus.FAIL
    assert v.unsupported_claims == ["x"]


def test_evidence_validation_from_dict_non_dict_raises() -> None:
    with pytest.raises(ApplicationCopilotParseError):
        EvidenceValidation.from_dict(123)  # type: ignore[arg-type]


# ---- (3) ApplicationDraft model ---------------------------------------


def test_application_draft_defaults() -> None:
    d = ApplicationDraft()
    assert d.target_role == ""
    assert d.source_evidence_ids == []
    assert d.content == ""
    assert isinstance(d.evidence_validation, EvidenceValidation)


def test_application_draft_from_dict() -> None:
    d = ApplicationDraft.from_dict(
        {
            "target_role": "Eng",
            "source_evidence_ids": ["EXP-001"],
            "content": "letter",
            "evidence_validation": {"status": "pass"},
        }
    )
    assert d.target_role == "Eng"
    assert d.source_evidence_ids == ["EXP-001"]
    assert d.evidence_validation.status is ValidationStatus.PASS


def test_application_draft_from_dict_non_dict_raises() -> None:
    with pytest.raises(ApplicationCopilotParseError):
        ApplicationDraft.from_dict([])  # type: ignore[arg-type]


# ---- TailoredResumeSuggestion / CoverLetterDraft / BulletSuggestion ----


def test_tailored_suggestion_from_dict() -> None:
    s = TailoredResumeSuggestion.from_dict(
        {"section": "skills", "suggested_content": "x", "validation": {"status": "pass"}}
    )
    assert s.section == "skills"
    assert s.validation.status is ValidationStatus.PASS


def test_cover_letter_draft_from_dict() -> None:
    c = CoverLetterDraft.from_dict(
        {"cover_letter": "Dear...", "evidence_ids": ["EXP-001"], "warnings": ["w"]}
    )
    assert c.cover_letter == "Dear..."
    assert c.evidence_ids == ["EXP-001"]
    assert c.warnings == ["w"]


def test_bullet_suggestion_from_dict() -> None:
    b = BulletSuggestion.from_dict(
        {"original_bullet": "o", "optimized_bullet": "x", "validation": {"status": "warning"}}
    )
    assert b.original_bullet == "o"
    assert b.optimized_bullet == "x"
    assert b.validation.status is ValidationStatus.WARNING


def test_round_trip_to_dict() -> None:
    s = TailoredResumeSuggestion(
        section="skills",
        suggested_content="x",
        validation=EvidenceValidation(status=ValidationStatus.FAIL, unsupported_claims=["m"]),
    )
    d = s.to_dict()
    assert d["validation"]["status"] == "fail"
    again = TailoredResumeSuggestion.from_dict(d)
    assert again.validation.status is ValidationStatus.FAIL


def test_error_hierarchy() -> None:
    assert issubclass(ApplicationCopilotParseError, ApplicationCopilotError)
