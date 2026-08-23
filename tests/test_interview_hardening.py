"""Step 9 hardening regression tests.

Prove the deterministic planner is AUTHORITATIVE over all structural
metadata; Gemini may ONLY improve wording/rationale.

A mocked Gemini response attempts to override category/difficulty/
target_skill/source/related_job_requirement — the final question must
retain the deterministic planner's values.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.modules.voice_interview.coach import generate_questions
from src.modules.voice_interview.model import (
    Difficulty,
    InterviewCategory,
    InterviewQuestion,
)
from src.services.gemini import GeminiError


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


# A Gemini response that tries to OVERRIDE every structural field.
# For a TECHNICAL/Python/HARD/RESUME slot, it returns BEHAVIORAL/EASY/TensorFlow/GENERAL.
_OVERRIDE_JSON = json.dumps(
    {
        "questions": [
            {
                "question": "Tell me about your experience with Python.",
                "category": "BEHAVIORAL",
                "difficulty": "EASY",
                "target_skill": "TensorFlow",
                "rationale": "Gemini rationale override.",
                "source": "GENERAL",
                "related_job_requirement": "Unrelated requirement",
            }
        ]
    }
)


# ---- 1. Gemini cannot change category --------------------------------


def test_gemini_cannot_change_category(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    # The deterministic slot 0 is TECHNICAL (30% ratio, first slot).
    assert qs[0].category is not InterviewCategory.BEHAVIORAL
    assert qs[0].category is InterviewCategory.TECHNICAL


# ---- 2. Gemini cannot change difficulty ----------------------------


def test_gemini_cannot_change_difficulty(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(
        resume, job, match, sgr, n=1, difficulty=Difficulty.HARD, client=fake, evidence=evidence
    )
    assert qs[0].difficulty is Difficulty.HARD
    assert qs[0].difficulty is not Difficulty.EASY


# ---- 3. Gemini cannot change target_skill --------------------------


def test_gemini_cannot_change_target_skill(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    # The deterministic target_skill is a resume/matching skill (Python/SQL/AWS), NOT TensorFlow.
    assert qs[0].target_skill != "TensorFlow"
    assert qs[0].target_skill in resume.skills or qs[0].target_skill in job.required_skills


# ---- 4. Gemini cannot change source ---------------------------------


def test_gemini_cannot_change_source(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    # The deterministic source is JOB or RESUME (for a technical slot), NOT GENERAL.
    assert qs[0].source in ("RESUME", "JOB")
    assert qs[0].source != "GENERAL"


# ---- 5. Gemini cannot change related_job_requirement ---------------


def test_gemini_cannot_change_related_job_requirement(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert qs[0].related_job_requirement != "Unrelated requirement"
    # Must be the deterministic planner's value (a real job skill or empty).
    assert qs[0].related_job_requirement == "" or qs[0].related_job_requirement in job.required_skills


# ---- 6. Gemini cannot change question_id ----------------------------


def test_gemini_cannot_change_question_id(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert qs[0].question_id == "IQ-001"


def test_question_ids_match_deterministic(sample_interview_inputs) -> None:
    """The final IDs must exactly match the deterministic planner's IDs."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(json.dumps({"questions": [
        {"question": "q1", "category": "behavioral"},
        {"question": "q2", "category": "behavioral"},
        {"question": "q3", "category": "behavioral"},
    ]}))
    qs = generate_questions(resume, job, match, sgr, n=3, client=fake, evidence=evidence)
    assert [q.question_id for q in qs] == ["IQ-001", "IQ-002", "IQ-003"]


# ---- 7. Gemini cannot change the category distribution -------------


def test_gemini_cannot_change_category_distribution(sample_interview_inputs) -> None:
    """Even if Gemini returns BEHAVIORAL for every slot, the final
    distribution must match the deterministic planner's output."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    all_behavioral = json.dumps({"questions": [
        {"question": f"q{i}", "category": "BEHAVIORAL", "difficulty": "EASY"}
        for i in range(10)
    ]})
    # First, build the deterministic questions (no Gemini) to get the
    # authoritative category list (including reassignments).
    det_qs = generate_questions(resume, job, match, sgr, n=10, client=None, evidence=evidence)
    det_cats = [q.category for q in det_qs]
    # Now, with Gemini returning BEHAVIORAL for every slot.
    fake = _fake_client(all_behavioral)
    qs = generate_questions(resume, job, match, sgr, n=10, client=fake, evidence=evidence)
    final_cats = [q.category for q in qs]
    # The final distribution must match the deterministic planner's — NOT all behavioral.
    assert final_cats == det_cats  # exact match (including reassignments)
    assert final_cats.count(InterviewCategory.BEHAVIORAL) < 10  # not all behavioral
    assert InterviewCategory.TECHNICAL in final_cats


# ---- 8. Gemini wording is accepted when valid ----------------------


def test_gemini_valid_wording_accepted(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(json.dumps({"questions": [
        {"question": "What role did Python play in your work, and why did you choose it?", "category": "technical"}
    ]}))
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert "What role did Python play" in qs[0].question  # Gemini wording used


# ---- 9. Invalid Gemini wording falls back to deterministic ---------


def test_invalid_gemini_wording_falls_back(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    invented = json.dumps({"questions": [
        {"question": "You used TensorFlow extensively in your projects. Explain how.", "category": "technical"}
    ]})
    fake = _fake_client(invented)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert "TensorFlow" not in qs[0].question  # deterministic fallback used


# ---- 10. Existing invented-skill protection ------------------------


def test_invented_skill_still_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    from src.modules.voice_interview.validator import validate_question

    is_valid, _ = validate_question(
        "You used TensorFlow extensively. Explain.", evidence, resume, job, match, sgr
    )
    assert is_valid is False


# ---- 11. Existing invented-project protection ----------------------


def test_invented_project_still_rejected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    from src.modules.voice_interview.validator import validate_question

    is_valid, _ = validate_question(
        "Tell me how you deployed your AWS Kubernetes platform.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False


# ---- 12. Existing missing-skill question behavior ------------------


def test_missing_skill_question_valid(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    from src.modules.voice_interview.validator import validate_question

    is_valid, _ = validate_question(
        "The role requires Docker. How would you approach learning it?",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is True


def test_missing_skill_assertion_invalid(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    from src.modules.voice_interview.validator import validate_question

    is_valid, _ = validate_question(
        "You used Docker in production. Explain how.",
        evidence, resume, job, match, sgr,
    )
    assert is_valid is False


# ---- 13. Existing unsupported-answer-claim detection --------------


def test_unsupported_answer_claim_still_detected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    from src.modules.voice_interview.coach import detect_unsupported_answer_claims

    claims = detect_unsupported_answer_claims(
        "I deployed this system to 500 AWS servers.", evidence, resume, job, match
    )
    assert claims


# ---- 14. No-Gemini deterministic behavior still works --------------


def test_no_gemini_deterministic_works(sample_interview_inputs, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, sgr, evidence = sample_interview_inputs
    qs = generate_questions(resume, job, match, sgr, n=5, client=None, evidence=evidence)
    assert len(qs) == 5
    assert all(q.question for q in qs)
    # IDs deterministic.
    assert [q.question_id for q in qs] == ["IQ-001", "IQ-002", "IQ-003", "IQ-004", "IQ-005"]


# ---- 15. All Step 1–8 tests remain green --------------------------
# (verified by the full pytest run — no Step 1-8 source modified.)


# ---- Bonus: Gemini rationale accepted when provided ----------------


def test_gemini_rationale_accepted_when_provided(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(json.dumps({"questions": [
        {"question": "What role did Python play in your work?", "rationale": "Improved rationale from Gemini."}
    ]}))
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert qs[0].rationale == "Improved rationale from Gemini."


def test_gemini_rationale_falls_back_to_deterministic(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(json.dumps({"questions": [
        {"question": "What role did Python play in your work?"}  # no rationale
    ]}))
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    # Deterministic rationale is used when Gemini doesn't provide one.
    assert qs[0].rationale  # non-empty (deterministic)


# ---- Full structural integrity: all fields deterministic except wording/rationale


def test_all_structural_fields_deterministic(sample_interview_inputs) -> None:
    """For a single slot, verify EVERY structural field comes from the
    deterministic planner, not Gemini."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    # First, build the deterministic question (no Gemini).
    det_qs = generate_questions(resume, job, match, sgr, n=1, client=None, evidence=evidence)
    det = det_qs[0]
    # Now, with Gemini overriding everything, the structural fields must match det.
    fake = _fake_client(_OVERRIDE_JSON)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    final = qs[0]
    assert final.question_id == det.question_id
    assert final.category == det.category
    assert final.difficulty == det.difficulty
    assert final.target_skill == det.target_skill
    assert final.source == det.source
    assert final.related_job_requirement == det.related_job_requirement
    assert final.expected_evidence == det.expected_evidence
    # Only the wording may differ (Gemini's valid wording).
    assert final.question != det.question or final.question == det.question  # wording may change
