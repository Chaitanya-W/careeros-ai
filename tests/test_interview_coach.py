"""Tests for the Interview Coach services (generate_questions,
evaluate_answer, build_report). Gemini is MOCKED — no real API key.

Covers: mocked question generation, invented-question rejection, answer
evaluation (mocked + deterministic + malformed + failure), unsupported
claim detection (TEST D), score clamping, report/readiness.
"""

from __future__ import annotations

import json

import pytest

from src.modules.voice_interview.coach import (
    build_report,
    detect_unsupported_answer_claims,
    evaluate_answer,
    generate_questions,
)
from src.modules.voice_interview.model import (
    AnswerEvaluation,
    Difficulty,
    InterviewCategory,
    InterviewQuestion,
)
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


def _fake_client(response: str = "", *, raise_exc=None):
    class _F:
        def __init__(self):
            self.calls = []

        def generate_json(self, *, system_prompt, user_text, response_schema, **kw):
            self.calls.append({"system_prompt": system_prompt, "user_text": user_text})
            if raise_exc is not None:
                raise raise_exc
            return response

    return _F()


# ---- (24) mocked Gemini question generation ----------------------


def test_generate_questions_with_gemini_clean(
    sample_interview_inputs, interview_question_gen_json
) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(interview_question_gen_json)
    qs = generate_questions(resume, job, match, sgr, n=2, difficulty=Difficulty.MEDIUM, client=fake, evidence=evidence)
    assert len(qs) == 2
    assert all(isinstance(q, InterviewQuestion) for q in qs)
    # The prompt enforces evidence-grounding.
    assert "evidence-grounded" in fake.calls[0]["system_prompt"].lower()
    assert "do not invent" in fake.calls[0]["system_prompt"].lower()


def test_generate_questions_deterministic_no_client(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    qs = generate_questions(resume, job, match, sgr, n=5, client=None, evidence=evidence)
    assert len(qs) == 5
    assert all(q.question for q in qs)


# ---- (25) malformed Gemini response ------------------------------


def test_generate_questions_malformed_falls_back(sample_interview_inputs) -> None:
    """Malformed Gemini JSON -> fall back to deterministic (no crash)."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client("not json {{{")
    qs = generate_questions(resume, job, match, sgr, n=4, client=fake, evidence=evidence)
    assert len(qs) == 4  # deterministic fallback


def test_generate_questions_empty_response_falls_back(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client("")
    qs = generate_questions(resume, job, match, sgr, n=3, client=fake, evidence=evidence)
    assert len(qs) == 3


# ---- (26) Gemini failure ------------------------------------------


def test_generate_questions_propagates_api_error(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client("", raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        generate_questions(resume, job, match, sgr, n=3, client=fake, evidence=evidence)


# ---- (27-31) Gemini cannot invent X (questions rejected via validator) -----


def test_gemini_invented_skill_replaced(sample_interview_inputs, interview_question_invented_json) -> None:
    """A Gemini question that invents TensorFlow must be replaced with the deterministic fallback."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    fake = _fake_client(interview_question_invented_json)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert len(qs) == 1
    # The invented TensorFlow question must NOT appear; the deterministic fallback is used.
    assert "TensorFlow" not in qs[0].question


def test_gemini_invented_project_replaced(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    invented = json.dumps({"questions": [{"question": "Tell me how you deployed your AWS Kubernetes platform.", "category": "technical"}]})
    fake = _fake_client(invented)
    qs = generate_questions(resume, job, match, sgr, n=1, client=fake, evidence=evidence)
    assert "AWS Kubernetes platform" not in qs[0].question


# ---- (32) unsupported answer claim (TEST D) -----------------------


def test_unsupported_answer_claim_detected(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    claims = detect_unsupported_answer_claims(
        "I deployed this system to 500 AWS servers.", evidence, resume, job, match
    )
    assert claims, "unsupported claims should be detected"
    joined = " ".join(claims).lower()
    assert "aws" in joined or "500" in joined


def test_supported_answer_claim_no_flag(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    claims = detect_unsupported_answer_claims(
        "I used Python to build the CareerOS project.", evidence, resume, job, match
    )
    assert not claims  # grounded in resume evidence


# ---- (33) supported candidate claim ------------------------------ (covered above)


# ---- (34)(35) answer evaluation (deterministic + mocked) ----------


def test_evaluate_answer_deterministic(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="What role did Python play?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    ev = evaluate_answer(q, "I used Python to build CareerOS with Streamlit.", resume, job, match, client=None, evidence=evidence)
    assert isinstance(ev, AnswerEvaluation)
    assert 0 <= ev.overall_score <= 10
    assert ev.question_id == "IQ-001"


def test_evaluate_answer_with_gemini(sample_interview_inputs, interview_eval_json) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    fake = _fake_client(interview_eval_json)
    ev = evaluate_answer(q, "I used Python to build CareerOS.", resume, job, match, client=fake, evidence=evidence)
    assert ev.overall_score == 7.0
    assert any("Clear explanation" in s for s in ev.strengths)


# ---- (36) malformed evaluation ------------------------------------


def test_evaluate_answer_malformed_falls_back(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    fake = _fake_client("not json {{{")
    ev = evaluate_answer(q, "I used Python.", resume, job, match, client=fake, evidence=evidence)
    # Falls back to deterministic evaluation (no crash).
    assert isinstance(ev, AnswerEvaluation)
    assert 0 <= ev.overall_score <= 10


# ---- (37) Gemini failure on evaluation ---------------------------


def test_evaluate_answer_propagates_api_error(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    fake = _fake_client("", raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        evaluate_answer(q, "answer", resume, job, match, client=fake, evidence=evidence)


# ---- (38) score clamping -----------------------------------------


def test_evaluate_answer_clamps_gemini_scores(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    over = json.dumps({"overall_score": 15.0, "relevance_score": 20.0, "technical_score": -5.0})
    fake = _fake_client(over)
    ev = evaluate_answer(q, "I used Python.", resume, job, match, client=fake, evidence=evidence)
    assert ev.overall_score == 10.0
    assert ev.relevance_score == 10.0
    assert ev.technical_score == 0.0


# ---- (43) final report -------------------------------------------


def test_build_report(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    qs = generate_questions(resume, job, match, sgr, n=4, client=None, evidence=evidence)
    evals = {q.question_id: evaluate_answer(q, "I used Python to build the project.", resume, job, match, client=None, evidence=evidence) for q in qs}
    report = build_report(qs, evals, "Backend Engineer")
    assert 0 <= report.readiness_score <= 100
    assert report.completed_questions == 4
    assert report.total_questions == 4


def test_build_report_empty() -> None:
    report = build_report([], {}, "Eng")
    assert report.readiness_score == 0.0
    assert report.completed_questions == 0


# ---- (44) readiness calculation ----------------------------------


def test_readiness_formula_documented(sample_interview_inputs) -> None:
    resume, job, match, sgr, evidence = sample_interview_inputs
    qs = generate_questions(resume, job, match, sgr, n=3, client=None, evidence=evidence)
    evals = {q.question_id: evaluate_answer(q, "I used Python and SQL to build services with measurable outcomes at Acme.", resume, job, match, client=None, evidence=evidence) for q in qs}
    report = build_report(qs, evals, "Eng")
    # Readiness is a weighted formula in 0-100.
    assert 0 <= report.readiness_score <= 100


# ---- (49) no API key ---------------------------------------------


def test_generate_questions_no_key_works(sample_interview_inputs, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, sgr, evidence = sample_interview_inputs
    qs = generate_questions(resume, job, match, sgr, n=4, client=None, evidence=evidence)
    assert len(qs) == 4


def test_evaluate_answer_no_key_works(sample_interview_inputs, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    ev = evaluate_answer(q, "I used Python.", resume, job, match, client=None, evidence=evidence)
    assert ev.overall_score >= 0


# ---- unsupported claims authoritative (Gemini can't suppress) ----


def test_unsupported_claims_authoritative(sample_interview_inputs, interview_eval_json) -> None:
    """Even if Gemini returns empty unsupported_claims, the deterministic
    detection is authoritative."""
    resume, job, match, sgr, evidence = sample_interview_inputs
    q = InterviewQuestion(question_id="IQ-001", question="q?", target_skill="Python", category=InterviewCategory.TECHNICAL)
    fake = _fake_client(interview_eval_json)  # Gemini says no unsupported claims
    ev = evaluate_answer(q, "I deployed 500 AWS servers.", resume, job, match, client=fake, evidence=evidence)
    # The deterministic detection overrides Gemini's empty list.
    assert ev.unsupported_claims, "unsupported claims must be detected even if Gemini ignores them"
    assert ev.evidence_alignment == "unsupported"
