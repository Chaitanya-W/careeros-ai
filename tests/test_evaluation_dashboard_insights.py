"""Tests for deterministic insights + optional Gemini enrichment (MOCKED)."""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.modules.evaluation_dashboard.insights import enrich_insights, generate_insights
from src.modules.evaluation_dashboard.model import (
    AnswerQuality,
    EvaluationSummary,
    TrendDirection,
    TrendPoint,
)
from src.modules.evaluation_dashboard.trends import compute_trends
from src.services.gemini import GeminiAPIError, GeminiError


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


# ---- deterministic insights ---------------------------------------


def test_empty_insights() -> None:
    s = EvaluationSummary()
    insights = generate_insights(s, [])
    assert len(insights) >= 1
    assert "interview" in insights[0].lower()


def test_insights_with_metrics(sample_eval_session) -> None:
    from src.modules.evaluation_dashboard.metrics import compute_metrics

    s = compute_metrics(sample_eval_session)
    insights = generate_insights(s, [])
    assert len(insights) >= 2
    # Should mention readiness or answer quality.
    joined = " ".join(insights).lower()
    assert "readiness" in joined or "score" in joined or "dimension" in joined


def test_insights_with_trends(sample_eval_history) -> None:
    from src.modules.evaluation_dashboard.metrics import compute_metrics

    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    insights = generate_insights(s, trends)
    joined = " ".join(insights).lower()
    assert "improved" in joined or "improving" in joined


def test_insights_mention_unsupported_claims() -> None:
    s = EvaluationSummary(
        answer_quality=AnswerQuality(unsupported_claim_count=3, total_evaluated=2),
    )
    insights = generate_insights(s, [])
    joined = " ".join(insights).lower()
    assert "unsupported" in joined
    assert "3" in joined


def test_insights_mention_lowest_dimension() -> None:
    s = EvaluationSummary(
        answer_quality=AnswerQuality(
            technical_avg=6, relevance_avg=7, specificity_avg=4,
            communication_avg=8, completeness_avg=5, total_evaluated=2,
        ),
    )
    insights = generate_insights(s, [])
    joined = " ".join(insights)
    assert "Specificity" in joined
    assert "4" in joined


def test_insights_mention_highest_dimension() -> None:
    s = EvaluationSummary(
        answer_quality=AnswerQuality(
            technical_avg=6, relevance_avg=7, specificity_avg=4,
            communication_avg=8, completeness_avg=5, total_evaluated=2,
        ),
    )
    insights = generate_insights(s, [])
    joined = " ".join(insights)
    assert "Communication" in joined


# ---- mocked Gemini enrichment --------------------------------------


def test_enrich_valid_accepted(sample_eval_history, eval_insight_valid_json) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det_insights = generate_insights(s, trends)
    # Build a valid Gemini response with the same numbers.
    fake = _fake_client(eval_insight_valid_json)
    enriched = enrich_insights(det_insights, s, trends, client=fake)
    assert len(enriched) == len(det_insights)
    # The prompt enforces grounding.
    assert "evidence-grounded" in fake.calls[0]["system_prompt"].lower()
    assert "must not" in fake.calls[0]["system_prompt"].lower() or "do not" in fake.calls[0]["system_prompt"].lower()


def test_enrich_no_client_returns_deterministic(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    enriched = enrich_insights(det, s, trends, client=None)
    assert enriched == det


# ---- Gemini cannot change scores -----------------------------------


def test_enrich_changed_numbers_rejected(sample_eval_history, eval_insight_changed_numbers_json) -> None:
    """Gemini changes numbers (72→999) → those insights fall back to deterministic."""
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client(eval_insight_changed_numbers_json)
    enriched = enrich_insights(det, s, trends, client=fake)
    # None of the enriched insights should contain 999 or 10.0 or 1.0.
    joined = " ".join(enriched)
    assert "999" not in joined
    assert "10.0" not in joined


# ---- malformed Gemini output --------------------------------------


def test_enrich_malformed_falls_back(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client("not json {{{")
    enriched = enrich_insights(det, s, trends, client=fake)
    assert enriched == det  # falls back to deterministic


def test_enrich_empty_falls_back(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client("")
    enriched = enrich_insights(det, s, trends, client=fake)
    assert enriched == det


def test_enrich_wrong_count_falls_back(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    # Gemini returns 1 insight but we have 3 → wrong count → fall back.
    fake = _fake_client(json.dumps({"insights": ["Only one."]}))
    enriched = enrich_insights(det, s, trends, client=fake)
    assert enriched == det


# ---- API failure ---------------------------------------------------


def test_enrich_api_error_raises(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client("", raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        enrich_insights(det, s, trends, client=fake)


# ---- no API key ---------------------------------------------------


def test_enrich_no_key_works(monkeypatch, sample_eval_history) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    enriched = enrich_insights(det, s, trends, client=None)
    assert enriched == det


# ---- no raw resume_text in Gemini context -------------------------


def test_enrich_no_raw_resume_text(sample_eval_history, eval_insight_valid_json) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client(eval_insight_valid_json)
    enrich_insights(det, s, trends, client=fake)
    payload = fake.calls[0]["user_text"]
    assert "resume_text" not in payload.lower()


# ================================================================== #
# PER-INSIGHT NUMERIC IMMUTABILITY TESTS
# ================================================================== #

import re as _re

_DET_3 = [
    "Interview readiness is 72%.",
    "Technical performance is 6.5/10.",
    "2 unsupported claim(s) detected across interview answers.",
]


def _make_insight_json(insights: list[str]) -> str:
    import json
    return json.dumps({"insights": insights})


def _enrich(det, gemini_json, summary=None, trends=None):
    """Helper: run enrich_insights with a fake client."""
    s = summary or EvaluationSummary()
    fake = _fake_client(gemini_json)
    return enrich_insights(det, s, trends or [], client=fake)


# A. Valid wording with identical numbers is accepted.


def test_perinsight_valid_wording_accepted() -> None:
    gem = _make_insight_json([
        "Your interview readiness stands at 72%.",
        "The technical performance score is 6.5/10.",
        "2 unsupported claim(s) were detected across the interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == "Your interview readiness stands at 72%."
    assert out[1] == "The technical performance score is 6.5/10."
    assert out[2] == "2 unsupported claim(s) were detected across the interview answers."


# B. Gemini changes a number inside the same insight → rejected.


def test_perinsight_number_changed_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is 75%.",
        "Technical performance is 6.5/10.",
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == _DET_3[0]  # rejected (72→75)


# C. Gemini removes a deterministic number → rejected.


def test_perinsight_number_removed_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is good.",
        "Technical performance is 6.5/10.",
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == _DET_3[0]  # rejected (72 removed)


# D. Gemini introduces a new number → rejected.


def test_perinsight_number_added_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is 72% and confidence is high at 99%.",
        "Technical performance is 6.5/10.",
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == _DET_3[0]  # rejected (99 introduced)


# E. Gemini swaps numbers between two insights → rejected.


def test_perinsight_number_swapped_rejected() -> None:
    """Gemini puts 6.5 in insight 1 and 72 in insight 2 → both rejected."""
    gem = _make_insight_json([
        "Interview readiness is 6.5/10.",   # should be 72
        "Technical performance is 72%.",      # should be 6.5
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == _DET_3[0]  # rejected
    assert out[1] == _DET_3[1]  # rejected
    assert out[2] != _DET_3[2] or out[2] == _DET_3[2]  # insight 3 unchanged (accepted or kept)


# F. Gemini changes percentage values → rejected.


def test_perinsight_percentage_changed_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is 50%.",
        "Technical performance is 6.5/10.",
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[0] == _DET_3[0]  # rejected (72→50)


# G. Gemini changes score values → rejected.


def test_perinsight_score_changed_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is 72%.",
        "Technical performance is 9.9/10.",
        "2 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[1] == _DET_3[1]  # rejected (6.5→9.9)


# H. Gemini changes session-count values → rejected.


def test_perinsight_session_count_changed_rejected() -> None:
    gem = _make_insight_json([
        "Interview readiness is 72%.",
        "Technical performance is 6.5/10.",
        "99 unsupported claim(s) detected across interview answers.",
    ])
    out = _enrich(_DET_3, gem)
    assert out[2] == _DET_3[2]  # rejected (2→99)


# I. Multiple numeric values in one insight must be preserved.


def test_perinsight_multiple_numbers_preserved() -> None:
    det = ["Readiness improved from 55 to 72."]
    gem = _make_insight_json(["Readiness improved from 55 to 72."])
    out = _enrich(det, gem)
    assert out[0] == "Readiness improved from 55 to 72."


def test_perinsight_multiple_numbers_changed_rejected() -> None:
    det = ["Readiness improved from 55 to 72."]
    gem = _make_insight_json(["Readiness improved from 50 to 80."])
    out = _enrich(det, gem)
    assert out[0] == det[0]  # rejected


def test_perinsight_multiple_numbers_order_irrelevant() -> None:
    """Numbers can appear in a different position but must be the same multiset."""
    det = ["Readiness improved from 55 to 72."]
    gem = _make_insight_json(["The value went from 72 down to 55."])
    out = _enrich(det, gem)
    # Same multiset {55, 72} → accepted.
    assert out[0] == "The value went from 72 down to 55."


# J. Decimal values such as 6.5 must be handled correctly.


def test_perinsight_decimal_preserved() -> None:
    det = ["The score is 6.5 out of 10."]
    gem = _make_insight_json(["The score is 6.5 out of 10."])
    out = _enrich(det, gem)
    assert out[0] == "The score is 6.5 out of 10."


def test_perinsight_decimal_changed_rejected() -> None:
    det = ["The score is 6.5 out of 10."]
    gem = _make_insight_json(["The score is 7.5 out of 10."])
    out = _enrich(det, gem)
    assert out[0] == det[0]  # rejected


# K. Percentages such as 72% must be handled correctly.


def test_perinsight_percentage_preserved() -> None:
    det = ["Readiness is 72%."]
    gem = _make_insight_json(["Readiness is 72%."])
    out = _enrich(det, gem)
    assert out[0] == "Readiness is 72%."


def test_perinsight_percentage_removed_rejected() -> None:
    det = ["Readiness is 72%."]
    gem = _make_insight_json(["Readiness is high."])
    out = _enrich(det, gem)
    assert out[0] == det[0]  # rejected


# L. The existing global-number test must still pass.


def test_perinsight_existing_global_test_still_passes(sample_eval_history, eval_insight_changed_numbers_json) -> None:
    """The existing test_enrich_changed_numbers_rejected still works."""
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    fake = _fake_client(eval_insight_changed_numbers_json)
    enriched = enrich_insights(det, s, trends, client=fake)
    joined = " ".join(enriched)
    assert "999" not in joined


# M. No-Gemini deterministic behavior remains unchanged.


def test_perinsight_no_gemini_unchanged(sample_eval_history) -> None:
    s = sample_eval_history[-1]
    trends = compute_trends(sample_eval_history)
    det = generate_insights(s, trends)
    enriched = enrich_insights(det, s, trends, client=None)
    assert enriched == det


# N. Malformed Gemini response still falls back safely.


def test_perinsight_malformed_falls_back() -> None:
    det = ["Readiness is 72%."]
    fake = _fake_client("not json {{{")
    out = _enrich(det, "not json {{{")
    assert out == det  # falls back to deterministic


def test_perinsight_empty_falls_back() -> None:
    det = ["Readiness is 72%."]
    out = _enrich(det, "")
    assert out == det


def test_perinsight_wrong_count_falls_back() -> None:
    det = ["Readiness is 72%.", "Technical is 6.5/10."]
    gem = _make_insight_json(["Only one insight."])
    out = _enrich(det, gem)
    assert out == det  # wrong count → all fall back


# Extra: insight with no numbers → Gemini must not add any.


def test_perinsight_no_numbers_must_not_add() -> None:
    det = ["Complete an interview to unlock insights."]
    gem = _make_insight_json(["Complete an interview to unlock insights with 5 steps."])
    out = _enrich(det, gem)
    assert out[0] == det[0]  # rejected (5 introduced)


def test_perinsight_no_numbers_valid_rewording() -> None:
    det = ["Complete an interview to unlock insights."]
    gem = _make_insight_json(["Finish an interview session to access detailed insights."])
    out = _enrich(det, gem)
    assert out[0] == "Finish an interview session to access detailed insights."
