"""Tests for the Skill Gap Gemini enrichment (MOCKED — no API key)."""

from __future__ import annotations

import json

import pytest

from src.modules.skill_gap.enricher import (
    SkillGapEnrichmentError,
    apply_enrichment,
    enrich_skill_gaps,
)
from src.modules.skill_gap.model import Priority, SkillGap, SkillGapReport
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


# ---- (20) mocked Gemini enrichment (happy path) -------------------------


def test_enrich_success(
    sample_skill_gap_inputs,
    sample_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    resume, job, _match = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=sample_enrichment_json)
    gaps = [
        SkillGap(skill="Docker", priority=Priority.HIGH),
        SkillGap(skill="Kubernetes", priority=Priority.MEDIUM),
    ]
    enrichment = enrich_skill_gaps(gaps, job=job, resume=resume, client=fake)
    assert set(enrichment.keys()) == {"Docker", "Kubernetes"}
    assert "Containers" in enrichment["Docker"]["why_it_matters"]
    assert enrichment["Docker"]["learning_objectives"]
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert "do NOT add new skills" in call["system_prompt"].lower() or "do not add" in call["system_prompt"].lower()
    # The user_text must list the supplied skills.
    assert "Docker" in call["user_text"]
    assert "Kubernetes" in call["user_text"]


def test_enrich_empty_gaps_returns_empty_no_call(fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(response="{}")
    out = enrich_skill_gaps([], job=JobAnalysis(), resume=ResumeAnalysis(), client=fake)
    assert out == {}
    assert fake.calls == []  # Gemini never invoked


def test_enrich_context_does_not_leak_full_resume(
    sample_skill_gap_inputs,
    sample_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    """The context sent to Gemini must not include the full resume text."""
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=sample_enrichment_json)
    gaps = [SkillGap(skill="Docker")]
    enrich_skill_gaps(gaps, job=job, resume=resume, client=fake)
    payload = fake.calls[0]["user_text"]
    # Resume sections are never echoed wholesale.
    assert "Built Python services" not in payload


# ---- (27) Gemini enrichment CANNOT add new skills -----------------------


def test_enrich_rejects_ai_added_skills(
    sample_skill_gap_inputs,
    skill_adding_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    """An AI-invented skill must be rejected — never appears in output."""
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=skill_adding_enrichment_json)
    gaps = [
        SkillGap(skill="Docker", priority=Priority.HIGH),
        SkillGap(skill="Kubernetes", priority=Priority.MEDIUM),
    ]
    enrichment = enrich_skill_gaps(gaps, job=job, resume=resume, client=fake)
    # Docker is supplied -> kept; InventedSkill is NOT -> rejected.
    assert "Docker" in enrichment
    assert "InventedSkill" not in enrichment
    assert "InventedSkill" not in json.dumps(enrichment)


def test_apply_enrichment_never_adds_skills_to_report(
    sample_skill_gap_inputs,
    skill_adding_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    """apply_enrichment must not increase the report's gap count."""
    from src.modules.skill_gap.report import build_skill_gap_report

    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    gap_count_before = len(report.skill_gaps)
    names_before = {g.skill for g in report.skill_gaps}

    fake = fake_gemini_client_factory(response=skill_adding_enrichment_json)
    enrichment = enrich_skill_gaps(
        report.skill_gaps, job=job, resume=resume, client=fake
    )
    apply_enrichment(report, enrichment)

    assert len(report.skill_gaps) == gap_count_before  # no new skills added
    assert {g.skill for g in report.skill_gaps} == names_before
    assert report.enriched is True  # Docker was enriched
    # InventedSkill never entered the report.
    docker = next(g for g in report.skill_gaps if g.skill == "Docker")
    assert docker.why_it_matters == "Containers matter."


# ---- (21) invalid Gemini response --------------------------------------


def test_enrich_invalid_json_raises(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response="not json {{{")
    with pytest.raises(SkillGapEnrichmentError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


def test_enrich_empty_response_raises(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response="")
    with pytest.raises(SkillGapEnrichmentError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


def test_enrich_non_object_json_raises(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=json.dumps(["not", "an", "object"]))
    with pytest.raises(SkillGapEnrichmentError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


def test_enrich_missing_skills_field_raises(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=json.dumps({"other": 1}))
    with pytest.raises(SkillGapEnrichmentError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


def test_enrich_code_fenced_json_is_parsed(
    sample_skill_gap_inputs,
    sample_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fenced = "```json\n" + sample_enrichment_json + "\n```"
    fake = fake_gemini_client_factory(response=fenced)
    enrichment = enrich_skill_gaps(
        [SkillGap(skill="Docker"), SkillGap(skill="Kubernetes")],
        job=job, resume=resume, client=fake,
    )
    assert "Docker" in enrichment


# ---- (22) Gemini API error ---------------------------------------------


def test_enrich_propagates_api_error(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


def test_enrich_propagates_config_error(sample_skill_gap_inputs, fake_gemini_client_factory) -> None:
    resume, job, _ = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(raise_exc=GeminiConfigError("no key"))
    with pytest.raises(GeminiError):
        enrich_skill_gaps([SkillGap(skill="Docker")], job=job, resume=resume, client=fake)


# ---- (23) missing Gemini API key (default client) ---------------------


def test_enrich_default_client_without_key_raises_config(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError):
        enrich_skill_gaps(
            [SkillGap(skill="Docker")],
            job=JobAnalysis(),
            resume=ResumeAnalysis(),
        )


# ---- (24) session-state-style storage (report round-trips) -------------


def test_enriched_report_round_trips_via_dict(
    sample_skill_gap_inputs,
    sample_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    from src.modules.skill_gap.report import build_skill_gap_report

    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    fake = fake_gemini_client_factory(response=sample_enrichment_json)
    enrichment = enrich_skill_gaps(
        report.skill_gaps, job=job, resume=resume, client=fake
    )
    apply_enrichment(report, enrichment)

    # Simulate session-state storage: round-trip via to_dict/from_dict.
    stored = report.to_dict()
    restored = SkillGapReport.from_dict(stored)
    assert restored.enriched is True
    assert restored.total_gaps == report.total_gaps
    docker = next(g for g in restored.skill_gaps if g.skill == "Docker")
    assert docker.why_it_matters  # enrichment preserved


# ---- partial enrichment (model omits a supplied skill) -----------------


def test_enrich_partial_omitted_skill_stays_default(
    sample_skill_gap_inputs,
    fake_gemini_client_factory,
) -> None:
    """If Gemini omits a supplied skill, that gap's enrichment stays empty."""
    from src.modules.skill_gap.report import build_skill_gap_report

    resume, job, match = sample_skill_gap_inputs
    report = build_skill_gap_report(resume, job, match)
    # Only enrich Docker, omit everything else.
    partial = json.dumps(
        {
            "skills": [
                {
                    "skill": "Docker",
                    "why_it_matters": "Containers matter.",
                    "learning_objectives": [],
                    "learning_path": [],
                    "practice_project": "",
                    "estimated_effort": "",
                }
            ]
        }
    )
    fake = fake_gemini_client_factory(response=partial)
    enrichment = enrich_skill_gaps(
        report.skill_gaps, job=job, resume=resume, client=fake
    )
    assert enrichment == {"Docker": {
        "why_it_matters": "Containers matter.",
        "learning_objectives": [],
        "learning_path": [],
        "practice_project": "",
        "estimated_effort": "",
    }}
    apply_enrichment(report, enrichment)
    docker = next(g for g in report.skill_gaps if g.skill == "Docker")
    assert docker.why_it_matters == "Containers matter."
    kubernetes = next(g for g in report.skill_gaps if g.skill == "Kubernetes")
    assert kubernetes.why_it_matters == ""  # untouched


# local imports for type clarity
from src.modules.jd_analysis.model import JobAnalysis  # noqa: E402
from src.modules.resume_intelligence.model import ResumeAnalysis  # noqa: E402
