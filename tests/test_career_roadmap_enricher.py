"""Tests for the Career Roadmap Gemini enrichment (MOCKED — no API key).

Covers: happy-path enrichment, invalid response, Gemini errors, and the
critical guarantees that Gemini cannot ADD / REMOVE / REORDER skills.
"""

from __future__ import annotations

import json

import pytest

from src.modules.career_roadmap.builder import build_career_roadmap
from src.modules.career_roadmap.enricher import (
    RoadmapEnrichmentError,
    apply_enrichment,
    enrich_roadmap,
)
from src.modules.career_roadmap.model import CareerRoadmap
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


# ---- (18) mocked Gemini enrichment (happy path) -----------------------


def test_enrich_success(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_roadmap_enrichment_json)
    enrichment = enrich_roadmap(sample_career_roadmap, client=fake)
    assert "phases" in enrichment
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert "do NOT add" in call["system_prompt"].lower() or "do not add" in call["system_prompt"].lower()
    # The context lists the roadmap skills.
    assert "Docker" in call["user_text"]


def test_apply_enrichment_updates_wording_not_skills(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_roadmap_enrichment_json)
    enrichment = enrich_roadmap(sample_career_roadmap, client=fake)
    apply_enrichment(sample_career_roadmap, enrichment)

    assert sample_career_roadmap.enriched is True
    # Readiness unchanged (deterministic).
    assert sample_career_roadmap.current_readiness == sample_career_roadmap.current_readiness
    # Wording refined.
    assert "Refined" in sample_career_roadmap.phases[0].objective
    assert sample_career_roadmap.estimated_total_effort == "8-16 weeks of focused study."


def test_enrich_empty_roadmap_returns_empty_no_call(
    sample_skill_gap_inputs, fake_gemini_client_factory
) -> None:
    resume, job, match = sample_skill_gap_inputs
    # Build an empty-gaps roadmap.
    from src.modules.skill_gap.model import SkillGapReport

    empty_report = SkillGapReport(target_role="Eng", skill_gaps=[])
    roadmap = build_career_roadmap(empty_report, match, job)
    fake = fake_gemini_client_factory(response="{}")
    out = enrich_roadmap(roadmap, client=fake)
    assert out == {}
    assert fake.calls == []  # Gemini never invoked


# ---- (20) Gemini cannot ADD skills ------------------------------------


def test_enrich_rejects_added_skill(
    sample_career_roadmap,
    roadmap_skill_adding_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=roadmap_skill_adding_json)
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_apply_enrichment_does_not_add_skills_to_roadmap(
    sample_career_roadmap,
    roadmap_skill_adding_json: str,
    fake_gemini_client_factory,
) -> None:
    """Even if validation is bypassed, apply_enrichment never adds skills."""
    fake = fake_gemini_client_factory(response=roadmap_skill_adding_json)
    # enrich_roadmap would raise; call apply_enrichment directly with a crafted
    # payload that tries to add a phase with InventedSkill.
    crafted = json.loads(roadmap_skill_adding_json)
    apply_enrichment(sample_career_roadmap, crafted)
    # No new skill appeared in the roadmap.
    assert "InventedSkill" not in sample_career_roadmap.all_skills


# ---- (21) Gemini cannot REMOVE skills --------------------------------


def test_enrich_rejects_removed_skill(
    sample_career_roadmap,
    roadmap_skill_removing_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=roadmap_skill_removing_json)
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_rejects_reordered_skills(
    sample_career_roadmap,
    roadmap_skill_reordering_json: str,
    fake_gemini_client_factory,
) -> None:
    """Reordering skills within a phase must also be rejected."""
    fake = fake_gemini_client_factory(response=roadmap_skill_reordering_json)
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_rejects_phase_count_mismatch(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    data = json.loads(sample_roadmap_enrichment_json)
    data["phases"] = data["phases"][:1]  # drop a phase
    fake = fake_gemini_client_factory(response=json.dumps(data))
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_rejects_unknown_phase_number(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    data = json.loads(sample_roadmap_enrichment_json)
    data["phases"][0]["phase_number"] = 99
    fake = fake_gemini_client_factory(response=json.dumps(data))
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


# ---- (19) invalid Gemini response ------------------------------------


def test_enrich_invalid_json_raises(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(response="not json {{{")
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_empty_response_raises(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(response="")
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_non_object_raises(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(response=json.dumps(["not", "object"]))
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_missing_phases_field_raises(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(response=json.dumps({"other": 1}))
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_phase_skills_not_list_raises(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    data = json.loads(sample_roadmap_enrichment_json)
    data["phases"][0]["skills"] = "Docker"  # not a list
    fake = fake_gemini_client_factory(response=json.dumps(data))
    with pytest.raises(RoadmapEnrichmentError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_code_fenced_json_is_parsed(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    fenced = "```json\n" + sample_roadmap_enrichment_json + "\n```"
    fake = fake_gemini_client_factory(response=fenced)
    enrichment = enrich_roadmap(sample_career_roadmap, client=fake)
    assert "phases" in enrichment


# ---- (22) Gemini API error / missing key -----------------------------


def test_enrich_propagates_api_error(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_propagates_config_error(sample_career_roadmap, fake_gemini_client_factory) -> None:
    fake = fake_gemini_client_factory(raise_exc=GeminiConfigError("no key"))
    with pytest.raises(GeminiError):
        enrich_roadmap(sample_career_roadmap, client=fake)


def test_enrich_default_client_without_key_raises_config(monkeypatch, sample_career_roadmap) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError):
        enrich_roadmap(sample_career_roadmap)


# ---- (23) session-state-style storage (round-trip) -------------------


def test_enriched_roadmap_round_trips_via_dict(
    sample_career_roadmap,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    fake = fake_gemini_client_factory(response=sample_roadmap_enrichment_json)
    enrichment = enrich_roadmap(sample_career_roadmap, client=fake)
    apply_enrichment(sample_career_roadmap, enrichment)

    stored = sample_career_roadmap.to_dict()
    restored = CareerRoadmap.from_dict(stored)
    assert restored.enriched is True
    assert restored.phase_count == sample_career_roadmap.phase_count
    assert restored.all_skills == sample_career_roadmap.all_skills
    assert "Refined" in restored.phases[0].objective


# ---- enrichment context does not leak resume text -------------------


def test_enrich_context_does_not_leak_resume_text(
    sample_career_roadmap,
    sample_skill_gap_inputs,
    sample_roadmap_enrichment_json: str,
    fake_gemini_client_factory,
) -> None:
    resume, _job, _match = sample_skill_gap_inputs
    fake = fake_gemini_client_factory(response=sample_roadmap_enrichment_json)
    enrich_roadmap(sample_career_roadmap, job=None, resume=resume, client=fake)
    payload = fake.calls[0]["user_text"]
    # Resume section text never echoed.
    assert "Built Python services" not in payload
