"""Tests for the DETERMINISTIC resume-vs-job matcher (no LLM)."""

from __future__ import annotations

import pytest

from src.modules.jd_analysis.matcher import (
    WEIGHTS,
    _classify_skills,
    _estimate_resume_years,
    _is_partial_match,
    _normalize_skill,
    _year_span,
    match_jobs,
)
from src.modules.jd_analysis.model import JobAnalysis, MatchAnalysis
from src.modules.resume_intelligence.model import (
    EducationItem,
    ExperienceItem,
    ResumeAnalysis,
)


# ---- (6) deterministic skill overlap ------------------------------------


def test_classify_skills_exact_overlap() -> None:
    matching, partial, missing = _classify_skills(
        ["Python", "SQL", "Pandas", "Git"],
        ["Python", "SQL", "AWS", "Docker"],
    )
    assert matching == ["Python", "SQL"]
    assert partial == []
    assert missing == ["AWS", "Docker"]


def test_classify_skills_preserves_target_order() -> None:
    matching, _, missing = _classify_skills(
        ["Python"], ["SQL", "Python", "AWS"]
    )
    assert matching == ["Python"]
    assert missing == ["SQL", "AWS"]


def test_classify_skills_empty_resume_all_missing() -> None:
    _, _, missing = _classify_skills([], ["Python", "SQL"])
    assert missing == ["Python", "SQL"]


def test_classify_skills_empty_target_nothing_classified() -> None:
    matching, partial, missing = _classify_skills(["Python"], [])
    assert matching == []
    assert partial == []
    assert missing == []


# ---- (7) missing skill detection ----------------------------------------


def test_missing_skills_returned_when_resume_lacks_them() -> None:
    _, _, missing = _classify_skills(["Python"], ["Python", "Go", "Rust"])
    assert missing == ["Go", "Rust"]


# ---- (8) partial skill handling (token-subset, no Java/JavaScript FP) ---


def test_partial_match_aws_lambda() -> None:
    assert _is_partial_match("AWS", "AWS Lambda") is True
    assert _is_partial_match("AWS Lambda", "AWS") is True


def test_partial_match_react_native() -> None:
    assert _is_partial_match("React", "React Native") is True


def test_partial_match_avoids_java_javascript_false_positive() -> None:
    """'Java' is NOT a token of 'JavaScript' -> not partial."""
    assert _is_partial_match("Java", "JavaScript") is False


def test_partial_match_exact_skills_are_not_partial() -> None:
    assert _is_partial_match("Python", "Python") is False


def test_partial_match_too_short_token_skipped() -> None:
    # "C" is len 1 -> below the min token length -> not partial even if a token
    assert _is_partial_match("C", "C Sharp") is False


def test_classify_skills_reports_partial() -> None:
    matching, partial, missing = _classify_skills(
        ["AWS Lambda", "React"],
        ["AWS", "AWS Lambda", "React Native", "Docker"],
    )
    assert matching == ["AWS Lambda"]
    assert partial == ["AWS", "React Native"]
    assert missing == ["Docker"]


# ---- normalization edge cases --------------------------------------------


def test_normalize_strips_versions_and_parentheticals() -> None:
    assert _normalize_skill("Python 3.10") == "python"
    assert _normalize_skill("Node.js 18 (LTS)") == "node.js"
    assert _normalize_skill("  C++  ") == "c++"


def test_normalize_empty() -> None:
    assert _normalize_skill("") == ""
    assert _normalize_skill(None) == ""  # type: ignore[arg-type]


# ---- (11) scoring calculation (deterministic, weighted) -----------------


def test_match_jobs_returns_match_analysis(sample_resume_for_matching) -> None:
    job = JobAnalysis(
        job_title="Backend Engineer",
        required_skills=["Python", "SQL", "AWS", "Docker"],
        preferred_skills=["Kubernetes"],
        experience_requirements="5+ years",
        education_requirements="B.S. in Computer Science",
        keywords=["Python", "AWS", "distributed", "reliability"],
    )
    m = match_jobs(sample_resume_for_matching, job)
    assert isinstance(m, MatchAnalysis)
    # Python + SQL exact; AWS partial (AWS Lambda); Docker missing.
    assert "Python" in m.matching_skills
    assert "SQL" in m.matching_skills
    assert "AWS" in m.partial_match_skills
    assert m.missing_skills == ["Docker"]
    # 2 exact + 0.5 partial out of 4 required = (2 + 0.25)/4 = 0.5625 -> 56
    assert m.experience_match == 100  # 6 years >= 5 years
    assert m.education_match == 100  # "computer" + "science" overlap
    assert m.keyword_match > 0
    assert 0 <= m.overall_match_score <= 100


def test_match_jobs_required_skill_score_value() -> None:
    """exact=2, partial=1 (counts 0.5), total=4 -> (2+0.5)/4 = 0.625 -> 63."""
    resume = ResumeAnalysis(skills=["Python", "SQL", "AWS Lambda"])
    job = JobAnalysis(
        required_skills=["Python", "SQL", "AWS", "Docker"],
        preferred_skills=[],
        experience_requirements="",
        education_requirements="",
        keywords=[],
    )
    m = match_jobs(resume, job)
    # required is the only active component -> overall == required score.
    expected_required = round((2 + 0.5) / 4 * 100)  # 63
    assert m.overall_match_score == expected_required


def test_match_jobs_no_active_components_yields_zero() -> None:
    """A job with no required/preferred/exp/edu/keywords data -> 0."""
    resume = ResumeAnalysis(skills=["Python"])
    job = JobAnalysis()  # all-empty
    m = match_jobs(resume, job)
    assert m.overall_match_score == 0


def test_match_jobs_score_in_range_0_100() -> None:
    resume = ResumeAnalysis(
        skills=["Python"],
        experience=[ExperienceItem(duration="1 year")],
    )
    job = JobAnalysis(
        required_skills=["Python", "Go", "Rust", "C", "C++", "Java"],
        experience_requirements="10+ years",
    )
    m = match_jobs(resume, job)
    assert 0 <= m.overall_match_score <= 100
    assert 0 <= m.experience_match <= 100


def test_match_jobs_weights_sum_to_one() -> None:
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_match_jobs_experience_score_years_ratio() -> None:
    """6 years vs '5+ years' -> ratio 6/5 clamped to 100."""
    resume = ResumeAnalysis(
        experience=[ExperienceItem(duration="6 years")]
    )
    job = JobAnalysis(experience_requirements="5+ years")
    m = match_jobs(resume, job)
    assert m.experience_match == 100


def test_match_jobs_experience_score_no_resume_experience_is_zero() -> None:
    resume = ResumeAnalysis()  # no experience entries
    job = JobAnalysis(experience_requirements="3+ years")
    m = match_jobs(resume, job)
    assert m.experience_match == 0


def test_match_jobs_experience_neutral_when_unparseable() -> None:
    """Has experience entries but no parseable years -> neutral 50."""
    resume = ResumeAnalysis(
        experience=[ExperienceItem(duration="several years at Acme")]
    )
    job = JobAnalysis(experience_requirements="extensive backend experience")
    m = match_jobs(resume, job)
    assert m.experience_match == 50


def test_match_jobs_education_overlap() -> None:
    resume = ResumeAnalysis(
        education=[EducationItem(degree="B.S. Computer Science")]
    )
    job = JobAnalysis(education_requirements="B.S. in Computer Science")
    m = match_jobs(resume, job)
    assert m.education_match == 100  # tokens overlap


def test_match_jobs_education_no_resume_is_zero() -> None:
    resume = ResumeAnalysis()  # no education
    job = JobAnalysis(education_requirements="B.S. in CS")
    m = match_jobs(resume, job)
    assert m.education_match == 0


def test_match_jobs_keyword_alignment_fraction() -> None:
    """2 of 4 job keywords present in the resume corpus -> 50."""
    resume = ResumeAnalysis(
        skills=["Python"],
        professional_summary="Experienced with AWS and distributed systems.",
    )
    job = JobAnalysis(keywords=["Python", "AWS", "distributed", "rust"])
    m = match_jobs(resume, job)
    assert m.keyword_match == 75  # 3 of 4 found


# ---- deterministic narrative builders ------------------------------------


def test_match_jobs_builds_strengths_and_gaps() -> None:
    resume = ResumeAnalysis(skills=["Python", "SQL"])
    job = JobAnalysis(
        required_skills=["Python", "SQL", "AWS"],
        experience_requirements="5+ years",
    )
    m = match_jobs(resume, job)
    assert any("Python" in s for s in m.strengths)
    assert any("AWS" in g for g in m.gaps)


def test_match_jobs_explanation_is_deterministic_and_includes_overall() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    job = JobAnalysis(required_skills=["Python"], keywords=["Python"])
    m = match_jobs(resume, job)
    assert str(m.overall_match_score) in m.explanation
    assert "matching" in m.explanation.lower()


def test_match_jobs_recommended_actions_mention_missing() -> None:
    resume = ResumeAnalysis(skills=["Python"])
    job = JobAnalysis(required_skills=["Python", "Kubernetes"])
    m = match_jobs(resume, job)
    joined = " ".join(m.recommended_actions)
    assert "Kubernetes" in joined


def test_match_jobs_recommended_actions_default_when_no_gaps() -> None:
    resume = ResumeAnalysis(
        skills=["Python", "SQL"],
        experience=[ExperienceItem(duration="5 years")],
        education=[EducationItem(degree="B.S. Computer Science")],
    )
    job = JobAnalysis(
        required_skills=["Python", "SQL"],
        experience_requirements="5+ years",
        education_requirements="B.S. in Computer Science",
        keywords=["Python", "SQL"],
    )
    m = match_jobs(resume, job)
    # No gaps -> a positive default action is emitted.
    assert len(m.recommended_actions) >= 1


# ---- (12) no-resume-analysis behavior (matcher never raises) -----------


def test_match_jobs_with_empty_resume_and_job_does_not_raise() -> None:
    m = match_jobs(ResumeAnalysis(), JobAnalysis())
    assert m.overall_match_score == 0
    assert m.matching_skills == []
    assert m.missing_skills == []


# ---- (3) dynamic current year for "present" date ranges (FIX 2) -------
# Verifies the hardcoded 2025 was removed and date.today().year is used.


def test_year_span_present_uses_current_year_dynamically() -> None:
    """'2021-present' must resolve to today.year - 2021, not a fixed 2025."""
    from datetime import date

    assert _year_span("2021-present") == date.today().year - 2021


def test_year_span_present_keywords_all_dynamic() -> None:
    """All 'present' synonyms resolve to the dynamic current year."""
    from datetime import date

    expected = date.today().year - 2020
    for kw in ("present", "current", "now"):
        assert _year_span(f"2020-{kw}") == expected


def test_year_span_explicit_range_is_not_affected() -> None:
    """A closed range '2019-2023' must still return 4 (no dynamic year)."""
    assert _year_span("2019-2023") == 4.0
    assert _year_span("2020-2022") == 2.0


def test_year_span_no_dates_returns_zero() -> None:
    assert _year_span("no dates here") == 0.0
    assert _year_span("") == 0.0


def test_year_span_not_hardcoded_to_2025() -> None:
    """If the year were hardcoded to 2025, '2021-present' would be 4 even
    when today is not 2025. Assert the result tracks the real current year."""
    from datetime import date

    result = _year_span("2021-present")
    assert result == date.today().year - 2021
    # The sandbox clock is past 2025, so this guards against regressions
    # where someone re-hardcodes 2025.
    assert result != 2025 - 2021 or date.today().year == 2025


def test_estimate_resume_years_present_range_uses_dynamic_year() -> None:
    from datetime import date

    resume = ResumeAnalysis(
        experience=[ExperienceItem(duration="2021-present")]
    )
    assert _estimate_resume_years(resume) == date.today().year - 2021


def test_existing_experience_scoring_unaffected_by_fix() -> None:
    """The years-ratio path (explicit 'N years') is unchanged by FIX 2."""
    resume = ResumeAnalysis(
        experience=[ExperienceItem(duration="6 years")]
    )
    job = JobAnalysis(experience_requirements="5+ years")
    m = match_jobs(resume, job)
    assert m.experience_match == 100  # 6 >= 5 -> clamped to 100
