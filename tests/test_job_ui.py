"""Tests for the Job Match UI workflow (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked (no Analyze
click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.jd_analysis.matcher import match_jobs
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis


@pytest.fixture
def job_and_match(sample_resume_for_matching: ResumeAnalysis) -> tuple:
    """A (JobAnalysis, MatchAnalysis) pair for results-rendering tests."""
    job = JobAnalysis(
        job_title="Senior Backend Engineer",
        company="Acme Corp",
        required_skills=["Python", "SQL", "AWS", "Docker"],
        preferred_skills=["Kubernetes"],
        responsibilities=["Build distributed services."],
        experience_requirements="5+ years",
        education_requirements="B.S. in Computer Science",
        keywords=["Python", "AWS", "distributed", "reliability"],
        domain="Backend Engineering",
    )
    match = match_jobs(sample_resume_for_matching, job)
    return job, match


# ---- (16) Job Match page empty state — no resume analyzed --------------


def test_job_match_renders_no_resume_message(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.run()

    assert not at.exception, "render raised an exception"
    assert not at.error, "render reported an error"

    # The "analyze your resume first" message must be shown.
    assert any(
        "analyze your resume first" in m.value.lower()
        for m in at.markdown
    ), "no-resume guard message missing"

    # A navigation button to Resume Intelligence must be present.
    assert any(
        "Resume Intelligence" in b.label for b in at.button
    ), "navigation to Resume Intelligence missing"

    # The job-description text area must NOT be shown without a resume.
    assert len(at.text_area) == 0, "text area should not render without a resume"


def test_job_match_no_resume_does_not_call_gemini(app_path: str) -> None:
    """Without a resume, no Gemini-using widgets should even appear."""
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.run()
    # No Analyze button should be present (only the nav-to-resume button).
    assert all("Analyze" not in b.label for b in at.button)


# ---- (17) Job Match result rendering (pre-seeded session state) --------


def test_job_match_renders_jd_input_when_resume_present(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    at.run()

    assert not at.exception
    assert not at.error
    # JD text area + Analyze button must now be present.
    assert len(at.text_area) == 1
    assert any("Analyze Job Match" in b.label for b in at.button)
    # The no-resume message must NOT appear.
    assert not any(
        "analyze your resume first" in m.value.lower()
        for m in at.markdown
    )


def test_job_match_renders_results_after_analysis(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
    job_and_match,
) -> None:
    job, match = job_and_match
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    at.session_state["job_analysis"] = job
    at.session_state["match_analysis"] = match
    at.run()

    assert not at.exception, "results render raised an exception"
    assert not at.error, "results render reported an error"

    subheaders = [s.value for s in at.subheader]
    # Required result sections must render.
    for expected in (
        "Match Results",
        "Matching skills",
        "Missing skills",
        "Skill / experience gaps",
        "Explanation",
    ):
        assert expected in subheaders, f"missing section: {expected}"

    # Overall score must be visible somewhere.
    assert any("100" in m.value for m in at.markdown) or any(
        str(match.overall_match_score) in m.value for m in at.markdown
    ), "overall score not displayed"

    # Component metric values (experience/education/keyword) must render.
    metric_labels = [m.label for m in at.metric]
    assert "Experience match" in metric_labels
    assert "Education match" in metric_labels
    assert "Keyword alignment" in metric_labels

    # The analyzed role header (job title / company) must render.
    assert any("Acme Corp" in m.value for m in at.markdown), "job header missing"

    # The no-resume message must NOT appear.
    assert not any(
        "analyze your resume first" in m.value.lower()
        for m in at.markdown
    ), "no-resume guard should not render when a resume exists"


def test_job_match_shows_empty_state_when_no_match_yet(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
) -> None:
    """Resume present but no match yet -> 'No job match yet' empty state."""
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    # job_analysis / match_analysis NOT set.
    at.run()

    assert not at.exception
    assert any(
        "No job match yet" in m.value for m in at.markdown
    ), "match empty-state missing"


# ---- Unexpected internal exception -> generic safe message (FIX 1) -----
# A valid-length JD so the analyzer is actually invoked (not rejected up
# front by the input validation).
_VALID_JD_FOR_UI: str = (
    "Senior Backend Engineer at Acme Corp requiring Python, SQL, AWS, "
    "Docker, and distributed systems experience."
)

# Substrings that must NEVER reach the end user when an unexpected error
# occurs. The raw exception below is deliberately full of sensitive-looking
# internals to prove the catch-all does not leak them.
_SENSITIVE_SUBSTRINGS = (
    "internal SDK detail",
    "/secret/path",
    "sk-xxx",
    "config=prod",
    "RuntimeError",
    "Traceback",
)


def _raising_analyze(*args, **kwargs):
    raise RuntimeError(
        "internal SDK detail at /secret/path with key=sk-xxx and config=prod"
    )


def test_unexpected_exception_shows_generic_safe_message(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
    monkeypatch,
) -> None:
    """(1) Unexpected internal errors must show the generic friendly message."""
    monkeypatch.setattr(
        "src.modules.jd_analysis.ui.analyze_job_description",
        _raising_analyze,
    )

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    at.session_state["job_description_input"] = _VALID_JD_FOR_UI
    at.run()

    # Click the "Analyze Job Match" button.
    analyze_btn = next(
        b for b in at.button if "Analyze Job Match" in b.label
    )
    analyze_btn.click().run()

    assert not at.exception, "the catch-all should have swallowed the error"

    error_values = [e.value for e in at.error]
    assert error_values, "no st.error was shown"
    assert any(
        "Something went wrong" in v for v in error_values
    ), f"generic message not shown: {error_values}"


def test_unexpected_exception_does_not_leak_raw_details(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
    monkeypatch,
) -> None:
    """(2) No raw exception text / paths / config / stack traces reach the user."""
    monkeypatch.setattr(
        "src.modules.jd_analysis.ui.analyze_job_description",
        _raising_analyze,
    )

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    at.session_state["job_description_input"] = _VALID_JD_FOR_UI
    at.run()

    next(b for b in at.button if "Analyze Job Match" in b.label).click().run()

    assert not at.exception
    error_values = [e.value for e in at.error]
    assert error_values, "no st.error was shown"
    leaked = [
        s for s in _SENSITIVE_SUBSTRINGS
        if any(s in v for v in error_values)
    ]
    assert not leaked, f"sensitive substrings leaked to UI: {leaked}"
    # The ONLY error shown must be the generic message.
    assert all(
        "Something went wrong" in v for v in error_values
    ), f"unexpected error content: {error_values}"


def test_typed_errors_keep_their_specific_messages(
    app_path: str,
    sample_resume_for_matching: ResumeAnalysis,
    monkeypatch,
) -> None:
    """Typed errors (EmptyJobDescriptionError) keep their specific message;
    they are NOT replaced by the generic catch-all."""
    from src.modules.jd_analysis.analyzer import EmptyJobDescriptionError

    def _raise_empty(*args, **kwargs):
        raise EmptyJobDescriptionError("Please paste a job description before analyzing.")

    monkeypatch.setattr(
        "src.modules.jd_analysis.ui.analyze_job_description", _raise_empty
    )

    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "jd_analysis"
    at.session_state["resume_analysis"] = sample_resume_for_matching
    at.session_state["job_description_input"] = _VALID_JD_FOR_UI
    at.run()
    next(b for b in at.button if "Analyze Job Match" in b.label).click().run()

    assert not at.exception
    error_values = [e.value for e in at.error]
    # The specific typed message is shown (NOT the generic one).
    assert any(
        "paste a job description" in v for v in error_values
    ), f"typed message lost: {error_values}"
    assert not any("Something went wrong" in v for v in error_values)
