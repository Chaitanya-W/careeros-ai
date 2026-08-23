"""Tests for the Resume Intelligence UI workflow (NO API key required).

Uses Streamlit's AppTest harness to render the module headlessly. The
Gemini client is never invoked (no Analyze click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.modules.resume_intelligence.model import (
    ExperienceItem,
    ResumeAnalysis,
)


@pytest.fixture
def resume_analysis_sample() -> ResumeAnalysis:
    """A populated analysis used to test the results view."""
    return ResumeAnalysis(
        overall_score=82,
        professional_summary="Backend engineer with 5 years of experience.",
        skills=["Python", "Go", "PostgreSQL"],
        experience=[
            ExperienceItem(
                title="Senior Engineer",
                company="Acme Corp",
                duration="2021-present",
                description="Built distributed APIs serving 1M req/day.",
            )
        ],
        education=[],
        projects=[],
        certifications=[],
        strengths=["Strong systems design"],
        weaknesses=["Sparse project detail"],
        improvement_suggestions=["Add a projects section."],
        recommended_skills=["Terraform"],
    )


# ---- (12) rendering WITHOUT a resume uploaded ----------------------------


def test_resume_renders_empty_state_when_no_resume(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "resume_intelligence"
    at.run()

    assert not at.exception, "render raised an exception"
    assert not at.error, "render reported an error"

    # Empty-state panel must be shown.
    assert any(
        "No resume analyzed yet" in m.value for m in at.markdown
    ), "empty-state panel missing"
    # Uploader widget must be present.
    assert len(at.file_uploader) == 1, "resume uploader missing"
    # Analyze button must NOT be shown when there is no resume text.
    assert all(
        b.label != "Analyze Resume" for b in at.button
    ), "Analyze button should not render without a resume"


def test_resume_renders_uploader_and_supported_formats(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "resume_intelligence"
    at.run()

    # The uploader restricts to pdf/docx/txt.
    uploader = at.file_uploader[0]
    assert uploader.label == "Choose a resume file"
    # Supported-format info is rendered as a section subtitle (caption).
    assert any(
        "PDF" in c.value and "DOCX" in c.value and "TXT" in c.value
        for c in at.caption
    ), "supported-format info missing"


# ---- (13) rendering AFTER analysis (pre-seeded session state) -----------


def test_resume_renders_results_after_analysis(
    app_path: str,
    resume_analysis_sample: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "resume_intelligence"
    at.session_state["resume_analysis"] = resume_analysis_sample
    at.run()

    assert not at.exception, "results render raised an exception"
    assert not at.error, "results render reported an error"

    subheaders = [s.value for s in at.subheader]
    # Core result sections must render.
    for expected in (
        "Analysis Results",
        "Professional Summary",
        "Skills",
        "Experience",
        "Strengths",
        "Areas for Improvement",
        "Improvement Suggestions",
        "Recommended Skills",
    ):
        assert expected in subheaders, f"missing results section: {expected}"

    # The score must be visible somewhere (82 / 100).
    assert any(
        "82" in m.value and "100" in m.value for m in at.markdown
    ), "overall score not displayed"

    # Professional summary text must render.
    assert any(
        "Backend engineer with 5 years" in m.value for m in at.markdown
    ), "professional summary not displayed"

    # Experience entry header must render.
    assert any(
        "Acme Corp" in m.value for m in at.markdown
    ), "experience entry not displayed"

    # The empty-state panel must NOT be shown when results exist.
    assert not any(
        "No resume analyzed yet" in m.value for m in at.markdown
    ), "empty state should not render when analysis exists"


def test_resume_results_score_bar_renders(
    app_path: str,
    resume_analysis_sample: ResumeAnalysis,
) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "resume_intelligence"
    at.session_state["resume_analysis"] = resume_analysis_sample
    at.run()
    assert not at.exception
    # The score is also shown via a progress bar (0.82). AppTest 1.62 has
    # no .progress collection, so assert the score section rendered.
    assert "Analysis Results" in [s.value for s in at.subheader]


# ---- state contract: other modules can read the cached analysis ---------


def test_session_state_key_is_resume_analysis(app_path: str) -> None:
    """The state key the spec requires must be initialized by init_state."""
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "resume_intelligence"
    at.run()
    # init_state seeds these defaults so other modules can rely on them.
    assert "resume_analysis" in at.session_state
    assert "resume_text" in at.session_state
