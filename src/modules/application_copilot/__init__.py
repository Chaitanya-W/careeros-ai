"""Application Copilot module.

Public API: :func:`render` draws the evidence-grounded Resume Tailor /
Cover Letter / Bullet Optimizer workflow. Internally delegates to
:mod:`src.modules.application_copilot.ui`.

Submodules
----------
- ``model``      : ValidationStatus, EvidenceItem, EvidenceValidation, ApplicationDraft,
                  TailoredResumeSuggestion, CoverLetterDraft, BulletSuggestion.
- ``evidence``   : :func:`extract_evidence` deterministic evidence layer (stable IDs).
- ``validator``  : :func:`validate_claims` conservative evidence validator (PASS/WARNING/FAIL).
- ``prompt``     : Gemini system prompts + JSON response schemas (with grounding rules).
- ``copilot``    : :func:`tailor_resume` / :func:`generate_cover_letter` / :func:`optimize_bullet`.
- ``ui``         : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.application_copilot.ui import render

MODULE_NAME: str = "Application Copilot"
MODULE_KEY: str = "application_copilot"
MODULE_DESCRIPTION: str = (
    "Draft tailored cover letters, outreach messages, and application "
    "materials."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
