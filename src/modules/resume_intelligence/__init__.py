"""Resume Intelligence module.

Public API: :func:`render` draws the upload -> extract -> analyze ->
results workflow. Internally delegates to
:mod:`src.modules.resume_intelligence.ui`.

Submodules
----------
- ``model``      : :class:`ResumeAnalysis` schema + robust parsing.
- ``extractor``  : PDF / DOCX / TXT text extraction.
- ``prompt``      : Gemini system prompt + JSON response schema.
- ``analyzer``   : :func:`analyze_resume` service.
- ``ui``         : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.resume_intelligence.ui import render

MODULE_NAME: str = "Resume Intelligence"
MODULE_KEY: str = "resume_intelligence"
MODULE_DESCRIPTION: str = (
    "Parse, analyze, and optimize resumes for ATS compatibility and "
    "recruiter readability."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
