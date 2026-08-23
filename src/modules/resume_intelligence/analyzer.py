"""Resume analysis service — bridge between the UI and Gemini.

The UI calls :func:`analyze_resume` with extracted resume text and gets
back a validated :class:`ResumeAnalysis`. All Gemini specifics live here
(behind :class:`GeminiClient`); the UI never imports the Gemini SDK.

Errors are typed so the UI can show friendly messages instead of
tracebacks:

- :class:`EmptyResumeError` — text is empty / too short.
- :class:`GeminiError` (+ subclasses) — missing key, SDK, or API failure.
- :class:`ResumeAnalysisError` — model response could not be parsed.
"""

from __future__ import annotations

from typing import Optional

from src.modules.resume_intelligence.model import (
    ResumeAnalysis,
    ResumeAnalysisError,
)
from src.modules.resume_intelligence.prompt import (
    RESPONSE_SCHEMA,
    RESUME_ANALYSIS_SYSTEM_PROMPT,
)
from src.services.gemini import GeminiClient, GeminiError

# Minimum resume text length (chars) before we attempt analysis.
_MIN_RESUME_LENGTH: int = 20

__all__ = [
    "EmptyResumeError",
    "analyze_resume",
    "RESUME_ANALYSIS_SYSTEM_PROMPT",
    "RESPONSE_SCHEMA",
]


class EmptyResumeError(Exception):
    """Raised when the supplied resume text is empty or too short."""


def analyze_resume(
    text: str,
    *,
    client: Optional[GeminiClient] = None,
) -> ResumeAnalysis:
    """Analyze resume ``text`` and return a structured :class:`ResumeAnalysis`.

    Args:
        text: plain-text resume content (from the extractor).
        client: optional :class:`GeminiClient` (injected in tests; default
            creates one reading the key from secrets/env).

    Returns:
        A validated :class:`ResumeAnalysis`.

    Raises:
        EmptyResumeError: text is None / empty / too short.
        GeminiError: missing key, SDK problem, or API failure.
        ResumeAnalysisError: model response could not be parsed/validated.
    """
    if text is None:
        raise EmptyResumeError("Resume text is missing.")
    cleaned = text.strip()
    if len(cleaned) < _MIN_RESUME_LENGTH:
        raise EmptyResumeError(
            "Resume text is too short to analyze — "
            "please upload a complete resume."
        )

    gemini = client or GeminiClient()
    raw_json = gemini.generate_json(
        system_prompt=RESUME_ANALYSIS_SYSTEM_PROMPT,
        user_text=cleaned,
        response_schema=RESPONSE_SCHEMA,
    )
    return ResumeAnalysis.from_json(raw_json)
