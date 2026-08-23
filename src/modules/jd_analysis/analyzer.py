"""Job-description analysis service — bridge between the UI and Gemini.

The UI calls :func:`analyze_job_description` with pasted job-description
text and gets back a validated :class:`JobAnalysis`. This reuses the
existing :class:`GeminiClient` (Step 3) — there is no second Gemini
implementation.

Typed errors so the UI shows friendly messages instead of tracebacks:

- :class:`EmptyJobDescriptionError` — text is empty / too short / too long.
- :class:`GeminiError` (+ subclasses) — missing key, SDK, or API failure.
- :class:`JobAnalysisError` — model response could not be parsed.
"""

from __future__ import annotations

from typing import Optional

from src.modules.jd_analysis.model import (
    JobAnalysis,
    JobAnalysisError,
)
from src.modules.jd_analysis.prompt import (
    JOB_ANALYSIS_SYSTEM_PROMPT,
    JOB_RESPONSE_SCHEMA,
)
from src.services.gemini import GeminiClient, GeminiError

# Input length bounds (characters). Tunable, not scientifically derived.
MIN_JD_LENGTH: int = 50
MAX_JD_LENGTH: int = 20_000

__all__ = [
    "EmptyJobDescriptionError",
    "analyze_job_description",
    "MIN_JD_LENGTH",
    "MAX_JD_LENGTH",
]


class EmptyJobDescriptionError(Exception):
    """Raised when the job description is empty / too short / too long."""


def analyze_job_description(
    text: str,
    *,
    client: Optional[GeminiClient] = None,
) -> JobAnalysis:
    """Analyze a job description and return a structured :class:`JobAnalysis`.

    Args:
        text: pasted job-description text.
        client: optional :class:`GeminiClient` (injected in tests).

    Raises:
        EmptyJobDescriptionError: text is None / empty / too short / too long.
        GeminiError: missing key, SDK problem, or API failure.
        JobAnalysisError: model response could not be parsed.
    """
    if text is None:
        raise EmptyJobDescriptionError("Job description is missing.")
    cleaned = text.strip()
    if not cleaned:
        raise EmptyJobDescriptionError(
            "Please paste a job description before analyzing."
        )
    if len(cleaned) < MIN_JD_LENGTH:
        raise EmptyJobDescriptionError(
            f"Job description is too short to analyze meaningfully "
            f"(minimum {MIN_JD_LENGTH} characters)."
        )
    if len(cleaned) > MAX_JD_LENGTH:
        raise EmptyJobDescriptionError(
            f"Job description is too long (maximum {MAX_JD_LENGTH:,} "
            f"characters). Please trim it."
        )

    gemini = client or GeminiClient()
    raw_json = gemini.generate_json(
        system_prompt=JOB_ANALYSIS_SYSTEM_PROMPT,
        user_text=cleaned,
        response_schema=JOB_RESPONSE_SCHEMA,
    )
    return JobAnalysis.from_json(raw_json)
