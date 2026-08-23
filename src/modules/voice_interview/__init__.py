"""AI Interview Coach module.

Public API: :func:`render` draws the configure -> start -> question ->
answer -> feedback -> report workflow. Internally delegates to
:mod:`src.modules.voice_interview.ui`.

Submodules
----------
- ``model``     : InterviewCategory, Difficulty, SessionStatus, InterviewQuestion,
                  InterviewAnswer, AnswerEvaluation, InterviewSession, InterviewReport.
- ``planner``   : deterministic question planning + grounded fallback templates (no AI).
- ``validator`` : question validation (reuses Application Copilot evidence validator).
- ``prompt``     : Gemini question-gen + answer-eval prompts + JSON schemas.
- ``coach``     : generate_questions / evaluate_answer / build_report (reuses GeminiClient).
- ``ui``        : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.voice_interview.ui import render

MODULE_NAME: str = "AI Interview"
MODULE_KEY: str = "voice_interview"
MODULE_DESCRIPTION: str = (
    "Practice realistic mock interviews with live feedback and scoring."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
