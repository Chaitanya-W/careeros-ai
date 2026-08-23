"""Application configuration & module registry.

Centralizes app metadata and the ordered list of navigation modules. The
sidebar is rendered *data-driven* from :data:`MODULES`, so updating the
list here automatically updates the sidebar (see ``src/ui/layout.py``).

Adding / reordering modules is a one-place change: edit :data:`MODULES`,
create a package under ``src/modules/<key>/`` exposing ``render()``, and
register it in ``src/modules/__init__.py``'s dispatcher.
"""

from __future__ import annotations

APP_NAME: str = "CareerOS AI"
APP_TAGLINE: str = "AI Career Intelligence & Interview Copilot"
APP_VERSION: str = "0.2.0"

# Ordered module registry — order defines sidebar navigation order.
# Each entry: {key, name, icon, description, status}
MODULES: list[dict[str, str]] = [
    {
        "key": "dashboard",
        "name": "Dashboard",
        "icon": "🏠",
        "description": "Your career command center — scores, readiness, and next steps at a glance.",
        "status": "Ready",
    },
    {
        "key": "resume_intelligence",
        "name": "Resume Intelligence",
        "icon": "📄",
        "description": "Parse, analyze, and optimize resumes for ATS compatibility and recruiter readability.",
        "status": "Planned",
    },
    {
        "key": "jd_analysis",
        "name": "Job Match",
        "icon": "🎯",
        "description": "Analyze job descriptions and measure match against your profile.",
        "status": "Planned",
    },
    {
        "key": "skill_gap",
        "name": "Skill Gap",
        "icon": "🧠",
        "description": "Compare your skills against target roles to surface actionable gaps.",
        "status": "Planned",
    },
    {
        "key": "career_roadmap",
        "name": "Career Roadmap",
        "icon": "🗺️",
        "description": "Generate tailored, milestone-based career growth plans with learning resources.",
        "status": "Planned",
    },
    {
        "key": "application_copilot",
        "name": "Application Copilot",
        "icon": "✍️",
        "description": "Draft tailored cover letters, outreach messages, and application materials.",
        "status": "Planned",
    },
    {
        "key": "rag",
        "name": "RAG",
        "icon": "📚",
        "description": "Ground AI responses in your uploaded documents for accurate, citation-backed answers.",
        "status": "Planned",
    },
    {
        "key": "voice_interview",
        "name": "AI Interview",
        "icon": "🎤",
        "description": "Practice realistic mock interviews with live feedback and scoring.",
        "status": "Planned",
    },
    {
        "key": "salary_negotiation",
        "name": "Salary Negotiator",
        "icon": "💰",
        "description": "Benchmark compensation and rehearse data-backed negotiation scripts.",
        "status": "Planned",
    },
    {
        "key": "evaluation_dashboard",
        "name": "Evaluation",
        "icon": "📊",
        "description": "Track your progress, AI answer quality, and engagement over time.",
        "status": "Planned",
    },
]


def get_module(key: str) -> dict[str, str] | None:
    """Return the module descriptor for ``key`` or ``None``."""
    return next((m for m in MODULES if m["key"] == key), None)
