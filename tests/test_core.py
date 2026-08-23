"""Unit tests for core helpers (config registry, session-state defaults).

These tests avoid executing Streamlit rendering; where `st.session_state`
is required, the `streamlit` testing helpers provide a lightweight context.
"""

from __future__ import annotations

from src.config.settings import MODULES, get_module


def test_modules_registered() -> None:
    """The ten nav modules from the sidebar spec must be present."""
    expected = {
        "dashboard",
        "resume_intelligence",
        "jd_analysis",
        "skill_gap",
        "career_roadmap",
        "application_copilot",
        "rag",
        "voice_interview",
        "salary_negotiation",
        "evaluation_dashboard",
    }
    assert {m["key"] for m in MODULES} == expected
    assert len(MODULES) == 10


def test_module_order_matches_sidebar_spec() -> None:
    """Sidebar order must match the required navigation sequence."""
    expected_order = [
        "dashboard",
        "resume_intelligence",
        "jd_analysis",
        "skill_gap",
        "career_roadmap",
        "application_copilot",
        "rag",
        "voice_interview",
        "salary_negotiation",
        "evaluation_dashboard",
    ]
    actual_order = [m["key"] for m in MODULES]
    assert actual_order == expected_order, (
        f"module order mismatch: {actual_order}"
    )


def test_rag_is_registered_between_application_copilot_and_interview() -> None:
    """RAG must sit between Application Copilot and AI Interview."""
    keys = [m["key"] for m in MODULES]
    rag_index = keys.index("rag")
    assert keys[rag_index - 1] == "application_copilot"
    assert keys[rag_index + 1] == "voice_interview"


def test_each_module_has_required_fields() -> None:
    """Every registry entry must carry key, name, icon, description, status."""
    required = {"key", "name", "icon", "description", "status"}
    for module in MODULES:
        assert required.issubset(module.keys()), (
            f"Module {module.get('key')} is missing fields: "
            f"{required - module.keys()}"
        )


def test_dashboard_is_first_module() -> None:
    """Dashboard must be the default landing module (index 0)."""
    assert MODULES[0]["key"] == "dashboard"


def test_get_module_lookup() -> None:
    """`get_module` must return the descriptor for a known key."""
    found = get_module("rag")
    assert found is not None
    assert found["name"] == "RAG"
    assert found["icon"] == "📚"
    assert "documents" in found["description"].lower()


def test_get_module_unknown_returns_none() -> None:
    """Unknown keys must resolve to None."""
    assert get_module("does_not_exist") is None
