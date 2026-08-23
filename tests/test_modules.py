"""Tests for the module dispatcher and each module's render() contract.

Uses Streamlit's AppTest harness to execute render() headlessly and
assert that no exceptions/errors are raised and that every registered
module is wired into the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config.settings import MODULES
from src.modules import _REGISTRY, render_selected

# Absolute path to app.py — AppTest resolves relative paths against the
# *calling* file's directory, so we anchor to the project root.
_APP_PATH: Path = Path(__file__).resolve().parent.parent / "app.py"


def test_dispatcher_covers_every_registered_module() -> None:
    """Every key in MODULES must have a render() in the dispatcher."""
    registry_keys = set(_REGISTRY.keys())
    nav_keys = {m["key"] for m in MODULES}
    assert registry_keys == nav_keys, (
        f"Mismatch: dispatcher has {registry_keys - nav_keys} extra, "
        f"missing {nav_keys - registry_keys}"
    )


def test_every_module_render_is_callable() -> None:
    """Each module's render() must be callable."""
    for key, fn in _REGISTRY.items():
        assert callable(fn), f"render for {key} is not callable"


def test_dashboard_renders_without_errors() -> None:
    """The dashboard (default landing view) must render cleanly."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    assert not at.exception, "dashboard render raised an exception"
    assert not at.error, "dashboard render reported an error"
    # The dashboard must show the branded hero + a welcome line.
    assert any("CareerOS AI" in m.value for m in at.markdown), (
        "dashboard branding hero did not render"
    )


def test_dashboard_has_three_score_cards() -> None:
    """The dashboard must show the three score-card metrics."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    metric_labels = [m.label for m in at.metric]
    for expected in ("Resume Score", "Job Match", "Career Readiness"):
        assert expected in metric_labels, f"missing metric: {expected}"


def test_dashboard_has_progress_indicator() -> None:
    """The dashboard must show a career-progress section.

    Asserts the section header rendered (AppTest 1.62 does not expose a
    dedicated ``progress`` element collection, so we verify the section
    that contains the progress bar rendered).
    """
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    subheaders = [s.value for s in at.subheader]
    assert "Career Progress" in subheaders, "career progress section missing"


def test_dashboard_shows_placeholder_values() -> None:
    """Score cards must show '--' until real data exists (no fake numbers)."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    for metric in at.metric:
        assert metric.value == "--", (
            f"metric {metric.label!r} shows fake value {metric.value!r}"
        )


def test_sidebar_navigation_changes_selected_module() -> None:
    """Clicking a sidebar nav button must update ``selected_module``."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    # Sidebar buttons are in MODULES order; index 3 = skill_gap.
    at.sidebar.button[3].click().run()
    assert at.session_state["selected_module"] == "skill_gap"
    assert not at.exception and not at.error


def test_sidebar_active_state_highlight() -> None:
    """The currently-selected module's sidebar button must be ``primary``.

    Note: AppTest's ``Button.type`` returns the widget *kind* (``'button'``),
    not the styling. The styling is exposed via ``Button.proto.type`` which
    returns ``'primary'`` or ``'secondary'``.
    """
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    # Default landing = dashboard -> its button must be primary.
    primary = [b for b in at.sidebar.button if b.proto.type == "primary"]
    assert len(primary) == 1, "exactly one sidebar button should be active"
    assert "Dashboard" in primary[0].label

    # Navigate to skill_gap -> its button must now be primary.
    at.sidebar.button[3].click().run()
    primary = [b for b in at.sidebar.button if b.proto.type == "primary"]
    assert len(primary) == 1, "exactly one sidebar button should be active"
    assert "Skill Gap" in primary[0].label


def test_dashboard_cta_navigates() -> None:
    """A dashboard Quick-Action CTA must switch the active module."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=15).run()
    assert at.session_state["selected_module"] == "dashboard"
    for b in at.button:
        if b.label == "Match a Job":
            b.click().run()
            break
    assert at.session_state["selected_module"] == "jd_analysis"
    assert not at.exception and not at.error
