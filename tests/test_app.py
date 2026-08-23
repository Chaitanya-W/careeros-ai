"""Smoke tests for the Streamlit entry point (no server required).

These tests import `app` to guarantee it parses and that its top-level
dependencies (`src.config`, `src.core`, `src.ui`) import cleanly. They
do **not** execute Streamlit rendering, so no ScriptRunContext is
required.
"""

from __future__ import annotations

import importlib


def test_app_module_imports() -> None:
    """`app` module must be importable without a running Streamlit server."""
    module = importlib.import_module("app")
    assert hasattr(module, "main"), "app.main entry point is missing"


def test_main_is_callable() -> None:
    """`app.main` must be a callable (not yet invoked here)."""
    import app

    assert callable(app.main)
