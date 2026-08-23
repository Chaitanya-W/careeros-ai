"""CareerOS AI — AI Career Intelligence & Interview Copilot.

Streamlit application entry point. This file stays thin: it configures
the page, initializes session state, renders the sidebar, and dispatches
to the selected module's ``render()`` function. All UI logic lives in
``src/ui`` and ``src/modules``.

Run locally::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.config.settings import APP_NAME, APP_TAGLINE, MODULES
from src.core.state import init_state
from src.modules import render_selected
from src.ui.layout import inject_chrome_css, render_header, render_sidebar


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_chrome_css()
    init_state()
    render_sidebar()

    selected_key: str = st.session_state.get(
        "selected_module", MODULES[0]["key"]
    )

    # The dashboard renders its own branded hero; feature modules use the
    # shared render_header for a consistent title + description.
    if selected_key != "dashboard":
        render_header(selected_key)

    render_selected(selected_key)


if __name__ == "__main__":
    main()
