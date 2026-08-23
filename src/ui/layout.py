"""Reusable Streamlit layout helpers (sidebar, header, chrome polish)."""

from __future__ import annotations

import streamlit as st

from src.config.settings import APP_NAME, APP_TAGLINE, APP_VERSION, MODULES
from src.ui.components.cta import navigate_to

# Minimal CSS to hide Streamlit's default chrome (hamburger menu + the
# "Made with Streamlit" footer) for a cleaner SaaS look. This does NOT
# replace any native component — it only trims default decorations.
_CHROME_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def inject_chrome_css() -> None:
    """Hide Streamlit's default menu and footer for a cleaner SaaS look."""
    st.markdown(_CHROME_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render the branded sidebar with module navigation.

    The currently-active module is rendered as a ``primary`` button so the
    user can always see where they are. Clicking any module switches the
    active module via :func:`navigate_to`.
    """
    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()

        st.caption("NAVIGATION")
        current = st.session_state.get("selected_module", MODULES[0]["key"])
        for module in MODULES:
            is_active = module["key"] == current
            if st.button(
                f"{module['icon']}  {module['name']}",
                key=f"nav_{module['key']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                navigate_to(module["key"])

        st.divider()
        st.caption(f"v{APP_VERSION} · Skeleton")


def render_header(selected_key: str) -> None:
    """Render the page header for a feature module (non-dashboard).

    The dashboard renders its own branded hero, so this is only used for
    the eight feature modules.
    """
    from src.config.settings import get_module

    module = get_module(selected_key) or MODULES[0]
    st.title(f"{module['icon']}  {module['name']}")
    st.caption(module["description"])
    st.divider()
