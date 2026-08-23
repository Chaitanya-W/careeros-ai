"""Call-to-action components and the shared navigation helper."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def navigate_to(module_key: str) -> None:
    """Set the active module and immediately rerun.

    Centralizing this keeps navigation behavior consistent whether it is
    triggered from the sidebar, a dashboard CTA, or a Getting-Started step.
    """
    st.session_state["selected_module"] = module_key
    st.rerun()


def nav_cta(
    label: str,
    target_module: str,
    *,
    key: Optional[str] = None,
    primary: bool = True,
) -> None:
    """A full-width button that navigates to ``target_module`` on click."""
    if st.button(
        label,
        key=key or f"cta_{target_module}",
        use_container_width=True,
        type="primary" if primary else "secondary",
    ):
        navigate_to(target_module)
