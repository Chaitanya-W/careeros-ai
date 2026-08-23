"""Branding components — the CareerOS hero block shown on the dashboard."""

from __future__ import annotations

import streamlit as st

from src.config.settings import APP_NAME, APP_TAGLINE, APP_VERSION


def brand_hero() -> None:
    """Render the CareerOS AI branded hero card."""
    with st.container(border=True):
        st.markdown(f"## {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.caption(f"Version {APP_VERSION} · Skeleton dashboard")
