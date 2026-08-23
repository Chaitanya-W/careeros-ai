"""Reusable card-style components for the CareerOS UI.

All helpers use Streamlit native primitives (``st.container(border=True)``,
``st.metric``, ``st.caption`` …) so the design composes cleanly and stays
responsive. Where real data is not yet available, helpers render the ``--``
placeholder rather than fabricated numbers.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

PLACEHOLDER: str = "--"


def metric_card(
    label: str,
    value: str = PLACEHOLDER,
    hint: Optional[str] = None,
) -> None:
    """A bordered metric card: small label, big value, optional hint.

    Defaults to ``--`` so missing data is obvious (no fake numbers).
    """
    with st.container(border=True):
        st.metric(label=label, value=value, border=False)
        if hint:
            st.caption(hint)


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    """A section heading with an optional muted subtitle."""
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def empty_state_panel(title: str, hint: str) -> None:
    """A helpful empty-state card — used wherever data has yet to be created."""
    with st.container(border=True):
        st.write(f"**{title}**")
        st.caption(hint)


def step_row(index: int, label: str, target_module: str) -> None:
    """A Getting-Started step row: status marker + label + Start button.

    The status marker is ``○`` (not started) — there is no real progress
    data yet, so we never imply completion we cannot back up.
    """
    status_col, label_col, button_col = st.columns([0.08, 0.62, 0.30])
    with status_col:
        st.markdown("○")
    with label_col:
        st.markdown(f"**{index}. {label}**")
    with button_col:
        if st.button(
            "Start",
            key=f"step_{target_module}",
            use_container_width=True,
        ):
            from src.ui.components.cta import navigate_to

            navigate_to(target_module)
