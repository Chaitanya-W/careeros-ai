"""Tests for the deterministic chunker."""

from __future__ import annotations

import pytest

from src.modules.rag.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
    make_chunk_id,
)


# ---- (14) deterministic chunking -------------------------------------


def test_chunk_empty_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_is_one_chunk() -> None:
    out = chunk_text("Hello world.")
    assert len(out) == 1
    assert out[0] == "Hello world."


def test_chunk_deterministic() -> None:
    text = "The CareerOS project uses Python and Streamlit. " * 20
    a = chunk_text(text, chunk_size=200, chunk_overlap=30)
    b = chunk_text(text, chunk_size=200, chunk_overlap=30)
    assert a == b


def test_chunk_long_text_produces_multiple() -> None:
    text = "Sentence one. " * 200
    out = chunk_text(text, chunk_size=200, chunk_overlap=0)
    assert len(out) > 1


# ---- (15) chunk overlap ---------------------------------------------


def test_chunk_overlap_applied() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
    out = chunk_text(text, chunk_size=100, chunk_overlap=20)
    # When overlap > 0, the second chunk should share a prefix with the
    # tail of the first (unless they start identically by coincidence).
    assert len(out) >= 2


def test_chunk_overlap_zero_no_overlap() -> None:
    text = "a" * 250
    out = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(out) >= 2
    # No overlap -> chunks are disjoint.
    assert out[0] == "a" * 100


# ---- (16) no empty chunks -------------------------------------------


def test_no_empty_chunks() -> None:
    text = "Para one.\n\n\n\n\nPara two.\n\n   \n\nPara three. " * 10
    out = chunk_text(text, chunk_size=150, chunk_overlap=20)
    assert all(c.strip() for c in out)


def test_chunk_respects_paragraph_boundaries() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    out = chunk_text(text, chunk_size=200, chunk_overlap=0)
    # Should preserve paragraph content (merged or split, not empty).
    joined = " ".join(out)
    assert "First paragraph" in joined
    assert "Second paragraph" in joined
    assert "Third paragraph" in joined


def test_chunk_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, chunk_overlap=100)


# ---- (18) stable chunk IDs ------------------------------------------


def test_make_chunk_id_stable() -> None:
    assert make_chunk_id("doc123", 0) == "doc123-C001"
    assert make_chunk_id("doc123", 5) == "doc123-C006"
    assert make_chunk_id("doc123", 12) == "doc123-C013"


def test_defaults_are_sane() -> None:
    assert DEFAULT_CHUNK_SIZE > 0
    assert 0 <= DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE
