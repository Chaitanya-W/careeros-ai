"""Tests for the local embedder + vector store (deterministic, isolated)."""

from __future__ import annotations

import pytest

from src.modules.rag.embedder import LocalEmbedder, cosine_similarity
from src.modules.rag.model import DocumentChunk
from src.modules.rag.store import VectorIndex


# ---- (19) embedding interface ---------------------------------------


def test_embed_returns_fixed_dim_vector() -> None:
    e = LocalEmbedder(dims=128)
    v = e.embed("hello world")
    assert len(v) == 128


def test_embed_empty_text_is_zero_vector() -> None:
    e = LocalEmbedder()
    v = e.embed("")
    assert all(x == 0.0 for x in v)


def test_embed_batch() -> None:
    e = LocalEmbedder()
    vecs = e.embed_batch(["a", "b", ""])
    assert len(vecs) == 3
    assert all(len(v) == e.dims for v in vecs)


# ---- (20) local embedding fallback (no API key, no download) --------


def test_local_embedder_is_deterministic_across_instances() -> None:
    a = LocalEmbedder().embed("Python and Streamlit")
    b = LocalEmbedder().embed("Python and Streamlit")
    assert a == b  # stable across instances (hashlib, not salted hash())


def test_local_embedder_no_dependencies() -> None:
    """The embedder is pure Python (no torch/sentence-transformers)."""
    import src.modules.rag.embedder as mod
    import inspect

    src = inspect.getsource(mod)
    assert "sentence_transformers" not in src
    assert "import torch" not in src


def test_cosine_similarity_identical_is_one() -> None:
    e = LocalEmbedder()
    v = e.embed("Python Streamlit")
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0] * 8, [1.0] * 8) == 0.0


# ---- (21) vector index creation -------------------------------------


def test_vector_index_starts_empty() -> None:
    idx = VectorIndex()
    assert len(idx) == 0
    assert idx.document_ids == []


# ---- (22) vector insertion ------------------------------------------


def test_vector_index_add_and_len() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    ch = DocumentChunk(chunk_id="A-C001", document_id="A", chunk_index=0, text="hello", metadata={})
    idx.add(ch, e.embed("hello"))
    assert len(idx) == 1
    assert idx.has_document("A")
    assert "A" in idx.document_ids


def test_vector_index_chunk_count_for() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    for i, t in enumerate(["a", "b", "c"]):
        idx.add(DocumentChunk(chunk_id=f"A-C{i}", document_id="A", chunk_index=i, text=t, metadata={}), e.embed(t))
    assert idx.chunk_count_for("A") == 3


# ---- (23) top-k retrieval -------------------------------------------


def test_retrieve_top_k() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    idx.add(DocumentChunk(chunk_id="A-C0", document_id="A", chunk_index=0, text="Python Streamlit", metadata={}), e.embed("Python Streamlit"))
    idx.add(DocumentChunk(chunk_id="B-C0", document_id="B", chunk_index=0, text="Docker containers", metadata={}), e.embed("Docker containers"))
    res = idx.retrieve(e.embed("Python Streamlit"), top_k=2)
    assert len(res) == 2
    top_chunk, top_score = res[0]
    assert top_chunk.chunk_id == "A-C0"
    assert top_score > 0.0


# ---- (24) document filtering ----------------------------------------


def test_retrieve_with_document_filter() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    idx.add(DocumentChunk(chunk_id="A-C0", document_id="A", chunk_index=0, text="Python", metadata={}), e.embed("Python"))
    idx.add(DocumentChunk(chunk_id="B-C0", document_id="B", chunk_index=0, text="Python", metadata={}), e.embed("Python"))
    res = idx.retrieve(e.embed("Python"), top_k=5, document_filter=["A"])
    assert all(c.document_id == "A" for c, _ in res)


# ---- (25) document isolation ---------------------------------------


def test_document_isolation_blocks_other_doc() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    idx.add(DocumentChunk(chunk_id="A-C0", document_id="A", chunk_index=0, text="Python", metadata={}), e.embed("Python"))
    idx.add(DocumentChunk(chunk_id="B-C0", document_id="B", chunk_index=0, text="Python", metadata={}), e.embed("Python"))
    res = idx.retrieve(e.embed("Python"), top_k=5, document_filter=["A"])
    assert all(c.document_id == "A" for c, _ in res)
    assert not any(c.document_id == "B" for c, _ in res)


# ---- (42) stale chunk prevention -----------------------------------


def test_remove_document_clears_chunks() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    for i, t in enumerate(["a", "b"]):
        idx.add(DocumentChunk(chunk_id=f"A-C{i}", document_id="A", chunk_index=i, text=t, metadata={}), e.embed(t))
    idx.add(DocumentChunk(chunk_id="B-C0", document_id="B", chunk_index=0, text="x", metadata={}), e.embed("x"))
    removed = idx.remove_document("A")
    assert removed == 2
    assert not idx.has_document("A")
    assert idx.has_document("B")
    # No stale A chunks retrievable.
    res = idx.retrieve(e.embed("a"), top_k=5)
    assert all(c.document_id != "A" for c, _ in res)


def test_clear_all() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    idx.add(DocumentChunk(chunk_id="A-C0", document_id="A", chunk_index=0, text="x", metadata={}), e.embed("x"))
    idx.clear()
    assert len(idx) == 0


def test_retrieve_empty_index() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    assert idx.retrieve(e.embed("x"), top_k=5) == []


def test_retrieve_top_k_zero() -> None:
    idx = VectorIndex()
    e = LocalEmbedder()
    idx.add(DocumentChunk(chunk_id="A-C0", document_id="A", chunk_index=0, text="x", metadata={}), e.embed("x"))
    assert idx.retrieve(e.embed("x"), top_k=0) == []
