"""Tests for retrieval (threshold, document filtering, insufficient) +
context building (size-limited, source metadata)."""

from __future__ import annotations

import pytest

from src.modules.rag.chunker import chunk_text, make_chunk_id
from src.modules.rag.context import MAX_CONTEXT_CHARS, build_context, format_citations
from src.modules.rag.embedder import LocalEmbedder
from src.modules.rag.model import DocumentChunk, RetrievalResult
from src.modules.rag.retrieval import DEFAULT_THRESHOLD, DEFAULT_TOP_K, retrieve
from src.modules.rag.store import VectorIndex


# ---- (24-25) document filtering / isolation (retrieval layer) -------


def _build_index(texts: dict, embedder):
    idx = VectorIndex()
    for doc_id, text in texts.items():
        parts = chunk_text(text)
        for i, c in enumerate(parts):
            ch = DocumentChunk(chunk_id=make_chunk_id(doc_id, i), document_id=doc_id, chunk_index=i, text=c, metadata={"filename": f"{doc_id}.txt", "page": 1, "chunk_index": i, "document_id": doc_id})
            idx.add(ch, embedder.embed(ch.text))
    return idx


def test_retrieve_returns_ranked_results(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit CareerOS", "B": "Docker containers"}, rag_embedder)
    res = retrieve("Python Streamlit", idx, rag_embedder, threshold=0.1)
    assert res
    assert all(r.rank >= 1 for r in res)
    # Ranks are sequential 1..N.
    assert [r.rank for r in res] == list(range(1, len(res) + 1))


# ---- (26) relevance threshold --------------------------------------


def test_retrieve_threshold_filters_low_scores(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit"}, rag_embedder)
    # A clearly unrelated query should be below threshold.
    res = retrieve("capital of France", idx, rag_embedder, threshold=DEFAULT_THRESHOLD)
    assert res == []  # filtered


def test_retrieve_threshold_configurable(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit"}, rag_embedder)
    res_low = retrieve("Python Streamlit", idx, rag_embedder, threshold=0.1)
    # A non-exact query won't reach cosine 1.0, so a very high threshold
    # filters everything.
    res_high = retrieve("Python and Streamlit technologies", idx, rag_embedder, threshold=0.999)
    assert len(res_low) >= 1
    assert res_high == []  # nothing clears 0.999


# ---- (27) insufficient retrieval -----------------------------------


def test_retrieve_empty_query_returns_empty(rag_embedder) -> None:
    idx = _build_index({"A": "Python"}, rag_embedder)
    assert retrieve("", idx, rag_embedder) == []
    assert retrieve("   ", idx, rag_embedder) == []


def test_retrieve_empty_index_returns_empty(rag_embedder) -> None:
    idx = VectorIndex()
    assert retrieve("anything", idx, rag_embedder) == []


def test_retrieve_document_isolation(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit", "B": "Python Streamlit"}, rag_embedder)
    res = retrieve("Python Streamlit", idx, rag_embedder, threshold=0.1, document_filter=["A"])
    assert all(r.chunk.document_id == "A" for r in res)


# ---- (28) context building -----------------------------------------


def test_build_context_numbers_sources(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit CareerOS project"}, rag_embedder)
    res = retrieve("Python Streamlit", idx, rag_embedder, threshold=0.1)
    context, sources = build_context(res)
    assert sources
    assert [s.source_id for s in sources] == [str(i + 1) for i in range(len(sources))]
    assert "[1]" in context


# ---- (29) context size limit ---------------------------------------


def test_build_context_respects_size_limit() -> None:
    chunks = [
        DocumentChunk(chunk_id=f"C{i}", document_id="A", chunk_index=i, text="x" * 2000, metadata={"filename": "f.txt", "page": 1, "chunk_index": i})
        for i in range(10)
    ]
    results = [RetrievalResult(chunk=c, score=0.9, rank=i + 1) for i, c in enumerate(chunks)]
    context, sources = build_context(results, max_chars=3000)
    assert len(context) <= 3500  # bounded (some overhead for labels)
    assert len(sources) <= 10


def test_build_context_skips_empty_text() -> None:
    chunks = [
        DocumentChunk(chunk_id="C0", document_id="A", chunk_index=0, text="", metadata={"filename": "f.txt", "page": 1, "chunk_index": 0}),
        DocumentChunk(chunk_id="C1", document_id="A", chunk_index=1, text="real text", metadata={"filename": "f.txt", "page": 1, "chunk_index": 1}),
    ]
    results = [RetrievalResult(chunk=c, score=0.9, rank=i + 1) for i, c in enumerate(chunks)]
    context, sources = build_context(results)
    assert len(sources) == 1  # empty-text chunk skipped


# ---- (37) citation metadata comes from retrieval -------------------


def test_citation_metadata_from_retrieval(rag_embedder) -> None:
    idx = _build_index({"A": "Python Streamlit"}, rag_embedder)
    res = retrieve("Python Streamlit", idx, rag_embedder, threshold=0.1)
    _, sources = build_context(res)
    for s in sources:
        assert s.filename == "A.txt"  # from chunk metadata
        assert s.page == 1
        assert s.document_id == "A"


def test_format_citations() -> None:
    from src.modules.rag.model import RAGSource

    sources = [
        RAGSource(source_id="1", filename="a.pdf", page=4),
        RAGSource(source_id="2", filename="b.txt", page=None, chunk_index=2),
    ]
    cites = format_citations(sources)
    assert cites[0] == "[1] a.pdf — page 4"
    assert cites[1] == "[2] b.txt — chunk 2"


def test_defaults() -> None:
    assert DEFAULT_TOP_K > 0
    assert 0.0 < DEFAULT_THRESHOLD < 1.0
    assert MAX_CONTEXT_CHARS > 0
