"""Tests for the RAG data models (RAGDocument, DocumentChunk, RetrievalResult,
RAGSource, RAGAnswer, Confidence)."""

from __future__ import annotations

import pytest

from src.modules.rag.model import (
    Confidence,
    DocumentChunk,
    RAGAnswer,
    RAGDocument,
    RAGError,
    RAGParseError,
    RAGSource,
    RetrievalResult,
)


# ---- Confidence enum (used by RAGAnswer) ------------------------------


def test_confidence_members() -> None:
    assert {c.value for c in Confidence} == {"high", "medium", "low"}


def test_confidence_from_value() -> None:
    assert Confidence.from_value("HIGH") is Confidence.HIGH
    assert Confidence.from_value(None) is Confidence.LOW
    assert Confidence.from_value("bogus") is Confidence.LOW


# ---- (1) RAGDocument model --------------------------------------------


def test_rag_document_defaults() -> None:
    d = RAGDocument()
    assert d.document_id == ""
    assert d.filename == ""
    assert d.chunk_count == 0
    assert d.metadata == {}


def test_rag_document_from_dict() -> None:
    d = RAGDocument.from_dict(
        {"document_id": "abc", "filename": "f.pdf", "file_type": "pdf", "character_count": 100, "chunk_count": 3, "metadata": {"filename": "f.pdf"}}
    )
    assert d.document_id == "abc"
    assert d.chunk_count == 3
    assert d.metadata["filename"] == "f.pdf"


def test_rag_document_from_dict_non_dict_raises() -> None:
    with pytest.raises(RAGParseError):
        RAGDocument.from_dict("nope")  # type: ignore[arg-type]


# ---- (2) DocumentChunk model ------------------------------------------


def test_document_chunk_defaults() -> None:
    c = DocumentChunk()
    assert c.chunk_id == ""
    assert c.chunk_index == 0
    assert c.metadata == {}


def test_document_chunk_from_dict() -> None:
    c = DocumentChunk.from_dict(
        {"chunk_id": "abc-C001", "document_id": "abc", "chunk_index": 0, "text": "hello", "metadata": {"page": 2}}
    )
    assert c.chunk_id == "abc-C001"
    assert c.metadata["page"] == 2


# ---- (3) RetrievalResult model ---------------------------------------


def test_retrieval_result() -> None:
    r = RetrievalResult(chunk=DocumentChunk(chunk_id="x"), score=0.8, rank=1)
    assert r.chunk.chunk_id == "x"
    assert r.score == 0.8
    assert r.rank == 1
    d = r.to_dict()
    assert d["score"] == 0.8


# ---- (4) RAGSource model ---------------------------------------------


def test_rag_source_defaults() -> None:
    s = RAGSource()
    assert s.source_id == ""
    assert s.page is None
    assert s.chunk_index == 0


def test_rag_source_from_dict() -> None:
    s = RAGSource.from_dict({"source_id": "1", "document_id": "abc", "filename": "f.pdf", "page": "4", "chunk_index": 2, "text": "snippet"})
    assert s.source_id == "1"
    assert s.page == 4  # coerced from "4"
    assert s.chunk_index == 2


def test_rag_source_page_none_when_not_int() -> None:
    s = RAGSource.from_dict({"page": "N/A"})
    assert s.page is None


# ---- (5) RAGAnswer model ---------------------------------------------


def test_rag_answer_defaults() -> None:
    a = RAGAnswer()
    assert a.answer == ""
    assert a.confidence is Confidence.LOW
    assert a.grounded is False
    assert a.sources == []
    assert a.insufficient_context is False


def test_rag_answer_from_dict() -> None:
    a = RAGAnswer.from_dict(
        {
            "answer": "yes",
            "confidence": "high",
            "grounded": True,
            "sources": [{"source_id": "1"}],
            "query": "q",
            "insufficient_context": False,
            "warnings": ["w"],
        }
    )
    assert a.answer == "yes"
    assert a.confidence is Confidence.HIGH
    assert a.grounded is True
    assert len(a.sources) == 1


def test_rag_answer_to_dict_round_trip() -> None:
    a = RAGAnswer(
        answer="x",
        confidence=Confidence.MEDIUM,
        grounded=True,
        sources=[RAGSource(source_id="1", filename="f.pdf")],
        query="q",
    )
    d = a.to_dict()
    assert d["confidence"] == "medium"
    again = RAGAnswer.from_dict(d)
    assert again.confidence is Confidence.MEDIUM
    assert again.grounded is True
    assert again.sources[0].source_id == "1"


def test_error_hierarchy() -> None:
    assert issubclass(RAGParseError, RAGError)
