"""Tests for the RAG assistant (ask_documents / summarize_document).

Gemini is MOCKED via FakeGeminiClient — no real API key required.

Covers hallucination defense A-E, citation validation, the most-important
RAG property tests (insufficient-context, grounded answer, adversarial
invalid citation), and the no-API-key fallback.
"""

from __future__ import annotations

import json

import pytest

from src.modules.rag.assistant import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    ask_documents,
    summarize_document,
)
from src.modules.rag.model import Confidence
from src.services.gemini import GeminiAPIError, GeminiConfigError, GeminiError


# ---- (30)(31) mocked Gemini answer / grounded answer ----------------


def test_ask_documents_grounded_answer(
    rag_index_careeros, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, _doc_id, _filename = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = ask_documents(
        "What technologies does CareerOS use?",
        idx, e, client=fake,
    )
    assert ans.grounded is True
    assert "Python" in ans.answer
    assert "Streamlit" in ans.answer
    # Source [1] must point to the actual document.
    assert ans.sources
    assert ans.sources[0].filename == "careeros.txt"
    assert ans.sources[0].source_id == "1"
    # Gemini was called exactly once.
    assert len(fake.calls) == 1
    # The prompt enforces grounded answering.
    assert "ONLY" in fake.calls[0]["system_prompt"] or "only" in fake.calls[0]["system_prompt"]


# ---- THE MOST IMPORTANT RAG PROPERTY TESTS (24) --------------------


def test_insufficient_context_no_gemini_call(rag_index_careeros, fake_gemini_client_factory) -> None:
    """Question about a cloud provider not in the doc -> insufficient,
    and Gemini MUST NOT be called."""
    idx, _doc_id, _filename = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response="{}")  # would be ignored
    ans = ask_documents(
        "What cloud provider does CareerOS use?",
        idx, e, client=fake,
    )
    assert ans.insufficient_context is True
    assert ans.grounded is False
    assert INSUFFICIENT_CONTEXT_MESSAGE in ans.answer
    # CRITICAL: Gemini was NOT called.
    assert fake.calls == [], "Gemini must not be called when retrieval is insufficient"


def test_grounded_answer_mentions_relevant_facts(
    rag_index_careeros, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, _doc_id, _filename = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = ask_documents(
        "What technologies does CareerOS use?",
        idx, e, client=fake,
    )
    assert ans.grounded is True
    assert "Python" in ans.answer
    assert "Streamlit" in ans.answer
    # Sources must point to the actual retrieved chunk.
    assert any(s.filename == "careeros.txt" for s in ans.sources)


def test_adversarial_invalid_citation_rejected(
    rag_index_careeros, rag_invalid_citation_json, fake_gemini_client_factory
) -> None:
    """Gemini returns 'AWS. [FAKE-123]' -> rejected (invalid source)."""
    idx, _doc_id, _filename = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_invalid_citation_json)
    ans = ask_documents(
        "What technologies does CareerOS use?",
        idx, e, client=fake,
    )
    # The answer is NOT trusted.
    assert ans.grounded is False
    assert any("invalid source IDs" in w or "FAKE-123" in w for w in ans.warnings)


# ---- (32) Gemini malformed response --------------------------------


def test_ask_documents_malformed_json_raises(rag_index_careeros, fake_gemini_client_factory) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response="not json {{{")
    from src.modules.rag.model import RAGParseError

    with pytest.raises(RAGParseError):
        ask_documents("What technologies does CareerOS use?", idx, e, client=fake)


def test_ask_documents_non_object_json_raises(rag_index_careeros, fake_gemini_client_factory) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=json.dumps(["not", "object"]))
    from src.modules.rag.model import RAGParseError

    with pytest.raises(RAGParseError):
        ask_documents("What technologies does CareerOS use?", idx, e, client=fake)


# ---- (33) Gemini failure --------------------------------------------


def test_ask_documents_propagates_api_error(rag_index_careeros, fake_gemini_client_factory) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(raise_exc=GeminiAPIError("rate limited"))
    with pytest.raises(GeminiError):
        ask_documents("What technologies does CareerOS use?", idx, e, client=fake)


# ---- (34) empty Gemini answer --------------------------------------


def test_ask_documents_empty_answer_falls_back(rag_index_careeros, fake_gemini_client_factory) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=json.dumps({"answer": "", "cited_source_ids": [], "insufficient_context": True}))
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=fake)
    assert ans.grounded is False
    assert ans.insufficient_context is True
    assert any("empty" in w.lower() for w in ans.warnings)


# ---- (35) invalid source citation ----------------------------------


def test_invalid_citation_marked_ungrounded(
    rag_index_careeros, rag_invalid_citation_json, fake_gemini_client_factory
) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_invalid_citation_json)
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=fake)
    assert ans.grounded is False
    assert any("FAKE-123" in w for w in ans.warnings)


# ---- (36)(38) valid citation / Gemini cannot invent source IDs -----


def test_valid_citation_is_trusted(
    rag_index_careeros, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=fake)
    assert ans.grounded is True
    assert ans.sources[0].source_id == "1"  # valid, supplied ID


def test_gemini_cannot_invent_source_ids(
    rag_index_careeros, rag_invalid_citation_json, fake_gemini_client_factory
) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_invalid_citation_json)
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=fake)
    # FAKE-123 is not a supplied source -> rejected.
    assert ans.grounded is False
    assert not any(s.source_id == "FAKE-123" for s in ans.sources)


def test_inline_citation_in_answer_extracted(
    rag_index_careeros, fake_gemini_client_factory
) -> None:
    """Citations inline in the answer text are also validated."""
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    # Answer cites [2] inline but [2] isn't supplied (only [1] is).
    fake = fake_gemini_client_factory(
        response=json.dumps({"answer": "Python and Streamlit. [2]", "cited_source_ids": []})
    )
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=fake)
    assert ans.grounded is False
    assert any("2" in w for w in ans.warnings)


# ---- (39) no Gemini call when retrieval insufficient --------------- (covered by test_insufficient_context_no_gemini_call)


# ---- (49) no API key (default client raises config) ----------------


def test_ask_documents_default_client_without_key_no_gemini(rag_index_careeros, monkeypatch) -> None:
    """Without a key, ask_documents with client=None returns a deterministic
    retrieval-only answer (no Gemini call)."""
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ans = ask_documents("What technologies does CareerOS use?", idx, e, client=None)
    # Retrieval-only deterministic answer.
    assert ans.grounded is False
    assert ans.sources  # sources still returned from retrieval
    assert any("Gemini configuration" in w for w in ans.warnings)


# ---- document isolation at the assistant level ----------------------


def test_ask_documents_document_isolation(
    rag_index_with_docs, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, docs, _chunks = rag_index_with_docs
    e = LocalEmbedder()
    doc_a_id = [d for d, f in docs.items() if f == "careeros.txt"][0]
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = ask_documents(
        "What technologies does CareerOS use?",
        idx, e, client=fake, document_filter=[doc_a_id],
    )
    # All retrieved chunks must be from doc A.
    for r in ans.retrieved_chunks:
        assert r.chunk.document_id == doc_a_id


# ---- summarize_document --------------------------------------------


def test_summarize_document_grounded(
    rag_index_careeros, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, doc_id, filename = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = summarize_document(doc_id, filename, idx, e, client=fake)
    assert ans.grounded is True
    assert ans.sources


def test_summarize_no_client_returns_retrieval_only(rag_index_careeros, monkeypatch) -> None:
    idx, doc_id, filename = rag_index_careeros
    e = LocalEmbedder()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ans = summarize_document(doc_id, filename, idx, e, client=None)
    assert ans.grounded is False
    assert any("Gemini configuration" in w for w in ans.warnings)


def test_summarize_unknown_document_insufficient(rag_index_careeros, rag_embedder) -> None:
    idx, _doc_id, filename = rag_index_careeros
    ans = summarize_document("nonexistent", filename, idx, rag_embedder, client=None)
    assert ans.insufficient_context is True


# ---- confidence heuristic ------------------------------------------


def test_confidence_high_for_strong_retrieval(
    rag_index_careeros, rag_grounded_json, fake_gemini_client_factory
) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response=rag_grounded_json)
    ans = ask_documents("Python Streamlit CareerOS", idx, e, client=fake)
    assert ans.confidence in (Confidence.HIGH, Confidence.MEDIUM)


def test_confidence_low_for_insufficient(rag_index_careeros, fake_gemini_client_factory) -> None:
    idx, _, _ = rag_index_careeros
    e = LocalEmbedder()
    fake = fake_gemini_client_factory(response="{}")
    ans = ask_documents("capital of France", idx, e, client=fake)
    assert ans.confidence is Confidence.LOW


# ---- local import for the embedder used in tests --------------------


from src.modules.rag.embedder import LocalEmbedder  # noqa: E402
