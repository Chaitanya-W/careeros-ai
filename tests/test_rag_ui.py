"""Tests for the RAG Career Assistant UI (NO API key required).

Uses Streamlit's AppTest harness. Gemini is never invoked for the render
tests (no Ask click), so no key is needed.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


# ---- (44) UI empty state --------------------------------------------


def test_rag_renders_empty_state(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.run()

    assert not at.exception, "render raised an exception"
    assert not at.error
    # Empty state shown when no documents indexed.
    assert any(
        "No documents indexed yet" in m.value for m in at.markdown
    ), "empty state missing"
    # Uploader must be present.
    assert len(at.file_uploader) == 1


# ---- (45) UI indexed document rendering -----------------------------


def test_rag_renders_indexed_document(app_path: str, sample_md_bytes: bytes) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    # Pre-seed an indexed document so the list renders without uploading.
    from src.modules.rag.model import RAGDocument
    from src.modules.rag.store import VectorIndex

    docs = {"docA": RAGDocument(document_id="docA", filename="guide.md", file_type="md", character_count=100, chunk_count=1, metadata={"filename": "guide.md"})}
    at.session_state["rag_documents"] = docs
    at.session_state["rag_chunks"] = {}
    at.session_state["rag_index"] = VectorIndex()
    at.run()

    assert not at.exception
    assert any("guide.md" in m.value for m in at.markdown), "indexed document not rendered"
    assert any("Remove" in b.label for b in at.button), "remove button missing"
    assert any("Clear all" in b.label for b in at.button), "clear-all button missing"


# ---- (46-48) UI answer / source / insufficient rendering -----------


def test_rag_ask_renders_answer_and_sources(
    app_path: str, rag_index_careeros, rag_grounded_json, monkeypatch
) -> None:
    """With a Gemini key (mocked via env) + pre-seeded answer, the answer
    + sources render. We pre-seed rag_last_answer to test rendering."""
    idx, doc_id, filename = rag_index_careeros
    # Pre-seed the index + a last answer + a document (so the Ask tab renders).
    from src.modules.rag.model import RAGAnswer, RAGDocument, RAGSource

    last = RAGAnswer(
        answer="The project uses Python and Streamlit. [1]",
        grounded=True,
        sources=[RAGSource(source_id="1", filename=filename, page=1, chunk_index=0)],
        query="What technologies does CareerOS use?",
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.session_state["rag_index"] = idx
    at.session_state["rag_documents"] = {
        doc_id: RAGDocument(document_id=doc_id, filename=filename, file_type="txt", character_count=50, chunk_count=1, metadata={"filename": filename})
    }
    at.session_state["rag_chunks"] = {}
    at.session_state["rag_last_answer"] = last
    at.run()

    assert not at.exception
    assert any("Python and Streamlit" in m.value for m in at.markdown), "answer not rendered"
    # Source citation must render.
    assert any("careeros.txt" in m.value for m in at.markdown), "source not rendered"
    # Grounded badge.
    assert any("Grounded" in s.value for s in at.success)


def test_rag_renders_insufficient_context_state(app_path: str, rag_index_careeros) -> None:
    from src.modules.rag.model import RAGAnswer, RAGDocument

    idx, doc_id, filename = rag_index_careeros
    last = RAGAnswer(
        answer="I don't have enough information in the uploaded documents to answer that confidently.",
        grounded=False,
        insufficient_context=True,
        query="What cloud provider does CareerOS use?",
    )
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.session_state["rag_index"] = idx
    at.session_state["rag_documents"] = {
        doc_id: RAGDocument(document_id=doc_id, filename=filename, file_type="txt", character_count=50, chunk_count=1, metadata={"filename": filename})
    }
    at.session_state["rag_chunks"] = {}
    at.session_state["rag_last_answer"] = last
    at.run()

    assert not at.exception
    assert any(
        "enough information" in w.value.lower() for w in at.warning
    ), "insufficient-context warning missing"


# ---- (43) session-state storage ------------------------------------


def test_rag_initializes_session_state(app_path: str) -> None:
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.run()
    assert not at.exception
    for key in ("rag_documents", "rag_chunks", "rag_index", "rag_messages", "rag_last_answer"):
        assert key in at.session_state, f"{key} not initialized"


def test_rag_does_not_overwrite_existing_state(app_path: str, sample_md_bytes: bytes) -> None:
    from src.modules.rag.model import RAGDocument
    from src.modules.rag.store import VectorIndex

    docs = {"docA": RAGDocument(document_id="docA", filename="g.md", file_type="md", character_count=10, chunk_count=1, metadata={})}
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.session_state["rag_documents"] = docs
    at.session_state["rag_index"] = VectorIndex()
    at.session_state["resume_analysis"] = "PRESERVED"
    at.run()
    assert not at.exception
    assert at.session_state["rag_documents"] is docs  # not overwritten
    assert at.session_state["resume_analysis"] == "PRESERVED"  # other modules untouched


# ---- no-API-key behavior -------------------------------------------


def test_rag_renders_without_gemini_key(app_path: str, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(app_path, default_timeout=15)
    at.session_state["selected_module"] = "rag"
    at.run()
    assert not at.exception
    # The uploader + empty state render without a key.
    assert len(at.file_uploader) == 1
