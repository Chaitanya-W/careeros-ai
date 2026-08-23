"""RAG Career Assistant UI workflow.

Tabs: Ask Documents / Summarize Document.

Flow: upload documents -> index (chunk + embed + vector store) -> choose
scope (all / one document) -> ask a question -> retrieval-grounded answer
+ source citations. Includes insufficient-context handling, invalid
citation warnings, duplicate detection, document removal, and clear-all.

State keys: rag_documents / rag_chunks / rag_index / rag_messages /
rag_last_answer. The existing CareerOS state is never overwritten.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.rag.assistant import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    ask_documents,
    summarize_document,
)
from src.modules.rag.chunker import chunk_text, make_chunk_id
from src.modules.rag.context import format_citations
from src.modules.rag.embedder import LocalEmbedder
from src.modules.rag.ingestion import (
    IngestionError,
    document_fingerprint,
    extract_sections,
)
from src.modules.rag.model import RAGAnswer, RAGDocument, DocumentChunk
from src.modules.rag.retrieval import DEFAULT_THRESHOLD, DEFAULT_TOP_K
from src.modules.rag.store import VectorIndex
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while processing your documents. Please try again."
)


def render() -> None:
    """Render the RAG Career Assistant workflow."""
    section_header(
        "RAG Career Assistant",
        "Ask questions about your uploaded career documents and receive "
        "source-grounded answers.",
    )
    _ensure_state()
    _render_upload()
    _render_document_list()
    _render_scope_and_query()


# --------------------------------------------------------------------------- #
# State initialization
# --------------------------------------------------------------------------- #


def _ensure_state() -> None:
    if not isinstance(st.session_state.get("rag_documents"), dict):
        st.session_state["rag_documents"] = {}
    if not isinstance(st.session_state.get("rag_chunks"), dict):
        st.session_state["rag_chunks"] = {}
    if not isinstance(st.session_state.get("rag_index"), VectorIndex):
        st.session_state["rag_index"] = VectorIndex()
    if not isinstance(st.session_state.get("rag_messages"), list):
        st.session_state["rag_messages"] = []
    if "rag_last_answer" not in st.session_state:
        st.session_state["rag_last_answer"] = None
    if "rag_embedder" not in st.session_state:
        st.session_state["rag_embedder"] = LocalEmbedder()


def _embedder() -> LocalEmbedder:
    return st.session_state["rag_embedder"]


def _index() -> VectorIndex:
    return st.session_state["rag_index"]


def _documents() -> dict:
    return st.session_state["rag_documents"]


def _chunks() -> dict:
    return st.session_state["rag_chunks"]


# --------------------------------------------------------------------------- #
# Upload + index
# --------------------------------------------------------------------------- #


def _render_upload() -> None:
    section_header("Upload documents", "PDF, DOCX, TXT, or MD — indexed locally.")
    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt", "md"],
        key="rag_uploader",
        help="Documents are extracted, chunked, and embedded locally (no paid API).",
    )
    if uploaded is None:
        return
    _index_upload(uploaded)


def _index_upload(uploaded) -> None:
    name = uploaded.name
    try:
        data = uploaded.getvalue()
    except Exception:
        st.error("Could not read the uploaded file.")
        return
    try:
        sections = extract_sections(name, data)
    except IngestionError as e:
        st.error(str(e))
        return
    except Exception:
        _logger.exception("Unexpected ingestion error")
        st.error(_UNEXPECTED_ERROR_MESSAGE)
        return

    doc_id = document_fingerprint(data, name)
    if doc_id in _documents():
        st.info(f"Document already indexed: {name}.")
        return

    # Chunk each section, tagging chunks with page/chunk metadata.
    full_text = "\n\n".join(s.text for s in sections)
    raw_chunks = chunk_text(full_text)
    chunk_objs: list[DocumentChunk] = []
    # Approximate page mapping: distribute chunks across pages by char ratio.
    page_map = _build_page_map(sections, len(full_text))
    for i, text in enumerate(raw_chunks):
        page = _page_for_chunk(i, len(raw_chunks), page_map)
        chunk_objs.append(
            DocumentChunk(
                chunk_id=make_chunk_id(doc_id, i),
                document_id=doc_id,
                chunk_index=i,
                text=text,
                metadata={
                    "filename": name,
                    "page": page,
                    "chunk_index": i,
                    "document_id": doc_id,
                },
            )
        )

    embedder = _embedder()
    index = _index()
    for ch in chunk_objs:
        index.add(ch, embedder.embed(ch.text))

    title = sections[0].text.strip().split("\n", 1)[0][:120] if sections else name
    doc = RAGDocument(
        document_id=doc_id,
        filename=name,
        file_type=_file_type(name),
        title=title,
        character_count=len(full_text),
        chunk_count=len(chunk_objs),
        metadata={"filename": name},
    )
    _documents()[doc_id] = doc
    _chunks()[doc_id] = chunk_objs
    st.success(f"Indexed '{name}': {len(chunk_objs)} chunk(s), {len(full_text):,} characters.")
    st.rerun()


def _build_page_map(sections, total_chars):
    """Map chunk-ratio boundaries to page numbers (best-effort)."""
    if not sections or total_chars == 0:
        return []
    boundaries = []
    cum = 0
    for s in sections:
        cum += len(s.text)
        boundaries.append((cum / total_chars, s.page))
    return boundaries


def _page_for_chunk(i, n, page_map):
    if not page_map:
        return None
    ratio = (i + 0.5) / n if n else 0
    page = None
    for bound, pg in page_map:
        if ratio <= bound:
            page = pg
            break
    return page


def _file_type(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


# --------------------------------------------------------------------------- #
# Document list + management
# --------------------------------------------------------------------------- #


def _render_document_list() -> None:
    docs = _documents()
    if not docs:
        empty_state_panel(
            title="No documents indexed yet",
            hint="Upload a PDF, DOCX, TXT, or MD document above to start.",
        )
        return
    section_header("Indexed documents", f"{len(docs)} document(s).")
    for doc_id, doc in list(docs.items()):
        with st.container(border=True):
            col_info, col_actions = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{doc.filename}**")
                st.caption(
                    f"{doc.file_type.upper()} · {doc.chunk_count} chunk(s) · "
                    f"{doc.character_count:,} chars"
                )
            with col_actions:
                if st.button("Remove", key=f"rag_remove_{doc_id}"):
                    _remove_document(doc_id)
                    st.rerun()
    if st.button("Clear all documents", key="rag_clear_all"):
        _clear_all()
        st.rerun()


def _remove_document(doc_id: str) -> None:
    _index().remove_document(doc_id)
    _documents().pop(doc_id, None)
    _chunks().pop(doc_id, None)
    st.session_state["rag_last_answer"] = None


def _clear_all() -> None:
    _index().clear()
    _documents().clear()
    _chunks().clear()
    st.session_state["rag_messages"] = []
    st.session_state["rag_last_answer"] = None


# --------------------------------------------------------------------------- #
# Scope + query (Ask / Summarize)
# --------------------------------------------------------------------------- #


def _render_scope_and_query() -> None:
    docs = _documents()
    if not docs:
        return
    section_header("Ask the documents", "Choose a scope and ask a question, or summarize.")
    options = ["All documents"] + [d.filename for d in docs.values()]
    scope = st.radio(
        "Document scope",
        options,
        horizontal=True,
        key="rag_scope",
        help="Restrict retrieval to the selected document (or all).",
    )
    doc_filter: Optional[list[str]] = None
    if scope != "All documents":
        for doc_id, doc in docs.items():
            if doc.filename == scope:
                doc_filter = [doc_id]
                break

    tab_ask, tab_summarize = st.tabs(["Ask Documents", "Summarize Document"])
    with tab_ask:
        _render_ask(doc_filter)
    with tab_summarize:
        _render_summarize(doc_filter, docs)


def _render_ask(doc_filter: Optional[list[str]]) -> None:
    query = st.text_input(
        "Question",
        key="rag_question_input",
        placeholder="e.g. What technologies does the document mention?",
    )
    if st.button("Ask", type="primary", key="rag_ask_btn", disabled=not (query and query.strip())):
        _run_ask(query, doc_filter)

    last = st.session_state.get("rag_last_answer")
    if last is not None:
        _render_answer(last)
    else:
        empty_state_panel(
            title="No answer yet",
            hint="Ask a question to get a retrieval-grounded answer with sources.",
        )


def _render_summarize(doc_filter: Optional[list[str]], docs: dict) -> None:
    if doc_filter is None:
        st.info("Select a single document in the scope above to summarize it.")
        return
    doc_id = doc_filter[0]
    doc = docs.get(doc_id)
    if doc is None:
        st.info("Select a single document to summarize.")
        return
    if st.button("Summarize document", type="primary", key="rag_summarize_btn"):
        _run_summarize(doc_id, doc.filename)


def _run_ask(query: str, doc_filter: Optional[list[str]]) -> None:
    with st.spinner("Retrieving + generating…"):
        try:
            client = _maybe_gemini_client()
            answer = ask_documents(
                query,
                _index(),
                _embedder(),
                client=client,
                top_k=DEFAULT_TOP_K,
                threshold=DEFAULT_THRESHOLD,
                document_filter=doc_filter,
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected RAG error")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    st.session_state["rag_last_answer"] = answer
    st.session_state["rag_messages"].append(("user", query))
    st.session_state["rag_messages"].append(("assistant", answer.answer))
    st.rerun()


def _run_summarize(doc_id: str, filename: str) -> None:
    with st.spinner("Summarizing…"):
        try:
            client = _maybe_gemini_client()
            answer = summarize_document(
                doc_id,
                filename,
                _index(),
                _embedder(),
                client=client,
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected RAG summarize error")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    st.session_state["rag_last_answer"] = answer
    st.rerun()


def _maybe_gemini_client():
    """Return a GeminiClient if a key is configured, else None."""
    try:
        from src.services.gemini import get_api_key

        get_api_key()
        from src.services.gemini import GeminiClient

        return GeminiClient()
    except GeminiConfigError:
        return None
    except Exception:
        _logger.exception("Unexpected error while checking Gemini key")
        return None


# --------------------------------------------------------------------------- #
# Answer rendering
# --------------------------------------------------------------------------- #


def _render_answer(answer: RAGAnswer) -> None:
    section_header("Answer", None)
    if answer.insufficient_context:
        st.warning(INSUFFICIENT_CONTEXT_MESSAGE)
    if answer.answer:
        st.write(answer.answer)
    # Grounding badge.
    if answer.grounded:
        st.success("Grounded: answer is backed by retrieved document context.")
    elif answer.warnings:
        for w in answer.warnings:
            st.warning(w)
    elif not answer.insufficient_context:
        st.info("Not grounded: AI synthesis requires Gemini configuration.")
    # Sources.
    if answer.sources:
        st.caption("Sources")
        for cite in format_citations(answer.sources):
            st.markdown(f"- {cite}")
        with st.expander("Source snippets", expanded=False):
            for s in answer.sources:
                st.markdown(f"**[{s.source_id}] {s.filename}**")
                st.write(s.text)
