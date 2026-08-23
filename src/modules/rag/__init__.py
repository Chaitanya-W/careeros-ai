"""RAG Career Assistant module.

Public API: :func:`render` draws the upload -> index -> ask/summarize ->
grounded-answer workflow. Internally delegates to
:mod:`src.modules.rag.ui`.

Submodules
----------
- ``model``      : RAGDocument, DocumentChunk, RetrievalResult, RAGSource, RAGAnswer, Confidence.
- ``ingestion``  : PDF/DOCX/TXT/MD extraction + normalization + fingerprint (decoupled).
- ``chunker``    : deterministic paragraph/sentence/char chunking with overlap.
- ``embedder``   : local deterministic n-gram-hashing embedder (no paid API).
- ``store``      : pure-Python in-memory VectorIndex with document isolation.
- ``retrieval``  : threshold-filtered top-k retrieval (defense A).
- ``context``    : size-limited, source-numbered context builder.
- ``prompt``     : Gemini RAG system prompt + JSON schema.
- ``assistant``  : ask_documents / summarize_document with hallucination defense A-E.
- ``ui``         : Streamlit workflow.
"""

from __future__ import annotations

from src.modules.rag.ui import render

MODULE_NAME: str = "RAG"
MODULE_KEY: str = "rag"
MODULE_DESCRIPTION: str = (
    "Ground AI responses in your uploaded documents for accurate, "
    "citation-backed answers."
)

__all__ = ["render", "MODULE_NAME", "MODULE_KEY", "MODULE_DESCRIPTION"]
