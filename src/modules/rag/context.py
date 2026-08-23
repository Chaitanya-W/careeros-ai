"""Context builder — assemble a size-limited, source-numbered context.

Pipeline stage: CONTEXT BUILDING.

Builds a numbered context string from retrieval results so Gemini can cite
``[1]``, ``[2]``, etc. Also builds the :class:`RAGSource` list (from
retrieval metadata — NEVER trusted from Gemini) so citations always map to
real retrieved chunks.

The total context size is capped (``MAX_CONTEXT_CHARS``) so only the
relevant chunks (not entire documents) are sent to Gemini.
"""

from __future__ import annotations

from typing import Final

from src.modules.rag.model import RAGSource, RetrievalResult

MAX_CONTEXT_CHARS: Final[int] = 4000


def build_context(
    results: list[RetrievalResult],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[RAGSource]]:
    """Return ``(context_string, sources)`` from retrieval results.

    Sources are numbered ``1..N`` matching the context citations. Each
    source's metadata (filename, page, chunk_index) comes from the
    retrieved chunk — not from Gemini.
    """
    context_parts: list[str] = []
    sources: list[RAGSource] = []
    total = 0
    for i, r in enumerate(results, start=1):
        snippet = r.chunk.text or ""
        if total + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - total)]
        if not snippet:
            continue
        meta = r.chunk.metadata or {}
        page = meta.get("page")
        page_str = f", page {page}" if isinstance(page, int) else ""
        context_parts.append(
            f"[{i}] (document: {meta.get('filename', 'unknown')}{page_str}, "
            f"chunk {r.chunk.chunk_index})\n{snippet}"
        )
        sources.append(
            RAGSource(
                source_id=str(i),
                document_id=r.chunk.document_id,
                filename=meta.get("filename", "unknown"),
                page=page if isinstance(page, int) else None,
                chunk_index=r.chunk.chunk_index,
                text=snippet,
            )
        )
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n\n".join(context_parts), sources


def format_citations(sources: list[RAGSource]) -> list[str]:
    """Human-readable citation strings, e.g. ``[1] Docker_Guide.pdf — page 4``."""
    out: list[str] = []
    for s in sources:
        loc = f"page {s.page}" if s.page is not None else f"chunk {s.chunk_index}"
        out.append(f"[{s.source_id}] {s.filename} — {loc}")
    return out
