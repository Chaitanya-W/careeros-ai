"""Pure-Python in-memory vector store with document isolation.

Pipeline stage: EMBEDDING -> VECTOR INDEX.

A lightweight, dependency-free vector index. Supports:
- add chunks + vectors,
- retrieve top-k by cosine similarity,
- metadata filtering (document isolation — restrict retrieval to one or
  more ``document_id`` values),
- remove all chunks for a given document (no stale chunks),
- clear.

No secrets are stored (only document text + numeric vectors + metadata).
Persistence is session-scoped (in-memory) — appropriate for a capstone demo.
For larger scale, swap this interface for FAISS/Chroma without changing
callers.
"""

from __future__ import annotations

from typing import Optional

from src.modules.rag.embedder import cosine_similarity
from src.modules.rag.model import DocumentChunk


class VectorIndex:
    """In-memory vector index with document isolation."""

    def __init__(self) -> None:
        self._vectors: list[list[float]] = []
        self._chunks: list[DocumentChunk] = []
        self._doc_ids: list[str] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def document_ids(self) -> list[str]:
        """Distinct document IDs currently in the index."""
        # Preserve insertion order, dedupe.
        seen: set[str] = set()
        out: list[str] = []
        for d in self._doc_ids:
            if d and d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def add(self, chunk: DocumentChunk, vector: list[float]) -> None:
        """Add a chunk + its embedding vector."""
        self._vectors.append(list(vector))
        self._chunks.append(chunk)
        self._doc_ids.append(chunk.document_id)

    def has_document(self, document_id: str) -> bool:
        return document_id in self._doc_ids

    def retrieve(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        document_filter: Optional[list[str]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the top-k ``(chunk, similarity)`` pairs.

        ``document_filter`` restricts retrieval to the given document IDs
        (document isolation). ``None`` means all documents.
        """
        if top_k <= 0 or not self._chunks:
            return []
        filter_set = set(document_filter) if document_filter is not None else None
        scored: list[tuple[DocumentChunk, float]] = []
        for vec, chunk, doc_id in zip(self._vectors, self._chunks, self._doc_ids):
            if filter_set is not None and doc_id not in filter_set:
                continue
            score = cosine_similarity(query_vector, vec)
            scored.append((chunk, score))
        # Sort by score desc, then chunk_id for stable tiebreak.
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks for ``document_id``; return the count removed."""
        keep_idx = [i for i, d in enumerate(self._doc_ids) if d != document_id]
        removed = len(self._chunks) - len(keep_idx)
        self._vectors = [self._vectors[i] for i in keep_idx]
        self._chunks = [self._chunks[i] for i in keep_idx]
        self._doc_ids = [self._doc_ids[i] for i in keep_idx]
        return removed

    def clear(self) -> None:
        self._vectors.clear()
        self._chunks.clear()
        self._doc_ids.clear()

    def chunk_count_for(self, document_id: str) -> int:
        return sum(1 for d in self._doc_ids if d == document_id)
