"""Retrieval — embed query, search the vector index, apply threshold.

Pipeline stages: RETRIEVAL -> RERANK/SCORE.

The minimum relevance threshold (default 0.45) is tuned for the
:class:`LocalEmbedder` score distribution (lexical n-gram cosine). With a
semantic embedding model, a lower threshold (0.2-0.3) would be appropriate.

Defense A (hallucination defense): if NO chunk clears the threshold,
retrieval returns an empty list — the caller (assistant) must NOT call
Gemini and must return the insufficient-context response.
"""

from __future__ import annotations

from typing import Final, Optional

from src.modules.rag.embedder import LocalEmbedder
from src.modules.rag.model import RetrievalResult
from src.modules.rag.store import VectorIndex

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_THRESHOLD: Final[float] = 0.45


def retrieve(
    query: str,
    index: VectorIndex,
    embedder: LocalEmbedder,
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    document_filter: Optional[list[str]] = None,
) -> list[RetrievalResult]:
    """Embed the query, search the index, filter by threshold, rank.

    Returns ``RetrievalResult`` objects sorted by score desc, with 1-based
    ranks. Results below ``threshold`` are dropped (defense A).
    """
    if not query or not query.strip() or len(index) == 0:
        return []
    qv = embedder.embed(query)
    hits = index.retrieve(
        qv, top_k=top_k, document_filter=document_filter
    )
    results: list[RetrievalResult] = []
    rank = 0
    for chunk, score in hits:
        if score < threshold:
            continue
        rank += 1
        results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
    return results
