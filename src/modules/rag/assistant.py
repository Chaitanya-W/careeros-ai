"""RAG assistant — orchestrates retrieval -> context -> Gemini -> validation.

Pipeline stages: RETRIEVAL -> CONTEXT BUILDING -> GEMINI -> GROUNDED ANSWER
-> SOURCE CITATIONS.

Reuses the existing :class:`GeminiClient` (Step 3) — no second client.

Hallucination defense (A-E):
- A. No relevant retrieval -> NO Gemini call -> insufficient-context answer.
- B. Retrieved context -> Gemini answer -> validate cited source IDs.
- C. Gemini cites a source ID not supplied -> reject (mark ungrounded).
- D. Gemini returns malformed output -> safe error.
- E. Gemini returns an empty answer -> safe fallback.

Citations are constructed from retrieval metadata (NEVER trusted from
Gemini). The user-facing source list is built by the context builder; the
assistant only validates that Gemini's cited IDs are a subset of the
supplied IDs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.modules.rag.context import build_context
from src.modules.rag.embedder import LocalEmbedder
from src.modules.rag.model import (
    Confidence,
    RAGAnswer,
    RAGError,
    RAGParseError,
)
from src.modules.rag.prompt import (
    RAG_ANSWER_SCHEMA,
    RAG_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
)
from src.modules.rag.retrieval import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    retrieve,
)
from src.modules.rag.store import VectorIndex
from src.services.gemini import GeminiClient, GeminiError

_logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE: str = (
    "I don't have enough information in the uploaded documents to answer "
    "that confidently."
)

__all__ = [
    "ask_documents",
    "summarize_document",
    "INSUFFICIENT_CONTEXT_MESSAGE",
]


# --------------------------------------------------------------------------- #
# Ask Documents
# --------------------------------------------------------------------------- #


def ask_documents(
    query: str,
    index: VectorIndex,
    embedder: LocalEmbedder,
    *,
    client: Optional[GeminiClient] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    document_filter: Optional[list[str]] = None,
) -> RAGAnswer:
    """Answer a question from the indexed documents.

    Defense A: if retrieval is insufficient (no chunk clears the
    threshold), return the insufficient-context answer WITHOUT calling
    Gemini.
    """
    results = retrieve(
        query,
        index,
        embedder,
        top_k=top_k,
        threshold=threshold,
        document_filter=document_filter,
    )

    # Defense A: no relevant retrieval -> no Gemini call.
    if not results:
        return RAGAnswer(
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            confidence=Confidence.LOW,
            grounded=False,
            query=query,
            retrieved_chunks=[],
            insufficient_context=True,
        )

    context, sources = build_context(results)
    valid_ids = {s.source_id for s in sources}

    if client is None:
        # No Gemini configured: return a deterministic retrieval-only answer.
        return RAGAnswer(
            answer=(
                "Retrieved relevant context (AI synthesis requires Gemini "
                "configuration). See the sources below."
            ),
            confidence=_confidence(results),
            grounded=False,
            sources=sources,
            query=query,
            retrieved_chunks=results,
            warnings=[
                "AI answer synthesis requires Gemini configuration."
            ],
        )

    # Defense D: malformed Gemini output raises RAGParseError (caught by caller).
    raw = client.generate_json(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_text=_build_user_prompt(query, context),
        response_schema=RAG_ANSWER_SCHEMA,
    )
    parsed = _safe_parse(raw)

    answer_text = _clean_str(parsed.get("answer"))
    cited_ids = _extract_cited_ids(parsed, answer_text)

    # Defense C: reject citations not in the supplied source set.
    invalid = [c for c in cited_ids if c not in valid_ids]

    # Defense E: empty answer -> safe fallback.
    if not answer_text.strip():
        return RAGAnswer(
            answer="",
            confidence=Confidence.LOW,
            grounded=False,
            sources=sources,
            query=query,
            retrieved_chunks=results,
            insufficient_context=True,
            warnings=["Gemini returned an empty answer."],
        )

    if invalid:
        # The answer cites invalid sources -> mark ungrounded, keep answer
        # text but clearly untrusted, with the invalid IDs listed.
        return RAGAnswer(
            answer=answer_text,
            confidence=Confidence.LOW,
            grounded=False,
            sources=sources,
            query=query,
            retrieved_chunks=results,
            warnings=[
                "Gemini cited invalid source IDs: "
                + ", ".join(invalid)
                + ". The answer is not trusted."
            ],
        )

    if parsed.get("insufficient_context"):
        return RAGAnswer(
            answer=answer_text or INSUFFICIENT_CONTEXT_MESSAGE,
            confidence=Confidence.LOW,
            grounded=False,
            sources=sources,
            query=query,
            retrieved_chunks=results,
            insufficient_context=True,
        )

    cited_sources = [s for s in sources if s.source_id in set(cited_ids)] or sources
    return RAGAnswer(
        answer=answer_text,
        confidence=_confidence(results),
        grounded=True,
        sources=cited_sources,
        query=query,
        retrieved_chunks=results,
    )


# --------------------------------------------------------------------------- #
# Summarize Document
# --------------------------------------------------------------------------- #


def summarize_document(
    document_id: str,
    filename: str,
    index: VectorIndex,
    embedder: LocalEmbedder,
    *,
    client: Optional[GeminiClient] = None,
    top_k: int = 8,
    threshold: float = 0.1,
) -> RAGAnswer:
    """Summarize a single document by retrieving its top chunks.

    Uses a low threshold (the query is the document's own title/topic) and
    a document_filter restricted to the single document.
    """
    # Use the document's own chunks as retrieval (broad recall within the doc).
    results = retrieve(
        filename,
        index,
        embedder,
        top_k=top_k,
        threshold=threshold,
        document_filter=[document_id],
    )
    if not results:
        return RAGAnswer(
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            confidence=Confidence.LOW,
            grounded=False,
            query=f"summarize:{filename}",
            retrieved_chunks=[],
            insufficient_context=True,
        )
    context, sources = build_context(results)
    valid_ids = {s.source_id for s in sources}

    if client is None:
        return RAGAnswer(
            answer=(
                "Retrieved document context (AI summary requires Gemini "
                "configuration). See the sources below."
            ),
            confidence=_confidence(results),
            grounded=False,
            sources=sources,
            query=f"summarize:{filename}",
            retrieved_chunks=results,
            warnings=["AI summary requires Gemini configuration."],
        )

    raw = client.generate_json(
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        user_text=_build_user_prompt(f"Summarize the document '{filename}'.", context),
        response_schema=RAG_ANSWER_SCHEMA,
    )
    parsed = _safe_parse(raw)
    answer_text = _clean_str(parsed.get("answer"))
    cited_ids = _extract_cited_ids(parsed, answer_text)
    invalid = [c for c in cited_ids if c not in valid_ids]

    if not answer_text.strip():
        return RAGAnswer(
            answer="",
            confidence=Confidence.LOW,
            grounded=False,
            sources=sources,
            query=f"summarize:{filename}",
            retrieved_chunks=results,
            insufficient_context=True,
            warnings=["Gemini returned an empty summary."],
        )

    if invalid:
        return RAGAnswer(
            answer=answer_text,
            confidence=Confidence.LOW,
            grounded=False,
            sources=sources,
            query=f"summarize:{filename}",
            retrieved_chunks=results,
            warnings=[
                "Gemini cited invalid source IDs: " + ", ".join(invalid) + "."
            ],
        )

    cited_sources = [s for s in sources if s.source_id in set(cited_ids)] or sources
    return RAGAnswer(
        answer=answer_text,
        confidence=_confidence(results),
        grounded=True,
        sources=cited_sources,
        query=f"summarize:{filename}",
        retrieved_chunks=results,
    )


# --------------------------------------------------------------------------- #
# Confidence heuristic (documented, NOT scientifically calibrated)
# --------------------------------------------------------------------------- #


def _confidence(results) -> Confidence:
    """Heuristic confidence from retrieval evidence.

    - HIGH: top score >= 0.6 AND >= 3 retrieved chunks.
    - MEDIUM: top score >= 0.45 OR >= 2 retrieved chunks.
    - LOW: otherwise.
    """
    if not results:
        return Confidence.LOW
    top = results[0].score
    n = len(results)
    if top >= 0.6 and n >= 3:
        return Confidence.HIGH
    if top >= 0.45 or n >= 2:
        return Confidence.MEDIUM
    return Confidence.LOW


# --------------------------------------------------------------------------- #
# Prompt building + parsing + citation extraction
# --------------------------------------------------------------------------- #


def _build_user_prompt(query: str, context: str) -> str:
    return (
        f"Context (use ONLY this information; cite [N] source IDs):\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )


def _safe_parse(raw: str) -> dict:
    if not raw or not raw.strip():
        raise RAGParseError("Gemini returned an empty response.")
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise RAGParseError(
            f"Gemini response was not valid JSON ({e.msg} at line {e.lineno} col {e.colno})."
        )
    if not isinstance(data, dict):
        raise RAGParseError("Gemini response was not a JSON object.")
    return data


def _extract_cited_ids(parsed: dict, answer_text: str) -> list[str]:
    """Collect cited source IDs from the schema field AND inline [N] refs."""
    ids: list[str] = []
    seen: set[str] = set()
    for c in parsed.get("cited_source_ids") or []:
        if isinstance(c, str):
            cid = c.strip()
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    # Also scan the answer text for inline [N] references.
    for m in re.finditer(r"\[([^\]]+)\]", answer_text):
        cid = m.group(1).strip()
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _clean_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
