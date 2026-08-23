"""Deterministic text chunking for the RAG pipeline.

Pipeline stage: NORMALIZATION -> CHUNKING.

Strategy (deterministic, same input -> same chunks):
1. Split text into paragraphs (blank-line separated).
2. If a paragraph fits within ``chunk_size``, keep it as a chunk candidate;
   small adjacent paragraphs are merged up to ``chunk_size``.
3. If a paragraph exceeds ``chunk_size``, split it into sentences and merge
   sentences up to ``chunk_size``.
4. If a single sentence exceeds ``chunk_size``, split it by characters with
   ``chunk_overlap``.
5. Apply ``chunk_overlap`` between consecutive chunks (the last
   ``overlap`` chars of the previous chunk prefix the next).

No empty chunks are ever produced. Chunk IDs are assigned by the caller
(stable: ``<doc_id>-C<index:03d>``).
"""

from __future__ import annotations

import re
from typing import Final

DEFAULT_CHUNK_SIZE: Final[int] = 800  # characters
DEFAULT_CHUNK_OVERLAP: Final[int] = 100

# Sentence-end pattern: . ! ? followed by whitespace/newline.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Paragraph split: 2+ newlines.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Chunk ``text`` into non-empty, deterministic chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    raw_chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(_split_large_paragraph(para, chunk_size))

    # Merge small adjacent chunks up to chunk_size for fuller context.
    merged: list[str] = []
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
        if merged and len(merged[-1]) + 1 + len(chunk) <= chunk_size:
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)

    # Apply overlap between consecutive chunks.
    final = _apply_overlap(merged, chunk_overlap)
    # Final guard: drop any empty chunks.
    return [c for c in final if c.strip()]


def make_chunk_id(document_id: str, index: int) -> str:
    """Stable chunk ID: ``<doc_id>-C001``."""
    return f"{document_id}-C{index + 1:03d}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    return paras


def _split_large_paragraph(para: str, chunk_size: int) -> list[str]:
    """Split a too-large paragraph by sentences, then by chars if needed."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip()]
    out: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > chunk_size:
            # Flush buffer first.
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_by_chars(sent, chunk_size))
            continue
        if buf and len(buf) + 1 + len(sent) <= chunk_size:
            buf = buf + " " + sent
        else:
            if buf:
                out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def _split_by_chars(text: str, chunk_size: int) -> list[str]:
    """Split a single long string by character boundaries."""
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += chunk_size
    return out


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix each chunk (after the first) with the last ``overlap`` chars
    of the previous chunk. Does not duplicate when chunks are tiny."""
    if overlap <= 0 or len(chunks) <= 1:
        return list(chunks)
    out: list[str] = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        prefix = prev[-overlap:] if len(prev) >= overlap else prev
        # Avoid silly duplication when prefix == start of cur.
        if cur.startswith(prefix):
            out.append(cur)
        else:
            out.append(prefix + cur)
    return out
