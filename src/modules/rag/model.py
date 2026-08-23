"""RAG Career Assistant data models.

Defines the structured models for the retrieval-augmented-generation
pipeline:

- :class:`RAGDocument` — an ingested document (id, filename, type, counts).
- :class:`DocumentChunk` — one chunk with stable ID + page/chunk metadata.
- :class:`RetrievalResult` — a ranked chunk + similarity score.
- :class:`RAGSource` — a citation source built from retrieval metadata.
- :class:`RAGAnswer` — the final grounded answer + sources + retrieval.
- :class:`Confidence` — constrained HIGH / MEDIUM / LOW.

All models are robust to missing information (``from_dict`` tolerates
absent keys / wrong types).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class RAGError(Exception):
    """Base error for RAG model / parsing problems."""


class RAGParseError(RAGError):
    """Raised when a model response cannot be parsed into the schema."""


class Confidence(str, Enum):
    """Constrained confidence for a RAG answer (heuristic, not calibrated)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_value(cls, value: Any) -> "Confidence":
        if isinstance(value, Confidence):
            return value
        if value is None:
            return Confidence.LOW
        s = str(value).strip().lower()
        for member in cls:
            if member.value == s:
                return member
        return Confidence.LOW


@dataclass
class RAGDocument:
    """An ingested document."""

    document_id: str = ""
    filename: str = ""
    file_type: str = ""  # pdf / docx / txt / md
    title: str = ""
    character_count: int = 0
    chunk_count: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "RAGDocument":
        if not isinstance(data, dict):
            raise RAGParseError("RAG document was not a JSON object.")
        return cls(
            document_id=_clean_str(data.get("document_id")),
            filename=_clean_str(data.get("filename")),
            file_type=_clean_str(data.get("file_type")),
            title=_clean_str(data.get("title")),
            character_count=_coerce_int(data.get("character_count")),
            chunk_count=_coerce_int(data.get("chunk_count")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentChunk:
    """One chunk of an ingested document."""

    chunk_id: str = ""
    document_id: str = ""
    chunk_index: int = 0
    text: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "DocumentChunk":
        if not isinstance(data, dict):
            raise RAGParseError("Document chunk was not a JSON object.")
        return cls(
            chunk_id=_clean_str(data.get("chunk_id")),
            document_id=_clean_str(data.get("document_id")),
            chunk_index=_coerce_int(data.get("chunk_index")),
            text=_clean_str(data.get("text")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """A ranked retrieval hit."""

    chunk: DocumentChunk = field(default_factory=DocumentChunk)
    score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


@dataclass
class RAGSource:
    """A citation source built from retrieval metadata (not trusted from LLM)."""

    source_id: str = ""  # "1", "2", ...
    document_id: str = ""
    filename: str = ""
    page: Optional[int] = None
    chunk_index: int = 0
    text: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "RAGSource":
        if not isinstance(data, dict):
            raise RAGParseError("RAG source was not a JSON object.")
        page = data.get("page")
        return cls(
            source_id=_clean_str(data.get("source_id")),
            document_id=_clean_str(data.get("document_id")),
            filename=_clean_str(data.get("filename")),
            page=int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
            chunk_index=_coerce_int(data.get("chunk_index")),
            text=_clean_str(data.get("text")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGAnswer:
    """A retrieval-grounded answer with sources and validation state."""

    answer: str = ""
    confidence: Confidence = Confidence.LOW
    grounded: bool = False
    sources: list[RAGSource] = field(default_factory=list)
    query: str = ""
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    insufficient_context: bool = False
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "RAGAnswer":
        if not isinstance(data, dict):
            raise RAGParseError("RAG answer was not a JSON object.")
        return cls(
            answer=_clean_str(data.get("answer")),
            confidence=Confidence.from_value(data.get("confidence")),
            grounded=bool(data.get("grounded", False)),
            sources=[RAGSource.from_dict(s) for s in _as_list(data.get("sources"))],
            query=_clean_str(data.get("query")),
            retrieved_chunks=[
                RetrievalResult(
                    chunk=DocumentChunk.from_dict(r.get("chunk")) if isinstance(r, dict) and isinstance(r.get("chunk"), dict) else DocumentChunk(),
                    score=float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0,
                    rank=int(r.get("rank", 0)) if isinstance(r, dict) else 0,
                )
                for r in _as_list(data.get("retrieved_chunks"))
                if isinstance(r, dict)
            ],
            insufficient_context=bool(data.get("insufficient_context", False)),
            warnings=_coerce_str_list(data.get("warnings")),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = self.confidence.value
        d["sources"] = [s.to_dict() for s in self.sources]
        d["retrieved_chunks"] = [r.to_dict() for r in self.retrieved_chunks]
        return d


# --------------------------------------------------------------------------- #
# Coercion helpers (module-private).
# --------------------------------------------------------------------------- #


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_str_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
