"""Document ingestion for the RAG pipeline (PDF / DOCX / TXT / MD).

Decoupled from Resume Intelligence: this module has its own supported-type
set (adds Markdown) and returns **page-aware sections** so chunking can tag
chunks with page numbers. Reuses the same lightweight libraries (pypdf,
python-docx) via lazy imports, but is an independent implementation.

Pipeline stage: DOCUMENT -> INGESTION -> TEXT EXTRACTION -> NORMALIZATION.

Normalization collapses whitespace and repeated blank lines and strips
obvious extraction artifacts WITHOUT changing semantic meaning.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Final, Optional

SUPPORTED_TYPES: Final[frozenset[str]] = frozenset({"pdf", "docx", "txt", "md"})

# Maximum accepted file size (bytes). Tunable; keeps the demo fast.
MAX_FILE_SIZE: Final[int] = 5_000_000


class IngestionError(Exception):
    """Base error for document ingestion problems."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when the uploaded file type is not supported."""


class EmptyDocumentError(IngestionError):
    """Raised when the document has no readable content."""


class MalformedDocumentError(IngestionError):
    """Raised when the document is corrupted / unparseable."""


class TooLargeDocumentError(IngestionError):
    """Raised when the document exceeds the size limit."""


@dataclass
class DocumentSection:
    """A page-bounded section of extracted text (page is None for non-PDF)."""

    page: Optional[int]
    text: str


def get_extension(file_name: str) -> str:
    if not file_name:
        return ""
    name = file_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def document_fingerprint(file_bytes: bytes, filename: str = "") -> str:
    """Deterministic document ID from file bytes (sha256, 16-char prefix).

    Two uploads of the exact same bytes produce the same ID, enabling
    duplicate detection.
    """
    h = hashlib.sha256()
    h.update(file_bytes)
    if filename:
        h.update(filename.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def extract_sections(file_name: str, file_bytes: bytes) -> list[DocumentSection]:
    """Extract + normalize text, returning page-aware sections.

    Raises typed errors for unsupported / empty / malformed / too-large files.
    """
    ext = get_extension(file_name)
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. Supported: PDF, DOCX, TXT, MD."
        )
    if not file_bytes:
        raise EmptyDocumentError("The uploaded document is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise TooLargeDocumentError(
            f"Document too large (max {MAX_FILE_SIZE:,} bytes)."
        )

    if ext == "pdf":
        sections = _extract_pdf_sections(file_bytes)
    elif ext == "docx":
        text = _extract_docx(file_bytes)
        sections = [DocumentSection(page=None, text=text)]
    else:  # txt / md
        text = _extract_txt(file_bytes)
        sections = [DocumentSection(page=None, text=text)]

    # Normalize each section; drop empty ones.
    out: list[DocumentSection] = []
    for s in sections:
        norm = normalize(s.text)
        if norm:
            out.append(DocumentSection(page=s.page, text=norm))
    if not out:
        raise EmptyDocumentError("No readable text could be extracted from the document.")
    return out


def normalize(text: str) -> str:
    """Normalize whitespace without changing semantic meaning.

    - Collapses runs of spaces/tabs to a single space.
    - Collapses 3+ newlines to 2 (paragraph breaks).
    - Strips leading/trailing whitespace per line and overall.
    - Removes common PDF extraction artifacts (e.g. form-feed \f, NULs).
    """
    if not text:
        return ""
    # Remove NULs and form-feeds (common PDF artifacts).
    t = text.replace("\x00", " ").replace("\f", "\n")
    # Normalize line endings.
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line.
    lines = [ln.rstrip() for ln in t.split("\n")]
    t = "\n".join(lines)
    # Collapse runs of spaces/tabs (not newlines) to a single space.
    t = re.sub(r"[ \t]+", " ", t)
    # Collapse 3+ newlines to 2 (preserve paragraph breaks).
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def derive_title(filename: str, first_text: str) -> str:
    """Best-effort title: first non-empty line of text, else filename."""
    if first_text:
        first_line = first_text.strip().split("\n", 1)[0].strip()
        if first_line and len(first_line) <= 200:
            return first_line
    return filename


# --------------------------------------------------------------------------- #
# Format-specific extractors (lazy imports).
# --------------------------------------------------------------------------- #


def _extract_txt(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return text


def _extract_pdf_sections(data: bytes) -> list[DocumentSection]:
    """Extract text per page so chunks can carry page numbers."""
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency declared
        raise IngestionError(
            "PDF ingestion requires the 'pypdf' package."
        ) from e
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise MalformedDocumentError(
            "The PDF file appears to be malformed or corrupted."
        ) from e
    if not reader.pages:
        raise EmptyDocumentError("The PDF has no pages.")
    sections: list[DocumentSection] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            sections.append(DocumentSection(page=page_num, text=page_text))
    if not sections:
        raise EmptyDocumentError("No text could be extracted from the PDF.")
    return sections


def _extract_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:  # pragma: no cover - dependency declared
        raise IngestionError(
            "DOCX ingestion requires the 'python-docx' package."
        ) from e
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise MalformedDocumentError(
            "The DOCX file appears to be malformed or corrupted."
        ) from e
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text.strip():
        raise EmptyDocumentError("The DOCX contains no readable text.")
    return text
