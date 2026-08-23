"""Resume text extraction for PDF, DOCX, and TXT.

Each format-specific extractor lazily imports its dependency so this
module imports cleanly even when a library is missing — extraction only
fails (with a friendly :class:`ExtractionError`) when that format is
actually used.

Design
------
- :func:`extract_text` validates the type, rejects empty/malformed
  input, and dispatches to the right extractor.
- Errors are typed (``UnsupportedFileTypeError`` / ``EmptyFileError`` /
  ``ExtractionError``) so the UI can show actionable messages instead of
  Python tracebacks.

The extractor NEVER sends binary file contents anywhere; it returns
extracted plain text only.
"""

from __future__ import annotations

import io
from typing import Final

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({"pdf", "docx", "txt"})


class ExtractionError(Exception):
    """Base error for resume text-extraction problems."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when the uploaded file type is not supported."""


class EmptyFileError(ExtractionError):
    """Raised when the file has no readable content."""


def get_extension(file_name: str) -> str:
    """Return the lowercase extension without the dot (e.g. ``'pdf'``)."""
    if not file_name:
        return ""
    name = file_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """Extract readable text from an uploaded resume file.

    Args:
        file_name: original file name (used only to detect the type).
        file_bytes: raw file bytes.

    Returns:
        The extracted, stripped text.

    Raises:
        UnsupportedFileTypeError: extension not in {pdf, docx, txt}.
        EmptyFileError: file is empty or has no extractable text.
        ExtractionError: malformed file or extraction-library failure.
    """
    ext = get_extension(file_name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. "
            f"Supported formats: PDF, DOCX, TXT."
        )
    if not file_bytes:
        raise EmptyFileError("The uploaded file is empty.")

    if ext == "txt":
        return _extract_txt(file_bytes)
    if ext == "pdf":
        return _extract_pdf(file_bytes)
    return _extract_docx(file_bytes)


def _extract_txt(data: bytes) -> str:
    """Decode text bytes; tolerate non-UTF-8 by falling back to latin-1."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    text = text.strip()
    if not text:
        raise EmptyFileError("The text file contains no readable characters.")
    return text


def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF using ``pypdf`` (lazy import)."""
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency declared
        raise ExtractionError(
            "PDF extraction requires the 'pypdf' package. "
            "Install it to analyze PDF resumes."
        ) from e
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:  # malformed / corrupted PDF
        raise ExtractionError(
            "The PDF file appears to be malformed or corrupted."
        ) from e
    if not reader.pages:
        raise EmptyFileError("The PDF has no pages.")
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # Skip unreadable pages but keep going.
            pages.append("")
    text = "\n".join(p for p in pages if p).strip()
    if not text:
        raise ExtractionError(
            "No text could be extracted from the PDF — "
            "it may be a scanned image."
        )
    return text


def _extract_docx(data: bytes) -> str:
    """Extract text from a DOCX using ``python-docx`` (lazy import)."""
    try:
        import docx
    except ImportError as e:  # pragma: no cover - dependency declared
        raise ExtractionError(
            "DOCX extraction requires the 'python-docx' package. "
            "Install it to analyze DOCX resumes."
        ) from e
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:  # not a valid docx (zip) file
        raise ExtractionError(
            "The DOCX file appears to be malformed or corrupted."
        ) from e
    paragraphs = [
        p.text for p in document.paragraphs if p.text and p.text.strip()
    ]
    text = "\n".join(paragraphs).strip()
    if not text:
        raise EmptyFileError("The DOCX contains no readable text.")
    return text
