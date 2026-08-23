"""Tests for the resume text extractor (PDF / DOCX / TXT + edge cases)."""

from __future__ import annotations

import pytest

from src.modules.resume_intelligence.extractor import (
    EmptyFileError,
    ExtractionError,
    UnsupportedFileTypeError,
    extract_text,
    get_extension,
)


# ---- extension detection --------------------------------------------------


def test_get_extension_lowercase() -> None:
    assert get_extension("resume.pdf") == "pdf"
    assert get_extension("Resume.PDF") == "pdf"
    assert get_extension("a/b/c/resume.docx") == "docx"
    assert get_extension("C:\\Users\\me\\resume.txt") == "txt"
    assert get_extension("noext") == ""
    assert get_extension("") == ""


# ---- (3) TXT extraction ---------------------------------------------------


def test_extract_txt(sample_txt_bytes: bytes) -> None:
    text = extract_text("resume.txt", sample_txt_bytes)
    assert "Jane Smith" in text
    assert "Senior Software Engineer" in text
    assert "Python" in text


def test_extract_txt_strips_whitespace(sample_txt_bytes: bytes) -> None:
    text = extract_text("resume.txt", sample_txt_bytes)
    assert text == text.strip()
    assert not text.startswith(" ")


def test_extract_txt_non_utf8_falls_back_gracefully() -> None:
    # latin-1 bytes that are invalid as UTF-8
    raw = "café résumé".encode("latin-1")
    text = extract_text("r.txt", raw)
    assert "caf" in text


# ---- (2) DOCX extraction --------------------------------------------------


def test_extract_docx(sample_docx_bytes: bytes) -> None:
    text = extract_text("resume.docx", sample_docx_bytes)
    assert "Alex Lee" in text
    assert "Backend Engineer" in text
    assert "Go" in text


# ---- (1) PDF extraction ---------------------------------------------------


def test_extract_pdf(sample_pdf_bytes: bytes) -> None:
    text = extract_text("resume.pdf", sample_pdf_bytes)
    assert "Sam Taylor" in text
    assert "Data Scientist" in text


# ---- (4) empty input ------------------------------------------------------


def test_empty_bytes_raises_empty_file_error(empty_bytes: bytes) -> None:
    with pytest.raises(EmptyFileError):
        extract_text("resume.txt", empty_bytes)


def test_empty_txt_raises_empty_file_error(empty_txt_bytes: bytes) -> None:
    with pytest.raises(EmptyFileError):
        extract_text("resume.txt", empty_txt_bytes)


# ---- (5) unsupported file type --------------------------------------------


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError) as ei:
        extract_text("resume.doc", b"some bytes")
    assert ".doc" in str(ei.value)


def test_no_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("resume", b"some bytes")


# ---- (6) malformed file handling -----------------------------------------


def test_malformed_pdf_raises_extraction_error(malformed_pdf_bytes: bytes) -> None:
    with pytest.raises(ExtractionError):
        extract_text("resume.pdf", malformed_pdf_bytes)


def test_malformed_docx_raises_extraction_error(malformed_docx_bytes: bytes) -> None:
    with pytest.raises(ExtractionError):
        extract_text("resume.docx", malformed_docx_bytes)


def test_malformed_pdf_message_does_not_leak_content(
    malformed_pdf_bytes: bytes,
) -> None:
    """Error messages must not echo back the binary content."""
    with pytest.raises(ExtractionError) as ei:
        extract_text("resume.pdf", malformed_pdf_bytes)
    msg = str(ei.value)
    assert "this is not really a valid pdf body structure" not in msg


# ---- type safety ---------------------------------------------------------


def test_extract_returns_str(sample_txt_bytes: bytes) -> None:
    text = extract_text("resume.txt", sample_txt_bytes)
    assert isinstance(text, str)
    assert text  # non-empty
