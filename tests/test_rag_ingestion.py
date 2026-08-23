"""Tests for document ingestion (PDF/DOCX/TXT/MD) + normalization + fingerprint."""

from __future__ import annotations

import pytest

from src.modules.rag.ingestion import (
    EmptyDocumentError,
    IngestionError,
    MalformedDocumentError,
    TooLargeDocumentError,
    UnsupportedFileTypeError,
    MAX_FILE_SIZE,
    document_fingerprint,
    extract_sections,
    get_extension,
    normalize,
)


# ---- (6-9) PDF / DOCX / TXT / MD ingestion ---------------------------


def test_ingest_pdf(sample_pdf_bytes: bytes) -> None:
    sections = extract_sections("doc.pdf", sample_pdf_bytes)
    assert sections
    text = " ".join(s.text for s in sections)
    assert "Sam Taylor" in text
    assert all(isinstance(s.page, int) for s in sections)  # PDF preserves pages


def test_ingest_docx(sample_docx_bytes: bytes) -> None:
    sections = extract_sections("doc.docx", sample_docx_bytes)
    assert sections
    text = sections[0].text
    assert "Alex Lee" in text
    assert sections[0].page is None  # DOCX has no pages


def test_ingest_txt(sample_txt_bytes: bytes) -> None:
    sections = extract_sections("doc.txt", sample_txt_bytes)
    assert sections
    assert "Jane Smith" in sections[0].text


def test_ingest_markdown(sample_md_bytes: bytes) -> None:
    sections = extract_sections("doc.md", sample_md_bytes)
    assert sections
    text = " ".join(s.text for s in sections)
    assert "CareerOS" in text
    assert "Streamlit" in text


# ---- (10) unsupported format -----------------------------------------


def test_ingest_unsupported_format_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_sections("doc.doc", b"some bytes")
    with pytest.raises(UnsupportedFileTypeError):
        extract_sections("noext", b"some bytes")


# ---- (11) malformed file --------------------------------------------


def test_ingest_malformed_pdf_raises(malformed_pdf_bytes: bytes) -> None:
    with pytest.raises(MalformedDocumentError):
        extract_sections("bad.pdf", malformed_pdf_bytes)


def test_ingest_malformed_docx_raises(malformed_docx_bytes: bytes) -> None:
    with pytest.raises(MalformedDocumentError):
        extract_sections("bad.docx", malformed_docx_bytes)


# ---- (12) empty file -------------------------------------------------


def test_ingest_empty_file_raises(empty_bytes: bytes) -> None:
    with pytest.raises(EmptyDocumentError):
        extract_sections("empty.txt", empty_bytes)


def test_ingest_pdf_no_text_raises() -> None:
    # A "PDF" with no extractable text -> EmptyDocumentError.
    with pytest.raises((EmptyDocumentError, MalformedDocumentError)):
        extract_sections("empty.pdf", b"%PDF-1.4\n(broken)")


# ---- (13) normalization ---------------------------------------------


def test_normalize_collapses_whitespace() -> None:
    raw = "Hello   world\t\n\n\n\n\nworld"
    norm = normalize(raw)
    assert "  " not in norm  # no double spaces
    assert "\n\n\n" not in norm  # no 3+ newlines
    assert "Hello" in norm and "world" in norm


def test_normalize_strips_artifacts() -> None:
    raw = "Hello\x00 world\fend"
    norm = normalize(raw)
    assert "\x00" not in norm
    assert "\f" not in norm


def test_normalize_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == "   ".strip()


# ---- (17) stable document ID ----------------------------------------


def test_document_fingerprint_stable() -> None:
    a = document_fingerprint(b"hello", "f.txt")
    b = document_fingerprint(b"hello", "f.txt")
    assert a == b


def test_document_fingerprint_differs_for_different_bytes() -> None:
    assert document_fingerprint(b"hello", "f.txt") != document_fingerprint(b"world", "f.txt")


def test_document_fingerprint_short() -> None:
    fp = document_fingerprint(b"x", "f.txt")
    assert len(fp) == 16


# ---- too-large file -------------------------------------------------


def test_ingest_too_large_raises() -> None:
    big = b"x" * (MAX_FILE_SIZE + 1)
    with pytest.raises(TooLargeDocumentError):
        extract_sections("big.txt", big)


def test_get_extension() -> None:
    assert get_extension("a.pdf") == "pdf"
    assert get_extension("A/B/c.DOCX") == "docx"
    assert get_extension("noext") == ""
    assert get_extension("") == ""
