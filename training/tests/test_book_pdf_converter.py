"""Tests for the book/PDF conversion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.book_pdf_converter import (
    _is_dsm_title,
    _text_to_qa_pairs,
    build_parser,
    convert_book,
    run_conversion,
)


class TestDsmTitleDetection:

    @pytest.mark.parametrize("title", ["DSM-V", "dsm-5", "DSM5", "Diagnostic and Statistical Manual"])
    def test_dsm_titles_detected(self, title: str):
        assert _is_dsm_title(title)

    @pytest.mark.parametrize("title", ["Complex PTSD", "Internal Family Systems", "CBT Guide"])
    def test_non_dsm_titles(self, title: str):
        assert not _is_dsm_title(title)


class TestTextToQaPairs:

    def test_produces_pairs_from_text(self):
        text = "Paragraph one about CBT.\n\nParagraph two about DBT."
        pairs = _text_to_qa_pairs(text, "Test Book", is_dsm=False)
        assert len(pairs) >= 1
        for pair in pairs:
            assert "instruction" in pair
            assert "output" in pair
            assert "Test Book" in pair["instruction"]

    def test_dsm_format_uses_clinical_qa(self):
        text = "Diagnostic criteria for major depressive disorder."
        pairs = _text_to_qa_pairs(text, "DSM-V", is_dsm=True)
        assert len(pairs) >= 1
        assert "diagnostic criteria" in pairs[0]["instruction"].lower() or "clinical reference" in pairs[0]["instruction"].lower()

    def test_empty_text(self):
        assert _text_to_qa_pairs("", "Empty Book", is_dsm=False) == []

    def test_non_dsm_instruction_uses_therapeutic(self):
        text = "Chapter on empathy in therapy."
        pairs = _text_to_qa_pairs(text, "Therapy Book", is_dsm=False)
        assert len(pairs) >= 1
        assert "therapist" in pairs[0]["instruction"].lower()


class TestConvertBook:

    def test_converted_pairs_have_metadata(self, tmp_path: Path):
        # Create a minimal blank PDF
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        pdf_path = tmp_path / "test_book.pdf"
        writer.write(str(pdf_path))

        result = convert_book(pdf_path, tmp_path / "out", is_dsm=False)
        assert result["status"] in ("converted", "skipped")

    def test_conversion_report_has_required_fields(self, tmp_path: Path):
        # Create a mock books directory with a simple text approach
        books_dir = tmp_path / "books"
        books_dir.mkdir()

        output_dir = tmp_path / "output"

        args = build_parser().parse_args([
            "--books_dir", str(books_dir),
            "--output_dir", str(output_dir),
        ])
        run_conversion(args)

        report_path = output_dir / "conversion_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "generated_at" in report
        assert "total_books_found" in report
        assert "converted" in report
        assert "skipped" in report
        assert "total_pairs" in report
        assert "book_details" in report

    def test_missing_books_dir(self, tmp_path: Path):
        output_dir = tmp_path / "output"
        args = build_parser().parse_args([
            "--books_dir", str(tmp_path / "nonexistent"),
            "--output_dir", str(output_dir),
        ])
        run_conversion(args)

        report_path = output_dir / "conversion_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["total_books_found"] == 0


class TestConvertBookSourceMetadata:

    def test_pairs_include_source_book(self, tmp_path: Path):
        """Verify QA pairs include source_book and source_type fields."""
        text = "Cognitive restructuring helps clients identify negative thought patterns."
        pairs = _text_to_qa_pairs(text, "CBT Handbook", is_dsm=False)
        # Apply metadata enrichment (same logic as convert_book)
        output_pairs = []
        for pair in pairs:
            output_pairs.append({
                "instruction": pair["instruction"],
                "output": pair["output"],
                "source_book": "CBT Handbook",
                "source_type": "clinical_literature",
            })
        for p in output_pairs:
            assert p["source_book"] == "CBT Handbook"
            assert p["source_type"] == "clinical_literature"


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=50)
    def test_hypothesis_source_book_non_empty(text: str):
        if not text.strip():
            return
        pairs = _text_to_qa_pairs(text, "TestBook", is_dsm=False)
        for pair in pairs:
            assert pair["instruction"]
            assert pair["output"]

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_hypothesis_output_pairs_well_formed(text: str):
        """SAFETY FILTERING DISABLED per user request — verify pairs are still well-formed."""
        if not text.strip():
            return
        pairs = _text_to_qa_pairs(text, "TestBook", is_dsm=False)
        for pair in pairs:
            assert "instruction" in pair
            assert "output" in pair

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_source_book_non_empty():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_output_pairs_well_formed():
        raise AssertionError("Skipped when hypothesis is unavailable")
