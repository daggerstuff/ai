"""Tests for MetadataExtractor and ContentAnonymizer."""

import os
import tempfile

import pytest

from ai.sourcing.academic.anonymization.anonymizer import ContentAnonymizer
from ai.sourcing.academic.metadata_extraction.metadata_extractor import (
    ExtractedMetadata,
    MetadataExtractor,
)


class TestMetadataExtractor:
    def test_extract_doi(self) -> None:
        text = "See the paper at DOI: 10.1037/0003-066X.59.1.29 for details."
        result = MetadataExtractor().extract_from_text(text)
        assert result.doi == "10.1037/0003-066X.59.1.29"

    def test_extract_isbn13(self) -> None:
        text = "ISBN: 978-1-59385-982-0"
        result = MetadataExtractor().extract_from_text(text)
        assert result.isbn is not None
        assert result.isbn.startswith("978")

    def test_extract_year(self) -> None:
        text = "Published in 2019 by Academic Press."
        result = MetadataExtractor().extract_from_text(text)
        assert result.publication_year == 2019

    def test_extract_title_and_authors(self) -> None:
        text = "Cognitive Behavioral Therapy Manual\nby Aaron Beck, Judith Beck\nAbstract: This paper..."
        result = MetadataExtractor().extract_from_text(text)
        assert result.title is not None
        assert "Cognitive" in result.title
        assert "Aaron Beck" in result.authors

    def test_extract_keywords(self) -> None:
        text = "Keywords: therapy, CBT, depression, anxiety"
        result = MetadataExtractor().extract_from_text(text)
        assert "therapy" in result.keywords
        assert "CBT" in result.keywords

    def test_extract_abstract(self) -> None:
        text = "Title\n\nAbstract: This study examines CBT effectiveness.\n\nIntroduction: ..."
        result = MetadataExtractor().extract_from_text(text)
        assert result.abstract is not None
        assert "CBT" in result.abstract

    def test_extract_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Test Paper\nDOI: 10.1000/test\nKeywords: a, b\n")
            f.flush()
            result = MetadataExtractor().extract(f.name)
            assert result.doi == "10.1000/test"
            assert result.title is not None
            os.unlink(f.name)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            MetadataExtractor().extract("/nonexistent/file.txt")

    def test_to_dict(self) -> None:
        metadata = ExtractedMetadata(title="Test", doi="10.1/test")
        d = metadata.to_dict()
        assert d["title"] == "Test"
        assert d["doi"] == "10.1/test"

    def test_source_format(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello world")
            f.flush()
            result = MetadataExtractor().extract(f.name)
            assert result.source_format == "txt"
            os.unlink(f.name)


class TestContentAnonymizer:
    def test_detect_email(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "Contact John at john.doe@example.com"
        result = anonymizer.anonymize(text)
        assert "john.doe@example.com" not in result.anonymized_text
        assert "REDACTED_EMAIL" in result.anonymized_text
        assert result.total_pii_found >= 1

    def test_detect_phone(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "Call (555) 123-4567 for info"
        result = anonymizer.anonymize(text)
        assert "555" not in result.anonymized_text

    def test_detect_ssn(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "SSN: 123-45-6789"
        result = anonymizer.anonymize(text)
        assert "123-45-6789" not in result.anonymized_text

    def test_hash_mode(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "Email: test@example.com"
        result = anonymizer.anonymize(text, mode="hash")
        assert "test@example.com" not in result.anonymized_text
        assert "[HASH:" in result.anonymized_text

    def test_no_pii(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "The quick brown fox jumps over the lazy dog."
        result = anonymizer.anonymize(text)
        assert result.anonymized_text == text
        assert result.total_pii_found == 0

    def test_detect_pii_only(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "Email: a@b.com Phone: 555-123-4567"
        findings = anonymizer.detect_pii(text)
        types = {f["type"] for f in findings}
        assert "email" in types

    def test_has_pii(self) -> None:
        anonymizer = ContentAnonymizer()
        assert anonymizer.has_pii("email: test@test.com")
        assert not anonymizer.has_pii("no PII here")

    def test_custom_patterns(self) -> None:
        anonymizer = ContentAnonymizer(custom_patterns={"patient_id": r"PATIENT-[A-Z]{5}"})
        text = "Record: PATIENT-ABCDE"
        result = anonymizer.anonymize(text)
        assert "PATIENT-ABCDE" not in result.anonymized_text
        assert "REDACTED_PATIENT_ID" in result.anonymized_text

    def test_redactions_track_positions(self) -> None:
        anonymizer = ContentAnonymizer()
        text = "Email: a@b.com"
        result = anonymizer.anonymize(text)
        assert len(result.redactions) >= 1
        assert "position" in result.redactions[0]
        assert result.redactions[0]["type"] == "email"

    def test_to_dict(self) -> None:
        anonymizer = ContentAnonymizer()
        result = anonymizer.anonymize("test@example.com")
        d = result.to_dict()
        assert "anonymized_text" in d
        assert "redactions" in d
        assert "total_pii_found" in d
