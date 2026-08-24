"""
Tests for Academic Sourcing Engine

Run with: uv run python -m pytest ai/sourcing/academic/tests/test_academic_sourcing.py
"""

from unittest.mock import MagicMock, patch

import pytest

from ai.sourcing.academic import (
    AcademicSourcingEngine,
    BookMetadata,
    SourceType,
    SourcingStrategy,
)


class TestAcademicSourcingEngine:
    """Test suite for AcademicSourcingEngine"""

    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = AcademicSourcingEngine()
        assert engine is not None
        assert engine.strategy == SourcingStrategy.HYBRID
        assert engine.output_path.exists()

    def test_engine_with_custom_strategy(self):
        """Test engine with custom sourcing strategy"""
        engine = AcademicSourcingEngine(strategy=SourcingStrategy.API_ONLY)
        assert engine.strategy == SourcingStrategy.API_ONLY

    def test_therapeutic_keywords_loaded(self):
        """Test therapeutic keywords are loaded"""
        engine = AcademicSourcingEngine()
        assert len(engine.therapeutic_keywords) > 0
        assert "therapy" in engine.therapeutic_keywords
        assert "cbt" in engine.therapeutic_keywords

    @patch("requests.Session.get")
    def test_fetch_arxiv_papers(self, mock_get):
        """Test ArXiv paper fetching"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Test Psychology Paper</title>
                <author><name>John Doe</name></author>
                <published>2024-01-01</published>
                <summary>Abstract about therapy</summary>
                <id>http://arxiv.org/abs/2401.00001</id>
            </entry>
        </feed>"""
        mock_get.return_value = mock_response

        engine = AcademicSourcingEngine()
        results = engine.fetch_arxiv_papers("therapy", limit=10)

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], BookMetadata)

    @patch("requests.Session.get")
    def test_fetch_semantic_scholar(self, mock_get):
        """Test Semantic Scholar fetching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "authors": [{"name": "Jane Smith"}],
                    "year": 2024,
                    "abstract": "Test abstract",
                    "paperId": "12345",
                }
            ]
        }
        mock_get.return_value = mock_response

        engine = AcademicSourcingEngine()
        results = engine.fetch_semantic_scholar("therapy", limit=10)

        assert isinstance(results, list)

    def test_score_therapeutic_relevance(self):
        """Test therapeutic relevance scoring"""
        engine = AcademicSourcingEngine()

        metadata = BookMetadata(
            title="Cognitive Behavioral Therapy for Depression",
            authors=["Test Author"],
            publisher="Test Publisher",
            publication_year=2024,
            abstract="This book covers CBT techniques for treating depression",
            keywords=["therapy", "cbt", "depression"],
        )

        score = engine._score_therapeutic_relevance(metadata)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high for therapy-related content

    def test_score_non_therapeutic_content(self):
        """Test scoring of non-therapeutic content"""
        engine = AcademicSourcingEngine()

        metadata = BookMetadata(
            title="Introduction to Quantum Physics",
            authors=["Test Author"],
            publisher="Test Publisher",
            publication_year=2024,
            abstract="This book covers quantum mechanics",
            keywords=["physics", "quantum"],
        )

        score = engine._score_therapeutic_relevance(metadata)
        assert score < 0.3  # Should be low for non-therapy content

    def test_deduplicate_results(self):
        """Test result deduplication"""
        engine = AcademicSourcingEngine()

        # Create duplicate entries
        metadata1 = BookMetadata(
            title="Test Book",
            authors=["Author One"],
            publisher="Publisher",
            publication_year=2024,
        )

        metadata2 = BookMetadata(
            title="Test Book",  # Same title
            authors=["Author One"],  # Same author
            publisher="Publisher",
            publication_year=2024,
        )

        metadata3 = BookMetadata(
            title="Different Book",
            authors=["Author Two"],
            publisher="Publisher",
            publication_year=2024,
        )

        results = [metadata1, metadata2, metadata3]
        deduplicated = engine._deduplicate_results(results)

        assert len(deduplicated) == 2  # Should remove one duplicate

    def test_export_results(self):
        """Test exporting results to JSON"""
        engine = AcademicSourcingEngine()

        metadata = BookMetadata(
            title="Test Book",
            authors=["Test Author"],
            publisher="Test Publisher",
            publication_year=2024,
        )

        results = [metadata]
        output_file = engine.export_results(results, "test_export.json")

        assert output_file.exists()
        assert output_file.suffix == ".json"

        # Cleanup
        output_file.unlink()

    def test_source_type_enum(self):
        """Test SourceType enum values"""
        assert SourceType.ARXIV.value == "arxiv"
        assert SourceType.PUBMED.value == "pubmed"
        assert SourceType.SPRINGER.value == "springer"
        assert SourceType.APA_PUBLISHER.value == "apa_publisher"

    def test_sourcing_strategy_enum(self):
        """Test SourcingStrategy enum values"""
        assert SourcingStrategy.API_ONLY.value == "api_only"
        assert SourcingStrategy.PUBLISHER_ONLY.value == "publisher_only"
        assert SourcingStrategy.HYBRID.value == "hybrid"


class TestBookMetadata:
    """Test suite for BookMetadata dataclass"""

    def test_book_metadata_creation(self):
        """Test creating BookMetadata"""
        metadata = BookMetadata(
            title="Test Book",
            authors=["Author 1", "Author 2"],
            publisher="Test Publisher",
            publication_year=2024,
            isbn="978-1234567890",
            doi="10.1234/test",
        )

        assert metadata.title == "Test Book"
        assert len(metadata.authors) == 2
        assert metadata.publication_year == 2024

    def test_book_metadata_to_dict(self):
        """Test converting BookMetadata to dict"""
        metadata = BookMetadata(
            title="Test Book",
            authors=["Author"],
            publisher="Publisher",
            publication_year=2024,
        )

        data_dict = metadata.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["title"] == "Test Book"
        assert "authors" in data_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
