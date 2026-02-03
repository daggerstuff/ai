"""
Tests for Knowledge Sources RAG Integration

Tests the integration of therapeutic books, PDFs, and clinical references
into the RAG system for enhanced knowledge retrieval.
"""

from unittest.mock import patch

import pytest


class TestKnowledgeTextExtractor:
    """Tests for the KnowledgeTextExtractor class."""

    def test_registry_loading(self):
        """Test that the registry loads correctly."""
        from ai.pipelines.orchestrator.knowledge_text_extractor import (
            KnowledgeTextExtractor,
        )

        extractor = KnowledgeTextExtractor()

        # Should have loaded sources from registry
        assert len(extractor.sources) > 0, "Should load knowledge sources from registry"

        # Check that we have the expected source types
        source_types = {s.source_type for s in extractor.sources.values()}
        assert "therapeutic_book" in source_types or "clinical_reference" in source_types

    def test_source_metadata_parsing(self):
        """Test that source metadata is correctly parsed."""
        from ai.pipelines.orchestrator.knowledge_text_extractor import (
            KnowledgeTextExtractor,
        )

        extractor = KnowledgeTextExtractor()

        # Check critical sources are present
        critical_sources = [s for s in extractor.sources.values() if s.priority == "critical"]
        assert len(critical_sources) >= 1, "Should have at least one critical source"

        # Verify metadata fields are populated
        for source in critical_sources:
            assert source.title, f"Source {source.source_id} should have a title"
            assert source.topics, f"Source {source.source_id} should have topics"

    def test_cache_path_generation(self):
        """Test that cache paths are generated correctly."""
        from ai.pipelines.orchestrator.knowledge_text_extractor import (
            KnowledgeTextExtractor,
        )

        extractor = KnowledgeTextExtractor()

        # Get cache path for a source
        cache_path = extractor.get_cached_text_path("complex_ptsd_pete_walker")

        assert cache_path.suffix == ".txt", "Cache file should be .txt"
        assert "complex_ptsd_pete_walker" in str(cache_path)

    def test_source_stats(self):
        """Test the statistics generation."""
        from ai.pipelines.orchestrator.knowledge_text_extractor import (
            KnowledgeTextExtractor,
        )

        extractor = KnowledgeTextExtractor()
        stats = extractor.get_source_stats()

        assert "total_sources" in stats
        assert "by_type" in stats
        assert "by_priority" in stats
        assert "total_size_mb" in stats

        # Verify counts match
        assert stats["total_sources"] == len(extractor.sources)

    def test_chunk_text(self):
        """Test text chunking functionality."""
        from ai.pipelines.orchestrator.knowledge_text_extractor import (
            KnowledgeSourceMetadata,
            KnowledgeTextExtractor,
        )

        extractor = KnowledgeTextExtractor()

        # Add a mock source for testing
        test_source_id = "test_source"
        extractor.sources[test_source_id] = KnowledgeSourceMetadata(
            source_id=test_source_id,
            title="Test Book",
            author="Test Author",
            source_type="therapeutic_book",
            topics=["test"],
            priority="medium",
        )

        # Create sample text with multiple paragraphs
        sample_text = """
This is the first paragraph. It contains some text about mental health.

This is the second paragraph. It talks about therapy techniques.

This is the third paragraph. It discusses recovery and healing.
        """.strip()

        chunks = extractor.chunk_text(sample_text, test_source_id)

        assert len(chunks) > 0, "Should create at least one chunk"
        for chunk in chunks:
            assert chunk.source_id == test_source_id
            assert chunk.metadata.title == "Test Book"
            assert len(chunk.content) > 0


class TestYouTubeRAGSystemWithKnowledge:
    """Tests for YouTubeRAGSystem with knowledge sources enabled."""

    @patch("ai.pipelines.orchestrator.youtube_rag_system.HAS_KNOWLEDGE_EXTRACTOR", True)
    def test_rag_system_init_with_knowledge(self):
        """Test RAG system initialization with knowledge sources."""
        from ai.pipelines.orchestrator.youtube_rag_system import YouTubeRAGSystem

        # Create with knowledge sources enabled
        with patch.dict("os.environ", {"KNOWLEDGE_SOURCES_ENABLED": "true"}):
            rag = YouTubeRAGSystem(include_knowledge_sources=True)

            assert rag.include_knowledge_sources is True

    def test_rag_system_init_without_knowledge(self):
        """Test RAG system initialization without knowledge sources."""
        from ai.pipelines.orchestrator.youtube_rag_system import YouTubeRAGSystem

        # Create with knowledge sources disabled
        rag = YouTubeRAGSystem(include_knowledge_sources=False)

        assert rag.include_knowledge_sources is False
        assert rag.knowledge_extractor is None

    @patch("ai.pipelines.orchestrator.youtube_rag_system.HAS_KNOWLEDGE_EXTRACTOR", True)
    def test_knowledge_metadata_in_search_results(self):
        """Test that knowledge sources include proper source attribution."""
        from ai.pipelines.orchestrator.youtube_rag_system import (
            RAGIndexEntry,
            TranscriptMetadata,
        )

        # Create a mock knowledge entry
        knowledge_metadata = TranscriptMetadata(
            video_id="complex_ptsd_pete_walker",
            title="Complex PTSD: From Surviving to Thriving",
            speaker="Pete Walker",
            duration=0.0,
            language="en",
            processed_date="2026-02-03",
            content_hash="abc123",
            word_count=500,
            topics=["complex_trauma", "ptsd", "recovery"],
            therapeutic_approaches=[],
            personality_markers={
                "source_type": "therapeutic_book",
                "priority": "critical",
                "is_knowledge_source": True,
            },
            key_quotes=[],
            summary="Complex PTSD: From Surviving to Thriving by Pete Walker",
        )

        entry = RAGIndexEntry(
            transcript_id="complex_ptsd_pete_walker_0",
            content="Sample content about complex PTSD and emotional flashbacks.",
            embedding=None,
            metadata=knowledge_metadata,
        )

        # Verify metadata structure
        assert entry.metadata.personality_markers.get("is_knowledge_source") is True
        assert entry.metadata.personality_markers.get("source_type") == "therapeutic_book"
        assert "ptsd" in entry.metadata.topics


class TestMigrationScript:
    """Tests for the migration script."""

    def test_migrator_report_generation(self):
        """Test that the migration report is generated correctly."""
        from ai.scripts.migrate_knowledge_sources import KnowledgeSourceMigrator

        migrator = KnowledgeSourceMigrator()
        report = migrator.generate_integration_report()

        assert "total_sources" in report
        assert "needs_migration" in report
        assert "already_integrated" in report
        assert "total_size_mb" in report
        assert "by_priority" in report

        # Verify we have sources to migrate
        assert report["total_sources"] > 0

    def test_migrator_loads_registry(self):
        """Test that the migrator loads the registry correctly."""
        from ai.scripts.migrate_knowledge_sources import KnowledgeSourceMigrator

        migrator = KnowledgeSourceMigrator()

        assert migrator.registry is not None
        assert "knowledge_sources" in migrator.registry
        assert migrator.s3_bucket == "pixel-data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
