"""
Tests for the EmbeddingAgentService.

Tests cover:
- Single text embedding
- Batch text embedding
- Similarity search
- Caching behavior
- Status reporting
"""

import pytest
from pathlib import Path

from ..models import (
    EmbeddingAgentConfig,
    EmbeddingModel,
    EmbeddingRequest,
    BatchEmbeddingRequest,
    SimilaritySearchRequest,
    KnowledgeType,
)
from ..service import EmbeddingAgentService


@pytest.fixture
def config() -> EmbeddingAgentConfig:
    """Create test configuration."""
    return EmbeddingAgentConfig(
        model_name=EmbeddingModel.MINILM_L6_V2,
        embedding_dimension=384,
        batch_size=8,
        max_text_length=512,
        normalize_embeddings=True,
        cache_embeddings=True,
        use_gpu=False,
    )


@pytest.fixture
def service(config: EmbeddingAgentConfig) -> EmbeddingAgentService:
    """Create test service instance."""
    return EmbeddingAgentService(
        config=config,
        project_root=Path(__file__).parent.parent.parent.parent,
    )


class TestSingleEmbedding:
    """Tests for single text embedding."""

    def test_embed_text_returns_embedding(self, service: EmbeddingAgentService):
        """Test that embedding is generated for valid text."""
        request = EmbeddingRequest(
            text="Hello, this is a test message for embedding.",
            knowledge_type=KnowledgeType.GENERAL,
        )

        response = service.embed_text(request)

        assert response.embedding is not None
        assert len(response.embedding) == service.config.embedding_dimension
        assert response.embedding_id is not None
        assert response.model_used is not None
        assert response.dimension == service.config.embedding_dimension
        assert response.text_hash is not None
        assert response.processing_time_ms >= 0

    def test_embed_text_normalized(self, service: EmbeddingAgentService):
        """Test that embeddings are normalized."""
        request = EmbeddingRequest(text="Test normalization")

        response = service.embed_text(request)

        # Check L2 norm is approximately 1
        norm_squared = sum(x ** 2 for x in response.embedding)
        assert abs(norm_squared - 1.0) < 0.01

    def test_embed_text_caching(self, service: EmbeddingAgentService):
        """Test that identical texts use cache."""
        text = "This text should be cached."
        request = EmbeddingRequest(text=text)

        # First request
        response1 = service.embed_text(request)
        assert not response1.cached

        # Second request should hit cache
        response2 = service.embed_text(request)
        assert response2.cached
        assert response1.embedding == response2.embedding

    def test_embed_text_deterministic(self, service: EmbeddingAgentService):
        """Test that same text produces same embedding."""
        text = "Deterministic embedding test"
        request = EmbeddingRequest(text=text)

        response1 = service.embed_text(request)

        # Clear cache
        service.clear_cache()

        response2 = service.embed_text(request)

        # Should produce same embedding
        for v1, v2 in zip(response1.embedding, response2.embedding):
            assert abs(v1 - v2) < 0.0001


class TestBatchEmbedding:
    """Tests for batch text embedding."""

    def test_batch_embed_multiple_texts(self, service: EmbeddingAgentService):
        """Test batch embedding of multiple texts."""
        request = BatchEmbeddingRequest(
            texts=[
                "First text for embedding",
                "Second text for embedding",
                "Third text for embedding",
            ]
        )

        response = service.embed_batch(request)

        assert response.total_count == 3
        assert len(response.embeddings) == 3
        assert response.generated_count + response.cached_count == 3

        # Each embedding should have correct structure
        for item in response.embeddings:
            assert len(item.embedding) == service.config.embedding_dimension
            assert item.embedding_id is not None

    def test_batch_embed_with_caching(self, service: EmbeddingAgentService):
        """Test that batch embedding uses cache."""
        texts = ["Cached text 1", "Cached text 2"]

        # First batch
        request1 = BatchEmbeddingRequest(texts=texts)
        response1 = service.embed_batch(request1)
        assert response1.cached_count == 0

        # Second batch with same texts
        request2 = BatchEmbeddingRequest(texts=texts)
        response2 = service.embed_batch(request2)
        assert response2.cached_count == 2

    def test_batch_embed_preserves_order(self, service: EmbeddingAgentService):
        """Test that batch results maintain original order."""
        texts = ["Text A", "Text B", "Text C"]
        request = BatchEmbeddingRequest(texts=texts)

        response = service.embed_batch(request)

        for i, item in enumerate(response.embeddings):
            assert item.index == i


class TestSimilaritySearch:
    """Tests for similarity search functionality."""

    def test_search_returns_results(self, service: EmbeddingAgentService):
        """Test that search returns results."""
        request = SimilaritySearchRequest(
            query="depression treatment options",
            top_k=5,
        )

        response = service.search_similar(request)

        assert response.query_embedding_id is not None
        assert response.processing_time_ms >= 0
        assert response.model_used is not None

    def test_search_with_knowledge_type_filter(self, service: EmbeddingAgentService):
        """Test search with knowledge type filtering."""
        request = SimilaritySearchRequest(
            query="therapeutic techniques",
            top_k=10,
            knowledge_types=[KnowledgeType.DSM5, KnowledgeType.CLINICAL],
        )

        response = service.search_similar(request)

        # All matches should be of specified types
        for match in response.matches:
            assert match.knowledge_type in [KnowledgeType.DSM5, KnowledgeType.CLINICAL]

    def test_search_respects_min_similarity(self, service: EmbeddingAgentService):
        """Test that search respects minimum similarity threshold."""
        request = SimilaritySearchRequest(
            query="test query",
            top_k=10,
            min_similarity=0.5,
        )

        response = service.search_similar(request)

        for match in response.matches:
            assert match.similarity_score >= 0.5

    def test_search_with_precomputed_embedding(self, service: EmbeddingAgentService):
        """Test search with pre-computed query embedding."""
        # First get an embedding
        embed_request = EmbeddingRequest(text="test query")
        embed_response = service.embed_text(embed_request)

        # Use it for search
        search_request = SimilaritySearchRequest(
            query="test query",  # Still required but won't be used
            query_embedding=embed_response.embedding,
            top_k=5,
        )

        response = service.search_similar(search_request)

        assert response.query_embedding_id is not None


class TestServiceStatus:
    """Tests for service status and management."""

    def test_get_status(self, service: EmbeddingAgentService):
        """Test status reporting."""
        status = service.get_status()

        assert status.model_name == service.config.model_name.value
        assert status.embedding_dimension == service.config.embedding_dimension
        assert status.uptime_seconds >= 0

    def test_status_tracks_requests(self, service: EmbeddingAgentService):
        """Test that status tracks processed requests."""
        initial_status = service.get_status()
        initial_count = initial_status.requests_processed

        # Make some requests
        for i in range(3):
            service.embed_text(EmbeddingRequest(text=f"Test {i}"))

        updated_status = service.get_status()
        assert updated_status.requests_processed == initial_count + 3

    def test_clear_cache(self, service: EmbeddingAgentService):
        """Test cache clearing."""
        # Generate some cached embeddings
        service.embed_text(EmbeddingRequest(text="Cache test 1"))
        service.embed_text(EmbeddingRequest(text="Cache test 2"))

        status_before = service.get_status()
        assert status_before.cache_size >= 2

        # Clear cache
        cleared = service.clear_cache()
        assert cleared >= 2

        status_after = service.get_status()
        assert status_after.cache_size == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_long_text_truncation(self, service: EmbeddingAgentService):
        """Test that long texts are truncated."""
        # Create text longer than max_length
        long_text = "word " * 1000
        request = EmbeddingRequest(text=long_text)

        # Should not raise, text will be truncated
        response = service.embed_text(request)

        assert response.embedding is not None
        assert len(response.embedding) == service.config.embedding_dimension

    def test_unicode_text(self, service: EmbeddingAgentService):
        """Test handling of unicode text."""
        request = EmbeddingRequest(
            text="This includes émojis 🎉 and üñíçödé characters 日本語"
        )

        response = service.embed_text(request)

        assert response.embedding is not None

    def test_empty_batch(self, service: EmbeddingAgentService):
        """Test that empty batch raises validation error."""
        with pytest.raises(ValueError):
            BatchEmbeddingRequest(texts=[])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

