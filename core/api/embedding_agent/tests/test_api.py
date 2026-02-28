"""
Tests for the Embedding Agent FastAPI endpoints.

Tests cover:
- Health check endpoint
- Single embedding endpoint
- Batch embedding endpoint
- Similarity search endpoint
- Cache management endpoints
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from ..app import create_app, get_embedding_service
from ..models import EmbeddingAgentConfig, EmbeddingModel
from ..service import EmbeddingAgentService


@pytest.fixture
def config() -> EmbeddingAgentConfig:
    """Create test configuration."""
    return EmbeddingAgentConfig(
        model_name=EmbeddingModel.MINILM_L6_V2,
        embedding_dimension=384,
        batch_size=8,
        max_text_length=512,
        cache_embeddings=True,
        use_gpu=False,
    )


@pytest.fixture
def app(config: EmbeddingAgentConfig):
    """Create test FastAPI app."""
    return create_app(config=config, project_root=Path(__file__).parent.parent.parent.parent)


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, client: TestClient):
        """Test that health endpoint returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data

    def test_health_check_structure(self, client: TestClient):
        """Test health response structure."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == "1.0.0"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["cache_available"], bool)


class TestStatusEndpoint:
    """Tests for status endpoint."""

    def test_status_returns_200(self, client: TestClient):
        """Test that status endpoint returns 200."""
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data
        assert "embedding_dimension" in data
        assert "cache_size" in data
        assert "uptime_seconds" in data

    def test_status_contains_metrics(self, client: TestClient):
        """Test that status contains performance metrics."""
        response = client.get("/status")
        data = response.json()

        assert "requests_processed" in data
        assert "average_response_time_ms" in data


class TestEmbedEndpoint:
    """Tests for single embedding endpoint."""

    def test_embed_valid_text(self, client: TestClient):
        """Test embedding generation for valid text."""
        response = client.post(
            "/api/v1/embeddings/embed",
            json={"text": "Hello, world!"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "embedding" in data
        assert len(data["embedding"]) == 384
        assert "embedding_id" in data
        assert "model_used" in data

    def test_embed_with_knowledge_type(self, client: TestClient):
        """Test embedding with knowledge type."""
        response = client.post(
            "/api/v1/embeddings/embed",
            json={
                "text": "Major Depressive Disorder symptoms",
                "knowledge_type": "dsm5"
            }
        )

        assert response.status_code == 200

    def test_embed_empty_text_returns_400(self, client: TestClient):
        """Test that empty text returns 400."""
        response = client.post(
            "/api/v1/embeddings/embed",
            json={"text": ""}
        )

        assert response.status_code == 422  # Validation error

    def test_embed_whitespace_only_returns_400(self, client: TestClient):
        """Test that whitespace-only text returns 400."""
        response = client.post(
            "/api/v1/embeddings/embed",
            json={"text": "   "}
        )

        assert response.status_code == 422

    def test_embed_missing_text_returns_422(self, client: TestClient):
        """Test that missing text field returns 422."""
        response = client.post(
            "/api/v1/embeddings/embed",
            json={}
        )

        assert response.status_code == 422


class TestBatchEmbedEndpoint:
    """Tests for batch embedding endpoint."""

    def test_batch_embed_multiple_texts(self, client: TestClient):
        """Test batch embedding of multiple texts."""
        response = client.post(
            "/api/v1/embeddings/embed/batch",
            json={
                "texts": ["Text one", "Text two", "Text three"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 3
        assert len(data["embeddings"]) == 3

    def test_batch_embed_empty_list_returns_422(self, client: TestClient):
        """Test that empty list returns 422."""
        response = client.post(
            "/api/v1/embeddings/embed/batch",
            json={"texts": []}
        )

        assert response.status_code == 422

    def test_batch_embed_too_many_texts_returns_422(self, client: TestClient):
        """Test that too many texts returns 422."""
        response = client.post(
            "/api/v1/embeddings/embed/batch",
            json={"texts": ["text"] * 101}
        )

        assert response.status_code == 422


class TestSearchEndpoint:
    """Tests for similarity search endpoint."""

    def test_search_valid_query(self, client: TestClient):
        """Test search with valid query."""
        response = client.post(
            "/api/v1/embeddings/search",
            json={"query": "depression treatment"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "matches" in data
        assert "query_embedding_id" in data
        assert "processing_time_ms" in data

    def test_search_with_top_k(self, client: TestClient):
        """Test search with custom top_k."""
        response = client.post(
            "/api/v1/embeddings/search",
            json={
                "query": "therapy techniques",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["matches"]) <= 5

    def test_search_with_min_similarity(self, client: TestClient):
        """Test search with minimum similarity."""
        response = client.post(
            "/api/v1/embeddings/search",
            json={
                "query": "test query",
                "min_similarity": 0.5
            }
        )

        assert response.status_code == 200
        data = response.json()

        for match in data["matches"]:
            assert match["similarity_score"] >= 0.5


class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    def test_get_cache_stats(self, client: TestClient):
        """Test getting cache statistics."""
        response = client.get("/api/v1/embeddings/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert "cache_size" in data
        assert "cache_enabled" in data

    def test_clear_cache(self, client: TestClient):
        """Test clearing cache."""
        # First generate some cached embeddings
        client.post(
            "/api/v1/embeddings/embed",
            json={"text": "Cache test text"}
        )

        # Clear cache
        response = client.delete("/api/v1/embeddings/cache")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestConfigEndpoint:
    """Tests for configuration endpoint."""

    def test_get_config(self, client: TestClient):
        """Test getting configuration."""
        response = client.get("/api/v1/embeddings/config")

        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data
        assert "embedding_dimension" in data
        assert "batch_size" in data


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client: TestClient):
        """Test listing available models."""
        response = client.get("/api/v1/embeddings/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert len(data["models"]) > 0

        for model in data["models"]:
            assert "id" in model
            assert "name" in model
            assert "dimension" in model
            assert "description" in model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

