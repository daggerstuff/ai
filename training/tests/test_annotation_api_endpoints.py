"""Tests for annotation_api FastAPI endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from annotation.api.database import Base
from annotation.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    # Use in-memory database for testing
    with patch("annotation.api.database.DATABASE_URL", "sqlite:///:memory:"):
        with patch("annotation.api.main.DATABASE_URL", "sqlite:///:memory:"):
            # Create all tables
            engine = create_engine("sqlite:///:memory:")
            Base.metadata.create_all(bind=engine)

            # Patch the engine used by get_db_session
            with patch("annotation.api.database.get_engine", return_value=engine):
                with TestClient(app) as test_client:
                    yield test_client


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self, client):
        """Test GET /health returns 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestQueueEndpoints:
    """Test queue management endpoints."""

    def test_add_to_queue_valid_data(self, client):
        """Test POST /queue with valid data returns 201 with item ID."""
        item_data = {
            "sample_text": "This is a sample text for testing.",
            "original_score": 0.5,
            "per_dimension_scores": {
                "technique": 0.6,
                "alliance": 0.4,
                "structure": 0.5,
                "cultural": 0.3,
                "ebp": 0.7,
                "dsm5": 0.2,
            },
            "routing_reason": "Borderline score between 0.4 and 0.6",
        }

        response = client.post("/queue", json=item_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["sample_text"] == item_data["sample_text"]
        assert data["original_score"] == item_data["original_score"]
        assert data["per_dimension_scores"] == item_data["per_dimension_scores"]
        assert data["routing_reason"] == item_data["routing_reason"]
        assert data["status"] == "pending"

    def test_add_to_queue_invalid_score(self, client):
        """Test POST /queue with invalid score returns 422."""
        item_data = {
            "sample_text": "Test text",
            "original_score": 1.5,  # Invalid: > 1.0
            "per_dimension_scores": {"technique": 0.5},
            "routing_reason": "Test",
        }

        response = client.post("/queue", json=item_data)
        assert response.status_code == 422

    def test_add_to_queue_invalid_dimension_scores(self, client):
        """Test POST /queue with invalid dimension scores returns 422."""
        item_data = {
            "sample_text": "Test text",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 1.5},  # Invalid: > 1.0
            "routing_reason": "Test",
        }

        response = client.post("/queue", json=item_data)
        assert response.status_code == 422

    def test_list_pending_items_empty(self, client):
        """Test GET /queue/pending with empty queue returns empty list."""
        response = client.get("/queue/pending")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_pending_items_with_items(self, client):
        """Test GET /queue/pending returns list of pending items."""
        # Add a pending item
        item_data = {
            "sample_text": "Test item 1",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        assert response.status_code == 201

        # Add a reviewed item
        item_data2 = {
            "sample_text": "Test item 2",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response2 = client.post("/queue", json=item_data2)
        item_id = response2.json()["id"]

        # Submit review to change status to reviewed
        review_data = {
            "reviewer_score": 0.8,
            "notes": "Good",
            "reviewer_id": "expert_1",
        }
        client.post(f"/queue/submit?item_id={item_id}", json=review_data)

        # Get pending items (should only have first item)
        response = client.get("/queue/pending")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["sample_text"] == "Test item 1"
        assert data[0]["status"] == "pending"

    def test_post_pending_items_endpoint(self, client):
        """Test POST /queue/pending also works."""
        response = client.post("/queue/pending")
        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestReviewEndpoints:
    """Test review submission endpoints."""

    def test_submit_review_valid(self, client):
        """Test POST /queue/submit with valid data returns 200."""
        # First add a queue item
        item_data = {
            "sample_text": "Test item for review",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]

        # Submit review
        review_data = {
            "reviewer_score": 0.8,
            "notes": "Good clinical response",
            "reviewer_id": "expert_1",
        }
        response = client.post(f"/queue/submit?item_id={item_id}", json=review_data)
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == item_id
        assert data["reviewer_score"] == 0.8
        assert data["notes"] == "Good clinical response"
        assert data["reviewer_id"] == "expert_1"

        # Verify item status changed to reviewed
        detail_response = client.get(f"/queue/detail?item_id={item_id}")
        assert detail_response.json()["status"] == "reviewed"

    def test_submit_review_nonexistent_item(self, client):
        """Test POST /queue/submit with non-existent item returns 404."""
        review_data = {
            "reviewer_score": 0.8,
            "notes": "Test",
            "reviewer_id": "expert_1",
        }
        response = client.post("/queue/submit?item_id=999", json=review_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_submit_review_invalid_score(self, client):
        """Test POST /queue/submit with invalid score returns 422."""
        # First add a queue item
        item_data = {
            "sample_text": "Test",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]

        # Submit review with invalid score
        review_data = {
            "reviewer_score": 1.5,  # Invalid: > 1.0
            "notes": "Test",
            "reviewer_id": "expert_1",
        }
        response = client.post(f"/queue/submit?item_id={item_id}", json=review_data)
        assert response.status_code == 422


class TestStatsEndpoint:
    """Test queue statistics endpoint."""

    def test_queue_stats_empty(self, client):
        """Test GET /queue/stats with empty queue."""
        response = client.get("/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["pending"] == 0
        assert data["reviewed"] == 0
        assert data["total"] == 0

    def test_queue_stats_with_items(self, client):
        """Test GET /queue/stats with mixed items."""
        # Add 2 pending items
        item_data = {
            "sample_text": "Test 1",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        client.post("/queue", json=item_data)
        client.post("/queue", json=item_data)

        # Add 1 reviewed item
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]
        review_data = {
            "reviewer_score": 0.8,
            "notes": "Good",
            "reviewer_id": "expert_1",
        }
        client.post(f"/queue/submit?item_id={item_id}", json=review_data)

        # Check stats
        response = client.get("/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["pending"] == 2
        assert data["reviewed"] == 1
        assert data["total"] == 3


class TestDetailEndpoint:
    """Test item detail endpoint."""

    def test_get_item_detail_valid(self, client):
        """Test GET /queue/detail with valid ID."""
        # Add an item
        item_data = {
            "sample_text": "Detailed test item",
            "original_score": 0.7,
            "per_dimension_scores": {"technique": 0.8, "alliance": 0.6},
            "routing_reason": "Detailed test",
        }
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]

        # Get detail
        response = client.get(f"/queue/detail?item_id={item_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == item_id
        assert data["sample_text"] == "Detailed test item"
        assert data["original_score"] == 0.7
        assert data["per_dimension_scores"] == {"technique": 0.8, "alliance": 0.6}
        assert data["routing_reason"] == "Detailed test"

    def test_get_item_detail_nonexistent(self, client):
        """Test GET /queue/detail with non-existent ID returns 404."""
        response = client.get("/queue/detail?item_id=999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_item_detail_alias_parameter(self, client):
        """Test GET /queue/detail works with 'item_id' parameter alias."""
        # Add an item
        item_data = {
            "sample_text": "Test",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]

        # Use 'item_id' parameter (the alias)
        response = client.get(f"/queue/detail?item_id={item_id}")
        assert response.status_code == 200
        assert response.json()["id"] == item_id


class TestConcurrentReviews:
    """Test concurrent review submission handling."""

    def test_multiple_reviews_same_item(self, client):
        """Test multiple reviews for same item - last one wins."""
        # Add an item
        item_data = {
            "sample_text": "Concurrent test",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        item_id = response.json()["id"]

        # Submit first review
        review1 = {
            "reviewer_score": 0.7,
            "notes": "First review",
            "reviewer_id": "expert_1",
        }
        response1 = client.post(f"/queue/submit?item_id={item_id}", json=review1)
        assert response1.status_code == 200
        review1_id = response1.json()["id"]

        # Submit second review
        review2 = {
            "reviewer_score": 0.9,
            "notes": "Second review",
            "reviewer_id": "expert_2",
        }
        response2 = client.post(f"/queue/submit?item_id={item_id}", json=review2)
        assert response2.status_code == 200
        review2_id = response2.json()["id"]

        # Both reviews should exist
        assert review1_id != review2_id

        # Item status should be reviewed
        detail_response = client.get(f"/queue/detail?item_id={item_id}")
        assert detail_response.json()["status"] == "reviewed"
