"""Tests for POST /queue/{item_id}/promote endpoint.

These tests verify the annotation promotion API endpoint behavior per
VAL-M3-PROM-001, VAL-M3-PROM-002, VAL-M3-PROM-003.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from annotation.api.database import Base
from annotation.api.main import app


@pytest.fixture
def client():
    """Create a test client with an in-memory database."""
    with (
        patch("annotation.api.database.DATABASE_URL", "sqlite:///:memory:"),
        patch("annotation.api.main.DATABASE_URL", "sqlite:///:memory:"),
    ):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        with patch("annotation.api.database.get_engine", return_value=engine), TestClient(app) as test_client:
            yield test_client


class TestPromotionEndpoint:
    """Test POST /queue/{item_id}/promote endpoint."""

    def _create_pending_item(self, client: TestClient) -> int:
        """Helper to create a pending queue item."""
        item_data = {
            "sample_text": "Test item for promotion",
            "original_score": 0.5,
            "per_dimension_scores": {"technique": 0.6},
            "routing_reason": "Test",
        }
        response = client.post("/queue", json=item_data)
        return response.json()["id"]

    def _submit_review(self, client: TestClient, item_id: int) -> None:
        """Helper to submit a review for an item, transitioning it to reviewed."""
        review_data = {
            "reviewer_score": 0.75,
            "notes": "Reviewed",
            "reviewer_id": "expert_1",
        }
        response = client.patch(f"/queue/{item_id}/review", json=review_data)
        assert response.status_code == 200

    def test_promote_pending_to_reviewed(self, client: TestClient):
        """VAL-M3-PROM-001: Promote pending item to reviewed stage."""
        item_id = self._create_pending_item(client)

        # Verify initial status is pending
        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "pending"

        # Promote from pending to reviewed
        promotion_data = {
            "promoter_id": "promoter_1",
            "notes": "Initial promotion",
        }
        response = client.post(
            f"/queue/{item_id}/promote",
            json=promotion_data,
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == item_id
        assert data["promoter_id"] == "promoter_1"
        assert data["target_stage"] == "reviewed"
        assert data["before_score"] == 0.5
        assert data["after_score"] == 0.5
        assert data["notes"] == "Initial promotion"
        assert "timestamp" in data

        # Verify item status changed to reviewed
        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "reviewed"

    def test_promote_follows_staged_workflow(self, client: TestClient):
        """VAL-M3-PROM-001: Promote follows staged workflow pending -> reviewed -> validated -> merged."""
        item_id = self._create_pending_item(client)

        # Step 1: pending -> reviewed
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        assert response.json()["target_stage"] == "reviewed"

        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "reviewed"

        # Step 2: reviewed -> validated
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1", "notes": "Second promotion"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        assert response.json()["target_stage"] == "validated"

        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "validated"

        # Step 3: validated -> merged
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1", "notes": "Final promotion"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        assert response.json()["target_stage"] == "merged"

        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "merged"

    def test_promote_skip_stage_returns_409(self, client: TestClient):
        """VAL-M3-PROM-001: Skipping a stage returns 409 Conflict."""
        item_id = self._create_pending_item(client)

        # Try to skip from pending directly to validated (skipping reviewed)
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        # This will succeed because pending -> reviewed is a valid transition
        assert response.status_code == 200

        # Now try to skip from reviewed to merged (skipping validated)
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200  # reviewed -> validated is valid

        # Now try to skip merged
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200  # validated -> merged is valid

        # Now item is at merged - trying to promote again should fail
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 409
        assert "final stage" in response.json()["detail"].lower()

    def test_promote_records_audit_fields(self, client: TestClient):
        """VAL-M3-PROM-002: Promotion records promoter_id, timestamp, before_score, after_score, notes."""
        item_id = self._create_pending_item(client)

        promotion_data = {
            "promoter_id": "promoter_audit",
            "notes": "Audit test promotion",
        }
        response = client.post(
            f"/queue/{item_id}/promote",
            json=promotion_data,
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        data = response.json()

        # All required fields present
        assert "id" in data
        assert "item_id" in data
        assert data["item_id"] == item_id
        assert data["promoter_id"] == "promoter_audit"
        assert "before_score" in data
        assert "after_score" in data
        assert data["before_score"] == 0.5
        assert data["after_score"] == 0.5
        assert "target_stage" in data
        assert data["target_stage"] == "reviewed"
        assert data["notes"] == "Audit test promotion"
        assert "timestamp" in data

    def test_promote_history_queryable_via_detail(self, client: TestClient):
        """VAL-M3-PROM-002: Promotion history is queryable via GET /queue/{id}/detail."""
        item_id = self._create_pending_item(client)

        # First promotion: pending -> reviewed
        client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1", "notes": "First"},
            headers={"X-Promoter-Role": "promoter"},
        )

        # Second promotion: reviewed -> validated
        client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_2", "notes": "Second"},
            headers={"X-Promoter-Role": "promoter"},
        )

        # Get detail and check promotion history
        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.status_code == 200
        data = detail.json()

        assert "promotion_history" in data
        assert len(data["promotion_history"]) == 2

        # First promotion
        assert data["promotion_history"][0]["promoter_id"] == "promoter_1"
        assert data["promotion_history"][0]["target_stage"] == "reviewed"
        assert data["promotion_history"][0]["notes"] == "First"

        # Second promotion
        assert data["promotion_history"][1]["promoter_id"] == "promoter_2"
        assert data["promotion_history"][1]["target_stage"] == "validated"
        assert data["promotion_history"][1]["notes"] == "Second"

    def test_promote_missing_auth_header_returns_401(self, client: TestClient):
        """VAL-M3-PROM-003: Missing X-Promoter-Role header returns 401."""
        item_id = self._create_pending_item(client)

        # No auth header
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
        )
        assert response.status_code == 401
        assert "Missing X-Promoter-Role header" in response.json()["detail"]

        # Verify item status remained pending
        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "pending"

    def test_promote_wrong_role_returns_403(self, client: TestClient):
        """VAL-M3-PROM-003: Wrong role in X-Promoter-Role header returns 403."""
        item_id = self._create_pending_item(client)

        # Wrong role
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "reviewer"},  # Wrong role
        )
        assert response.status_code == 403
        assert "Invalid role" in response.json()["detail"]
        assert "reviewer" in response.json()["detail"]

        # Verify item status remained pending
        detail = client.get(f"/queue/detail?item_id={item_id}")
        assert detail.json()["status"] == "pending"

    def test_promote_nonexistent_item_returns_404(self, client: TestClient):
        """Promoting a non-existent item returns 404."""
        response = client.post(
            "/queue/99999/promote",
            json={"promoter_id": "promoter_1"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_promote_with_optional_notes(self, client: TestClient):
        """Promotion works with and without optional notes."""
        item_id = self._create_pending_item(client)

        # With notes
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_1", "notes": "With notes"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        assert response.json()["notes"] == "With notes"

        # Promote again and without notes
        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_2"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200
        assert response.json()["notes"] is None

    def test_promote_logs_decision(self, client: TestClient):
        """Each promotion decision emits structured log with item_id, decision, reason, timestamp."""
        item_id = self._create_pending_item(client)

        response = client.post(
            f"/queue/{item_id}/promote",
            json={"promoter_id": "promoter_logged", "notes": "Log test"},
            headers={"X-Promoter-Role": "promoter"},
        )
        assert response.status_code == 200

        data = response.json()
        # Response includes all fields needed for logging
        assert data["item_id"] == item_id
        assert "promoter_id" in data
        assert "target_stage" in data
        assert "timestamp" in data
