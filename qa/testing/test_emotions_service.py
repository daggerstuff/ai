"""
Tests for the /analyze/emotions endpoint and FHE ciphertext hash integration.

PIX-4190: Verifies that the FHE ciphertext hash from the TypeScript provider
flows through to the Python safety filter's R1 receipt system.
"""

import pytest
from fastapi.testclient import TestClient

from ai.inference.api.emotions_service import app as emotions_app
from ai.qa.validation.inference_safety_filter import _receipt_ledger

# HTTP status codes
HTTP_OK = 200
HTTP_UNPROCESSABLE = 422

# Hash length
HASH_LEN = 64


@pytest.fixture
def client():
    """Create a test client for the emotions FastAPI app."""
    return TestClient(emotions_app)


class TestEmotionsEndpoint:
    """Tests for POST /emotions."""

    def test_basic_request_returns_200(self, client: TestClient) -> None:
        """A minimal valid request returns a successful response."""
        response = client.post("/emotions", json={"text": "I feel happy"})
        assert response.status_code == HTTP_OK
        data = response.json()
        assert "emotions" in data
        assert "dimensions" in data
        assert "confidence" in data
        assert "metadata" in data

    def test_response_contains_emotion_list(self, client: TestClient) -> None:
        """Response includes a non-empty emotions list with expected fields."""
        response = client.post("/emotions", json={"text": "I feel happy"})
        data = response.json()
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) > 0
        emotion = data["emotions"][0]
        assert "type" in emotion
        assert "intensity" in emotion
        assert "confidence" in emotion

    def test_response_contains_dimensions(self, client: TestClient) -> None:
        """Response includes valence, arousal, and dominance dimensions."""
        response = client.post("/emotions", json={"text": "I feel calm"})
        data = response.json()
        dims = data["dimensions"]
        assert "valence" in dims
        assert "arousal" in dims
        assert "dominance" in dims

    def test_missing_text_returns_422(self, client: TestClient) -> None:
        """Request without required 'text' field returns 422."""
        response = client.post("/emotions", json={})
        assert response.status_code == HTTP_UNPROCESSABLE


class TestFHECiphertextHashIntegration:
    """PIX-4190: FHE ciphertext hash flows through to R1 receipt."""

    def test_fhe_hash_produces_receipt(self, client: TestClient) -> None:
        """When fhe_ciphertext_hash is provided, a receipt is emitted."""
        fhe_hash = "a" * HASH_LEN
        response = client.post(
            "/emotions",
            json={
                "text": "I feel happy today",
                "fhe_ciphertext_hash": fhe_hash,
            },
        )
        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["receipt_root_hash"] is not None
        assert len(data["receipt_root_hash"]) == HASH_LEN

    def test_fhe_hash_embedded_in_receipt(self, client: TestClient) -> None:
        """The FHE hash is bound into the receipt envelope."""
        fhe_hash = "b" * HASH_LEN
        response = client.post(
            "/emotions",
            json={
                "text": "I feel anxious",
                "fhe_ciphertext_hash": fhe_hash,
            },
        )
        assert response.status_code == HTTP_OK

        # Verify the ledger contains a receipt with our FHE hash
        assert _receipt_ledger is not None
        assert len(_receipt_ledger._receipts) > 0
        last_receipt = _receipt_ledger._receipts[-1]
        assert last_receipt.fhe_ciphertext_hash == fhe_hash

    def test_no_fhe_hash_still_emits_receipt(self, client: TestClient) -> None:
        """Without fhe_ciphertext_hash, receipt is still emitted (with zero hash)."""
        response = client.post("/emotions", json={"text": "I feel neutral"})
        assert response.status_code == HTTP_OK
        data = response.json()
        # Receipt is emitted; FHE hash defaults to zero-filled
        assert data["receipt_root_hash"] is not None

    def test_different_fhe_hash_produces_different_receipt(self, client: TestClient) -> None:
        """Different FHE hashes produce different receipt root hashes."""
        response1 = client.post(
            "/emotions",
            json={"text": "Same text", "fhe_ciphertext_hash": "1" * HASH_LEN},
        )
        response2 = client.post(
            "/emotions",
            json={"text": "Same text", "fhe_ciphertext_hash": "2" * HASH_LEN},
        )
        assert response1.status_code == HTTP_OK
        assert response2.status_code == HTTP_OK
        # Different FHE hashes should produce different receipts
        assert response1.json()["receipt_root_hash"] != response2.json()["receipt_root_hash"]


class TestEmotionsHealthEndpoint:
    """Tests for GET /emotions/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 with status info."""
        response = client.get("/emotions/health")
        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "endpoint" in data
