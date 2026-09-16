"""
Tests for the /analyze/emotions endpoint and FHE ciphertext hash integration.

PIX-4190: Verifies that the FHE ciphertext hash from the TypeScript provider
flows through to the Python safety filter's R1 receipt system.

PIX-4379: Real LLM-backed emotion analysis (GLM/Qwen/Mistral — no LLaMA per
standing model policy) with PII scrubbing and lexicon fallback. Endpoint
tests stub the analyzer so runs are hermetic and fast; a live-API accuracy
test runs only when RUN_EMOTION_LIVE_TESTS=1 is set.
"""

import os
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from ai.inference.api import emotions_service
from ai.inference.api.emotions_service import EmotionAnalyzer, app as emotions_app
from ai.qa.validation.inference_safety_filter import _receipt_ledger

# HTTP status codes
HTTP_OK = 200
HTTP_UNPROCESSABLE = 422
HTTP_SERVER_ERROR = 500

# Hash length
HASH_LEN = 64

# Neutral VAD midpoint used by the fallback scorer
NEUTRAL_VAD = 0.5


@pytest.fixture
def client():
    """Create a test client for the emotions FastAPI app."""
    return TestClient(emotions_app)


def _stub_analyzer(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Replace the module-level analyzer with one returning `payload`."""

    class StubAnalyzer:
        def analyze(self, _text: str, model_label: str) -> dict:
            return {
                "emotions": [dict(e) for e in payload["emotions"]],
                "dimensions": dict(payload["dimensions"]),
                "confidence": payload["confidence"],
                "metadata": {"model_version": model_label, "analysis_type": "multidimensional"},
            }

    monkeypatch.setattr(emotions_service, "get_analyzer", StubAnalyzer)


VALID_STUB_PAYLOAD = {
    "emotions": [{"type": "sadness", "intensity": 0.85, "confidence": 0.95}],
    "dimensions": {"valence": 0.1, "arousal": 0.2, "dominance": 0.15},
    "confidence": 0.9,
}


@pytest.fixture
def stubbed_client(monkeypatch: pytest.MonkeyPatch):
    """Test client with the analyzer stubbed to a deterministic payload."""
    _stub_analyzer(monkeypatch, VALID_STUB_PAYLOAD)
    return TestClient(emotions_app)


class TestEmotionsEndpoint:
    """Tests for POST /emotions."""

    def test_basic_request_returns_200(self, stubbed_client: TestClient) -> None:
        """A minimal valid request returns a successful response."""
        response = stubbed_client.post("/emotions", json={"text": "I feel happy"})
        assert response.status_code == HTTP_OK
        data = response.json()
        assert "emotions" in data
        assert "dimensions" in data
        assert "confidence" in data
        assert "metadata" in data

    def test_response_contains_emotion_list(self, stubbed_client: TestClient) -> None:
        """Response includes a non-empty emotions list with expected fields."""
        response = stubbed_client.post("/emotions", json={"text": "I feel happy"})
        data = response.json()
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) > 0
        emotion = data["emotions"][0]
        assert "type" in emotion
        assert "intensity" in emotion
        assert "confidence" in emotion

    def test_response_reflects_actual_detection(self, stubbed_client: TestClient) -> None:
        """PIX-4379: the response carries the analyzer's result, not canned data."""
        response = stubbed_client.post(
            "/emotions", json={"text": "I have been feeling hopeless all week"}
        )
        data = response.json()
        assert data["emotions"][0]["type"] == "sadness"
        assert data["dimensions"]["valence"] == pytest.approx(0.1)
        assert data["confidence"] == pytest.approx(0.9)

    def test_metadata_includes_processing_time(self, stubbed_client: TestClient) -> None:
        """Metadata carries processing time and model version."""
        response = stubbed_client.post("/emotions", json={"text": "I feel calm"})
        data = response.json()
        assert data["metadata"]["processing_time_ms"] >= 0
        assert data["metadata"]["model_version"] == "qwen-emotion-v1.0"

    def test_response_contains_dimensions(self, stubbed_client: TestClient) -> None:
        """Response includes valence, arousal, and dominance dimensions."""
        response = stubbed_client.post("/emotions", json={"text": "I feel calm"})
        data = response.json()
        dims = data["dimensions"]
        assert "valence" in dims
        assert "arousal" in dims
        assert "dominance" in dims

    def test_missing_text_returns_422(self, stubbed_client: TestClient) -> None:
        """Request without required 'text' field returns 422."""
        response = stubbed_client.post("/emotions", json={})
        assert response.status_code == HTTP_UNPROCESSABLE


class TestPIIScrubbing:
    """PIX-4379: HIPAA guard — PHI identifiers never reach the LLM."""

    def test_pii_scrubbed_flag_false_for_clean_text(self, stubbed_client: TestClient) -> None:
        """Clean clinical text passes through unscrubbed."""
        response = stubbed_client.post(
            "/emotions", json={"text": "I have been feeling overwhelmed lately"}
        )
        data = response.json()
        assert data["metadata"]["pii_scrubbed"] is False

    def test_email_in_text_is_redacted(self, stubbed_client: TestClient) -> None:
        """Email addresses are redacted before analysis (HIPAA)."""
        response = stubbed_client.post(
            "/emotions",
            json={"text": "My name is John Smith, email john.smith@example.com, I feel anxious"},
        )
        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["metadata"]["pii_scrubbed"] is True
        assert "pii_note" in data["metadata"]


class TestEmotionAnalyzerUnit:
    """Unit tests for the analyzer: LLM parse, fallback, clamping."""

    def test_parse_llm_json_plain(self) -> None:
        analyzer = EmotionAnalyzer()
        parsed = analyzer._parse_llm_json(
            '{"emotions": [], "valence": 0.5, "arousal": 0.5, '
            '"dominance": 0.5, "overall_confidence": 0.8}'
        )
        assert parsed["valence"] == NEUTRAL_VAD

    def test_parse_llm_json_fenced(self) -> None:
        analyzer = EmotionAnalyzer()
        parsed = analyzer._parse_llm_json(
            '```json\n{"emotions": [{"type": "joy", "intensity": 0.9, '
            '"confidence": 0.9}], "valence": 0.8, "arousal": 0.6, '
            '"dominance": 0.7, "overall_confidence": 0.88}\n```'
        )
        assert parsed["emotions"][0]["type"] == "joy"

    def test_parse_llm_json_with_leading_prose(self) -> None:
        analyzer = EmotionAnalyzer()
        expected_valence = 0.2
        parsed = analyzer._parse_llm_json(
            'Here is the analysis: {"valence": 0.2, "arousal": 0.3, '
            '"dominance": 0.1, "emotions": [], "overall_confidence": 0.7} '
            "hope this helps"
        )
        assert parsed["valence"] == expected_valence

    def test_parse_llm_json_rejects_garbage(self) -> None:
        analyzer = EmotionAnalyzer()
        with pytest.raises(ValueError, match="No JSON object"):
            analyzer._parse_llm_json("no json here at all")

    def test_clamp_bounds(self) -> None:
        assert EmotionAnalyzer._clamp(1.7) == 1.0
        assert EmotionAnalyzer._clamp(-0.4) == 0.0
        assert EmotionAnalyzer._clamp(None) == NEUTRAL_VAD
        assert EmotionAnalyzer._clamp("not-a-number") == NEUTRAL_VAD

    def test_lexicon_fallback_runs_without_llm(self) -> None:
        """With no API key configured, analysis falls back to the lexicon classifier."""
        analyzer = EmotionAnalyzer()
        analyzer.api_key = None
        result = analyzer.analyze("I am so happy and grateful today", "test-model")
        assert result["metadata"]["engine"] == "lexicon"
        assert 0.0 <= result["dimensions"]["valence"] <= 1.0
        assert isinstance(result["emotions"], list)

    def test_llm_failure_falls_back_to_lexicon(self) -> None:
        """When the LLM call raises, the analyzer degrades to lexicon mode."""

        class FailingClient:
            class chat:  # noqa: N801 - mirrors OpenAI client nesting
                class completions:  # noqa: N801 - mirrors OpenAI client nesting
                    @staticmethod
                    def create(**_kwargs):
                        raise RuntimeError("endpoint down")

        analyzer = EmotionAnalyzer()
        analyzer._client = FailingClient  # instance attribute shadows the method
        result = analyzer.analyze("I feel scared and alone", "test-model")
        assert result["metadata"]["engine"] == "lexicon"


class TestFHECiphertextHashIntegration:
    """PIX-4190: FHE ciphertext hash flows through to R1 receipt."""

    def test_fhe_hash_produces_receipt(self, stubbed_client: TestClient) -> None:
        """When fhe_ciphertext_hash is provided, a receipt is emitted."""
        fhe_hash = "a" * HASH_LEN
        response = stubbed_client.post(
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

    def test_fhe_hash_embedded_in_receipt(self, stubbed_client: TestClient) -> None:
        """The FHE hash is bound into the receipt envelope."""
        fhe_hash = "b" * HASH_LEN
        response = stubbed_client.post(
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

    def test_no_fhe_hash_still_emits_receipt(self, stubbed_client: TestClient) -> None:
        """Without fhe_ciphertext_hash, receipt is still emitted (with zero hash)."""
        response = stubbed_client.post("/emotions", json={"text": "I feel neutral"})
        assert response.status_code == HTTP_OK
        data = response.json()
        # Receipt is emitted; FHE hash defaults to zero-filled
        assert data["receipt_root_hash"] is not None

    def test_different_fhe_hash_produces_different_receipt(self, stubbed_client: TestClient) -> None:
        """Different FHE hashes produce different receipt root hashes."""
        response1 = stubbed_client.post(
            "/emotions",
            json={"text": "Same text", "fhe_ciphertext_hash": "1" * HASH_LEN},
        )
        response2 = stubbed_client.post(
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
        assert "llm_configured" in data


@pytest.mark.skipif(
    os.environ.get("RUN_EMOTION_LIVE_TESTS") != "1",
    reason="Live LLM accuracy test — set RUN_EMOTION_LIVE_TESTS=1 to run",
)
class TestLiveLLMAccuracy:
    """PIX-4379 acceptance: detection accuracy on therapeutic samples."""

    # (text, expected primary Plutchik category); annotated as ClassVar so the
    # shared case list is not treated as a mutable per-instance default
    CASES: ClassVar[list[tuple[str, str]]] = [
        ("I have been feeling really down and hopeless all week.", "sadness"),
        ("I am so happy and grateful for everything right now!", "joy"),
        ("She betrayed me and I will never trust her again.", "anger"),
        ("I'm terrified about the results, my heart is racing.", "fear"),
        ("Wow, I did not see that coming at all.", "surprise"),
        ("I feel totally repulsed by what he did.", "disgust"),
    ]

    # Acceptance per PIX-510 Task 3: > 85% on therapeutic set
    ACCURACY_THRESHOLD = 0.85

    def test_live_detection_accuracy(self) -> None:
        analyzer = EmotionAnalyzer()
        if not analyzer.api_key:
            pytest.skip("No LLM API key configured")
        hits = 0
        for text, expected in self.CASES:
            result = analyzer.analyze(text, "live-test")
            types = [e["type"] for e in result["emotions"]]
            if expected in types:
                hits += 1
        accuracy = hits / len(self.CASES)
        assert accuracy > self.ACCURACY_THRESHOLD, f"accuracy {accuracy:.0%} below threshold"
