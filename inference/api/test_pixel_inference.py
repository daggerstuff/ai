"""
Integration tests for Pixel Model Inference Service

Tests cover:
- Model loading and initialization
- Inference endpoint functionality
- EQ scoring and metrics
- Crisis detection integration
- Performance requirements (<200ms)
- Batch processing
- Error handling and recovery
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add PAL framework path for test imports
_PAL_FRAMEWORK = str(Path(__file__).resolve().parents[1] / "training_corpus" / "wrapper" / "pal_framework")
if _PAL_FRAMEWORK not in sys.path:
    sys.path.insert(0, _PAL_FRAMEWORK)

from ai.inference.api.pixel_inference_service import (
    ConversationMessage,
    PixelInferenceEngine,
    PixelInferenceRequest,
)


class TestPixelInferenceEngine:
    """Test suite for PixelInferenceEngine"""

    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return PixelInferenceEngine()

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.inference_count == 0
        assert engine.total_inference_time == 0.0
        assert not engine.model_loaded

    def test_model_loading(self, engine):
        """Test model loading"""
        # Model should load (creates fresh if file not found)
        success = engine.load_model()
        assert success
        assert engine.model_loaded
        assert engine.model is not None

    def test_model_loaded_idempotent(self, engine):
        """Test that loading twice returns early"""
        engine.load_model()
        first_model = engine.model
        engine.load_model()
        second_model = engine.model
        # Same model instance
        assert first_model is second_model

    @pytest.mark.asyncio
    async def test_inference_basic(self, engine):
        """Test basic inference"""
        engine.load_model()

        request = PixelInferenceRequest(
            user_query="I'm feeling anxious about work",
            conversation_history=[],
            use_eq_awareness=True,
            include_metrics=True,
        )

        response = await engine.generate_response(request)

        assert response is not None
        assert len(response.response) > 0
        assert response.inference_time_ms > 0
        assert response.persona_mode in ["therapy", "assistant"]
        assert response.confidence > 0
        assert engine.inference_count == 1

    @pytest.mark.asyncio
    async def test_inference_latency_requirement(self, engine):
        """Test that inference meets <200ms latency requirement"""
        engine.load_model()

        request = PixelInferenceRequest(
            user_query="Help me understand this emotion",
            conversation_history=[],
        )

        response = await engine.generate_response(request)

        assert response.inference_time_ms < 250  # Allow 50ms buffer
        assert response.warning is None or "exceeded" not in response.warning.lower()

    @pytest.mark.asyncio
    async def test_inference_with_conversation_history(self, engine):
        """Test inference with multi-turn conversation"""
        engine.load_model()

        history = [
            ConversationMessage(role="user", content="I've been stressed lately"),
            ConversationMessage(
                role="assistant",
                content="I understand. Let's explore what's causing the stress.",
            ),
            ConversationMessage(role="user", content="It's mostly work-related"),
        ]

        request = PixelInferenceRequest(
            user_query="How can I manage this better?",
            conversation_history=history,
        )

        response = await engine.generate_response(request)

        assert response is not None
        assert len(response.response) > 0

    @pytest.mark.asyncio
    async def test_eq_scores_extraction(self, engine):
        """Test EQ scores are properly extracted"""
        engine.load_model()

        request = PixelInferenceRequest(
            user_query="I understand how you feel",
            use_eq_awareness=True,
        )

        response = await engine.generate_response(request)

        assert response.eq_scores is not None
        assert 0 <= response.eq_scores.emotional_awareness <= 1
        assert 0 <= response.eq_scores.empathy_recognition <= 1
        assert 0 <= response.eq_scores.emotional_regulation <= 1
        assert 0 <= response.eq_scores.social_cognition <= 1
        assert 0 <= response.eq_scores.interpersonal_skills <= 1
        assert 0 <= response.eq_scores.overall_eq <= 1

    @pytest.mark.asyncio
    async def test_crisis_detection_integration(self, engine):
        """Test crisis signal detection"""
        engine.load_model()

        crisis_queries = [
            "I want to hurt myself",
            "I'm suicidal",
            "I can't take it anymore",
        ]

        for query in crisis_queries:
            request = PixelInferenceRequest(
                user_query=query,
                include_metrics=True,
            )

            response = await engine.generate_response(request)

            # Crisis signals should be detected
            if response.conversation_metadata:
                # At least one should indicate concern
                assert response.conversation_metadata.safety_score < 1.0

    @pytest.mark.asyncio
    async def test_persona_mode_detection(self, engine):
        """Test persona mode detection based on context"""
        engine.load_model()

        # Crisis context should use therapy mode
        crisis_request = PixelInferenceRequest(
            user_query="I'm in crisis",
            context_type="crisis",
        )

        crisis_response = await engine.generate_response(crisis_request)
        assert crisis_response.persona_mode == "therapy"

        # Educational context could be assistant
        edu_request = PixelInferenceRequest(
            user_query="What is cognitive behavioral therapy?",
            context_type="educational",
        )

        edu_response = await engine.generate_response(edu_request)
        assert edu_response.persona_mode in ["therapy", "assistant"]

    @pytest.mark.asyncio
    async def test_conversation_metadata(self, engine):
        """Test conversation metadata generation"""
        engine.load_model()

        request = PixelInferenceRequest(
            user_query="I've been practicing CBT techniques and DBT skills",
            include_metrics=True,
        )

        response = await engine.generate_response(request)

        assert response.conversation_metadata is not None
        assert isinstance(response.conversation_metadata.detected_techniques, list)
        assert response.conversation_metadata.technique_consistency >= 0
        assert response.conversation_metadata.bias_score >= 0
        assert response.conversation_metadata.safety_score >= 0
        assert response.conversation_metadata.therapeutic_effectiveness_score >= 0

    @pytest.mark.asyncio
    async def test_batch_inference_consistency(self, engine):
        """Test batch processing consistency"""
        engine.load_model()

        queries = [
            "I feel overwhelmed",
            "How do I manage anxiety?",
            "I need support",
        ]

        responses = []
        for query in queries:
            request = PixelInferenceRequest(user_query=query)
            response = await engine.generate_response(request)
            responses.append(response)

        assert len(responses) == 3
        assert all(r.response for r in responses)
        assert engine.inference_count >= 3

    @pytest.mark.asyncio
    async def test_inference_without_metrics(self, engine):
        """Test inference without detailed metrics"""
        engine.load_model()

        request = PixelInferenceRequest(
            user_query="Hello",
            include_metrics=False,
            use_eq_awareness=False,
        )

        response = await engine.generate_response(request)

        assert response.response is not None
        assert response.eq_scores is None
        assert response.conversation_metadata is None

    def test_model_status(self, engine):
        """Test model status reporting"""
        engine.load_model()

        # Run an inference to populate stats
        asyncio.run(engine.generate_response(PixelInferenceRequest(user_query="test")))

        status = engine.get_status()

        assert status.model_loaded
        assert status.model_name == "PixelBaseModel"
        assert status.inference_engine == "PyTorch"
        assert status.performance_metrics.get("inference_count", 0) >= 1
        assert status.performance_metrics.get("average_inference_time_ms") is not None
        assert "eq_measurement" in status.available_features
        assert "crisis_detection" in status.available_features
        assert "bias_detection" in status.available_features

    @pytest.mark.asyncio
    async def test_error_handling_malformed_input(self, engine):
        """Test error handling with invalid input"""
        engine.load_model()

        # Empty query should still work
        request = PixelInferenceRequest(user_query="")
        response = await engine.generate_response(request)

        assert response is not None

    @pytest.mark.asyncio
    async def test_context_type_routing(self, engine):
        """Test proper context type routing"""
        engine.load_model()

        context_types = [
            "crisis",
            "clinical",
            "educational",
            "support",
            "informational",
        ]

        for ctx_type in context_types:
            request = PixelInferenceRequest(
                user_query=f"Test query for {ctx_type}",
                context_type=ctx_type,
            )

            response = await engine.generate_response(request)

            assert response is not None
            assert response.persona_mode is not None

    @pytest.mark.asyncio
    async def test_concurrent_inference(self, engine):
        """Test concurrent inference requests"""
        engine.load_model()

        async def run_inference(query: str):
            request = PixelInferenceRequest(user_query=query)
            return await engine.generate_response(request)

        # Run 5 concurrent inferences
        queries = [
            "Query 1",
            "Query 2",
            "Query 3",
            "Query 4",
            "Query 5",
        ]

        responses = await asyncio.gather(*[run_inference(q) for q in queries])

        assert len(responses) == 5
        assert all(r.response for r in responses)
        assert engine.inference_count == 5

    def test_device_selection(self, engine):
        """Test that engine selects appropriate device"""
        engine.load_model()

        assert engine.device is not None
        assert str(engine.device) in {"cpu", "cuda:0"}


class TestPixelInferencePerformance:
    """Performance benchmarking tests"""

    @pytest.fixture
    def engine(self):
        """Create engine instance"""
        engine = PixelInferenceEngine()
        engine.load_model()
        return engine

    @pytest.mark.asyncio
    async def test_latency_distribution(self, engine):
        """Test latency meets requirements across multiple runs"""
        latencies = []

        for _ in range(10):
            request = PixelInferenceRequest(user_query="Test query")
            response = await engine.generate_response(request)
            latencies.append(response.inference_time_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        # Should meet performance SLOs
        assert avg_latency < 200, f"Average latency too high: {avg_latency}ms"
        assert p95_latency < 250, f"P95 latency too high: {p95_latency}ms"
        assert p99_latency < 300, f"P99 latency too high: {p99_latency}ms"

    @pytest.mark.asyncio
    async def test_throughput(self, engine):
        """Test inference throughput"""
        queries = [f"Query {i}" for i in range(20)]

        start_time = time.time()

        for query in queries:
            request = PixelInferenceRequest(user_query=query)
            await engine.generate_response(request)

        duration = time.time() - start_time
        throughput = len(queries) / duration

        # Should handle at least 5 queries per second
        assert throughput > 5, f"Throughput too low: {throughput} queries/sec (expected >5)"


# ---------------------------------------------------------------------------
# PAL Endpoint Tests
# ---------------------------------------------------------------------------


class TestPalEndpoints:
    """Test suite for PAL inference endpoints on the Pixel service."""

    @pytest.fixture
    def client(self):
        """Build a TestClient with PAL wrapper initialized (stub clients)."""
        import importlib
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "PAL_SELECTOR_ENDPOINT": "",
                "PAL_GENERATOR_ENDPOINT": "",
                "PAL_LATENCY_BUDGET_SECONDS": "10.0",
            },
            clear=False,
        ):
            import ai.inference.api.pixel_inference_service as svc

            svc.pal_wrapper = None
            importlib.reload(svc)
            from fastapi.testclient import TestClient

            with TestClient(svc.app) as tc:
                yield tc

    def test_pal_select_persona(self, client) -> None:
        """POST /api/v1/pal/select returns 200 with valid input."""
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["persona_string"], str)
        assert len(data["persona_string"]) > 0
        assert isinstance(data["selected_index"], int)
        assert data["latency_seconds"] >= 0.0

    def test_pal_select_empty_dialogue(self, client) -> None:
        """POST /api/v1/pal/select with empty dialogue returns 422."""
        resp = client.post("/api/v1/pal/select", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_pal_select_missing_dialogue(self, client) -> None:
        """POST /api/v1/pal/select with missing dialogue returns 422."""
        resp = client.post("/api/v1/pal/select", json={})
        assert resp.status_code == 422

    def test_pal_generate_response(self, client) -> None:
        """POST /api/v1/pal/generate returns 200 with valid input."""
        resp = client.post(
            "/api/v1/pal/generate",
            json={
                "persona_string": "This patient is a 45-year-old female from Hanoi.",
                "dialogue_history": "Patient: I feel tired.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert data["latency_seconds"] >= 0.0

    def test_pal_generate_empty_persona(self, client) -> None:
        """POST /api/v1/pal/generate with empty persona returns 422."""
        resp = client.post(
            "/api/v1/pal/generate",
            json={"persona_string": "", "dialogue_history": "Patient: hello"},
        )
        assert resp.status_code == 422

    def test_pal_generate_missing_fields(self, client) -> None:
        """POST /api/v1/pal/generate with missing fields returns 422."""
        resp = client.post("/api/v1/pal/generate", json={"persona_string": "p"})
        assert resp.status_code == 422

    def test_pal_end_to_end(self, client) -> None:
        """POST /api/v1/pal/infer returns 200 with valid input."""
        resp = client.post("/api/v1/pal/infer", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert "selection" in data
        assert "generation" in data
        assert data["selection"]["persona_string"]
        assert data["generation"]["response"]
        assert data["total_latency_seconds"] >= 0.0
        assert data["dialogue"] == "Patient: I feel tired."

    def test_pal_infer_empty_dialogue(self, client) -> None:
        """POST /api/v1/pal/infer with empty dialogue returns 422."""
        resp = client.post("/api/v1/pal/infer", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_pal_infer_missing_dialogue(self, client) -> None:
        """POST /api/v1/pal/infer with missing dialogue returns 422."""
        resp = client.post("/api/v1/pal/infer", json={})
        assert resp.status_code == 422

    def test_pal_infer_vietnamese(self, client) -> None:
        """POST /api/v1/pal/infer handles Vietnamese dialogue."""
        resp = client.post(
            "/api/v1/pal/infer",
            json={"dialogue": "Bệnh nhân: Tôi cảm thấy mệt mỏi. Bác sĩ: Bao lâu rồi?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generation"]["response"]
        assert "mệt" in data["dialogue"] or "mỏi" in data["dialogue"]

    def test_pal_health_contains_pal_info(self, client) -> None:
        """GET /health includes PAL health info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "pal" in data
        assert "wrapper_initialized" in data["pal"]
        assert "n_candidate_personas" in data["pal"]
        assert "latency_budget_seconds" in data["pal"]
        assert data["pal"]["wrapper_initialized"] is True
        assert data["pal"]["n_candidate_personas"] >= 1

    def test_pal_select_no_json_leakage(self, client) -> None:
        """Selected persona must contain no JSON formatting characters."""
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        persona = resp.json()["persona_string"]
        for char in ("{", "}", '"', "'"):
            assert char not in persona, f"JSON leakage detected: {char} in {persona!r}"

    def test_pal_generate_no_json_leakage(self, client) -> None:
        """Generated response must contain no JSON formatting characters."""
        resp = client.post(
            "/api/v1/pal/generate",
            json={
                "persona_string": "This patient is a 45-year-old female from Hanoi.",
                "dialogue_history": "Patient: I feel tired.",
            },
        )
        assert resp.status_code == 200
        response = resp.json()["response"]
        for char in ("{", "}"):
            assert char not in response, f"JSON leakage detected: {char} in {response!r}"

    def test_pal_404_on_unknown(self, client) -> None:
        """Unknown PAL routes return 404."""
        resp = client.get("/api/v1/pal/nonexistent")
        assert resp.status_code == 404

    def test_pal_method_not_allowed(self, client) -> None:
        """GET on POST-only PAL endpoint returns 405."""
        resp = client.get("/api/v1/pal/infer")
        assert resp.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
