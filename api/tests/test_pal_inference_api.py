"""Tests for the PAL Inference Service (pal_inference_service.py).

Uses FastAPI TestClient with stub LLM clients (no network, no GPU).
Exercises all endpoints: health, select, generate, infer, and error paths.
"""

from __future__ import annotations

import importlib
import json
import os
from unittest.mock import patch

import pal_inference_service as svc
import pytest
from fastapi.testclient import TestClient
from inference_wrapper import DEFAULT_LATENCY_BUDGET_SECONDS
from pal_inference_service import (
    _build_latency_budget,
    _load_candidate_personas,
    _StubGenerator,
    _StubSelector,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Build a TestClient that re-initializes the wrapper for each test.

    Using context manager so FastAPI startup events are triggered.
    """
    with patch.dict(
        os.environ,
        {
            "PAL_SELECTOR_ENDPOINT": "",
            "PAL_GENERATOR_ENDPOINT": "",
            "PAL_LATENCY_BUDGET_SECONDS": "10.0",
        },
        clear=False,
    ):
        # Force module reimport so the startup event runs fresh
        svc._wrapper = None
        importlib.reload(svc)
        with TestClient(svc.app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_healthy(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["wrapper_initialized"] is True
        assert data["n_candidate_personas"] >= 1
        assert data["latency_budget_seconds"] >= 1.0

    def test_health_degraded_when_wrapper_fails(self) -> None:
        """When the wrapper cannot initialize, health should report degraded."""
        with patch.dict(os.environ, {"PAL_LATENCY_BUDGET_SECONDS": "-1"}, clear=False):
            svc._wrapper = None
            importlib.reload(svc)
            client = TestClient(svc.app)
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["wrapper_initialized"] is False


# ---------------------------------------------------------------------------
# select_persona (Stage 1)
# ---------------------------------------------------------------------------


class TestSelectPersona:
    def test_selects_first_persona(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["persona_string"], str)
        assert len(data["persona_string"]) > 0
        assert isinstance(data["selected_index"], int)
        assert data["latency_seconds"] >= 0.0

    def test_rejects_empty_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_rejects_missing_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={})
        assert resp.status_code == 422

    def test_selected_persona_is_valid_nl_string(self, client: TestClient) -> None:
        """The selected persona must be a natural-language string with no JSON leakage."""
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        persona = resp.json()["persona_string"]
        for char in ("{", "}", '"', "'"):
            assert char not in persona, f"JSON leakage detected: {char} in {persona!r}"

    def test_rejects_non_string_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": 123})
        assert resp.status_code == 422

    def test_latency_is_recorded(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: hello"})
        assert resp.status_code == 200
        assert resp.json()["latency_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# generate_response (Stage 2)
# ---------------------------------------------------------------------------


class TestGenerateResponse:
    def test_generates_response(self, client: TestClient) -> None:
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

    def test_rejects_empty_persona(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pal/generate",
            json={"persona_string": "", "dialogue_history": "Patient: hello"},
        )
        assert resp.status_code == 422

    def test_accepts_empty_dialogue_history(self, client: TestClient) -> None:
        """Empty history is valid — it means single-turn generation."""
        resp = client.post(
            "/api/v1/pal/generate",
            json={"persona_string": "a persona", "dialogue_history": ""},
        )
        assert resp.status_code == 200

    def test_rejects_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/generate", json={"persona_string": "p"})
        assert resp.status_code == 422

    def test_generated_response_no_json_leakage(self, client: TestClient) -> None:
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


# ---------------------------------------------------------------------------
# infer (end-to-end)
# ---------------------------------------------------------------------------


class TestInfer:
    def test_end_to_end_inference(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert "selection" in data
        assert "generation" in data
        assert data["selection"]["persona_string"]
        assert data["generation"]["response"]
        assert data["total_latency_seconds"] >= 0.0
        assert data["dialogue"] == "Patient: I feel tired."

    def test_inference_echoes_dialogue(self, client: TestClient) -> None:
        dialogue = "Patient: I have been experiencing headaches for three days."
        resp = client.post("/api/v1/pal/infer", json={"dialogue": dialogue})
        assert resp.status_code == 200
        assert resp.json()["dialogue"] == dialogue

    def test_inference_latency_within_budget(self, client: TestClient) -> None:
        """With stub clients, latency should be negligible."""
        resp = client.post("/api/v1/pal/infer", json={"dialogue": "Patient: hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_latency_seconds"] < 1.0  # generous — stubs are microseconds

    def test_rejects_empty_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_rejects_missing_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={})
        assert resp.status_code == 422

    def test_no_json_leakage_in_response(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        # Check both selection and generation
        for char in ("{", "}"):
            assert char not in data["selection"]["persona_string"]
            assert char not in data["generation"]["response"]

    def test_inference_with_vietnamese_dialogue(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pal/infer",
            json={"dialogue": "Bệnh nhân: Tôi cảm thấy mệt mỏi. Bác sĩ: Bao lâu rồi?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generation"]["response"]
        assert "mệt" in data["dialogue"] or "mỏi" in data["dialogue"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_503_when_wrapper_not_initialized(self) -> None:
        """If the wrapper fails to init, all endpoints return 503."""
        with patch.dict(os.environ, {"PAL_LATENCY_BUDGET_SECONDS": "-1"}, clear=False):
            svc._wrapper = None
            importlib.reload(svc)
            client = TestClient(svc.app)

            endpoints_and_bodies = [
                ("/health", "GET", None),
                ("/api/v1/pal/infer", "POST", {"dialogue": "test"}),
                ("/api/v1/pal/select", "POST", {"dialogue": "test"}),
                (
                    "/api/v1/pal/generate",
                    "POST",
                    {
                        "persona_string": "a persona",
                        "dialogue_history": "some history",
                    },
                ),
            ]
            for endpoint, _method, body in endpoints_and_bodies:
                if endpoint == "/health":
                    resp = client.get(endpoint)
                    assert resp.status_code == 200  # health returns degraded, not 503
                else:
                    resp = client.post(endpoint, json=body)
                    assert resp.status_code == 503, f"{endpoint} should return 503"

    def test_404_on_unknown_route(self, client: TestClient) -> None:
        resp = client.get("/api/v1/pal/nonexistent")
        assert resp.status_code == 404

    def test_method_not_allowed(self, client: TestClient) -> None:
        resp = client.get("/api/v1/pal/infer")
        assert resp.status_code == 405


# ---------------------------------------------------------------------------
# Stub clients
# ---------------------------------------------------------------------------


class TestStubClients:
    def test_stub_selector_returns_one(self) -> None:
        selector = _StubSelector()
        assert selector([{"role": "user", "content": "hi"}]) == "1"

    def test_stub_generator_returns_string(self) -> None:
        generator = _StubGenerator()
        result = generator([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


class TestLoadCandidatePersonas:
    def test_loads_from_env(self) -> None:
        personas = json.dumps([
            {"demographics": {"age": 25}, "healthcare_behavior": {}},
            {"demographics": {"age": 50}, "healthcare_behavior": {}},
        ])
        with patch.dict(os.environ, {"PAL_CANDIDATE_PERSONAS": personas}, clear=False):
            result = _load_candidate_personas()
            assert len(result) == 2
            assert result[0]["demographics"]["age"] == 25

    def test_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = _load_candidate_personas()
            assert len(result) >= 1
            assert "demographics" in result[0]

    def test_falls_back_on_invalid_json(self) -> None:
        with patch.dict(os.environ, {"PAL_CANDIDATE_PERSONAS": "not json"}, clear=False):
            result = _load_candidate_personas()
            assert len(result) >= 1

    def test_falls_back_on_empty_array(self) -> None:
        with patch.dict(os.environ, {"PAL_CANDIDATE_PERSONAS": "[]"}, clear=False):
            result = _load_candidate_personas()
            assert len(result) >= 1


class TestBuildLatencyBudget:
    def test_from_env(self) -> None:
        with patch.dict(os.environ, {"PAL_LATENCY_BUDGET_SECONDS": "5.0"}, clear=False):
            assert _build_latency_budget() == 5.0

    def test_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _build_latency_budget() == DEFAULT_LATENCY_BUDGET_SECONDS

    def test_fallback_on_invalid(self) -> None:
        with patch.dict(os.environ, {"PAL_LATENCY_BUDGET_SECONDS": "abc"}, clear=False):
            assert _build_latency_budget() == DEFAULT_LATENCY_BUDGET_SECONDS
