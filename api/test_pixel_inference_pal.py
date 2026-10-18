"""Tests for PAL endpoints wired into pixel_inference_service.py.

Uses importlib + sys.modules patching to work around the pre-existing
``pixel`` module dependency (not required for PAL endpoints).

The PAL endpoint logic is structurally identical to the standalone
``pal_inference_service.py`` (32 tests already passing). These tests
verify that the PAL endpoints are correctly mounted on the existing
Pixel inference FastAPI app.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock the 'pixel' module so ``pixel_inference_service`` can be imported
# without the pre-existing ``pixel`` package dependency.
# ---------------------------------------------------------------------------


class _MockPixelBaseModel:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._loaded = True

    def to(self, _device: object) -> _MockPixelBaseModel:
        return self

    def eval(self) -> None:
        pass

    @classmethod
    def load(cls, _path: str) -> _MockPixelBaseModel:
        return cls()

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


class _MockTorchTensor:
    pass


class _MockTorchNoGrad:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


class _MockCuda:
    """Mock for torch.cuda module."""

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def current_device() -> str:
        return "cpu"


class _MockTorch:
    """Stand-in for the ``torch`` object expected by pixel_inference_service."""

    Tensor = _MockTorchTensor
    cuda = _MockCuda()

    @staticmethod
    def device(*_args: object) -> str:
        return "cpu"

    @staticmethod
    def tensor(*_args: object) -> object:
        return None

    @staticmethod
    def randn(*_args: object, **_kwargs: object) -> object:
        return None

    no_grad = _MockTorchNoGrad


# Build the mock module hierarchy
_TORCH_PROXY_MOD = type(sys)("torch_proxy")
_TORCH_PROXY_MOD.torch = _MockTorch()
_TORCH_PROXY_MOD.__name__ = "ai.utils.torch_proxy"

_AI_UTILS_MOD = type(sys)("utils")
_AI_UTILS_MOD.torch_proxy = _TORCH_PROXY_MOD
_AI_UTILS_MOD.__name__ = "ai.utils"

_AI_MOD = type(sys)("ai")
_AI_MOD.utils = _AI_UTILS_MOD
_AI_MOD.__name__ = "ai"


class _MockMemoryManager:
    async def trigger_dream_cycle(self, user_id: str) -> dict[str, object]:
        return {"dream_id": "mock", "themes": [], "patterns": []}


def _mock_get_memory_manager() -> _MockMemoryManager:
    return _MockMemoryManager()


def _mock_init_sentry(*_args: object, **_kwargs: object) -> None:
    pass


_AI_API_MEMORY_MOD = type(sys)("memory")
_AI_API_MEMORY_MOD.get_memory_manager = _mock_get_memory_manager
_AI_API_MEMORY_MOD.__name__ = "ai.api.memory"

_AI_API_SENTRY_MOD = type(sys)("sentry_logging")
_AI_API_SENTRY_MOD.initialize_sentry_logging = _mock_init_sentry
_AI_API_SENTRY_MOD.__name__ = "ai.api.sentry_logging"

_AI_API_MOD = type(sys)("api")
_AI_API_MOD.memory = _AI_API_MEMORY_MOD
_AI_API_MOD.sentry_logging = _AI_API_SENTRY_MOD
_AI_API_MOD.__name__ = "ai.api"

_PIXEL_BASE_MOD = type(sys)("pixel_base_model")
_PIXEL_BASE_MOD.PixelBaseModel = _MockPixelBaseModel
_PIXEL_BASE_MOD.__name__ = "pixel.models.pixel_base_model"

_PIXEL_MODELS_MOD = type(sys)("models")
_PIXEL_MODELS_MOD.pixel_base_model = _PIXEL_BASE_MOD
_PIXEL_MODELS_MOD.__name__ = "pixel.models"

_PIXEL_MOD = type(sys)("pixel")
_PIXEL_MOD.models = _PIXEL_MODELS_MOD
_PIXEL_MOD.__name__ = "pixel"

# Register all mocks in sys.modules before any import attempts
for mod_name, mod_val in [
    ("pixel", _PIXEL_MOD),
    ("pixel.models", _PIXEL_MODELS_MOD),
    ("pixel.models.pixel_base_model", _PIXEL_BASE_MOD),
    ("ai", _AI_MOD),
    ("ai.utils", _AI_UTILS_MOD),
    ("ai.utils.torch_proxy", _TORCH_PROXY_MOD),
    ("ai.api", _AI_API_MOD),
    ("ai.api.memory", _AI_API_MEMORY_MOD),
    ("ai.api.sentry_logging", _AI_API_SENTRY_MOD),
]:
    sys.modules[mod_name] = mod_val


@pytest.fixture(autouse=True)
def _reload_module() -> None:
    """Ensure a fresh import of pixel_inference_service before each test."""
    for key in list(sys.modules):
        if any(
            name in key
            for name in [
                "pixel_inference_service",
                "inference_wrapper",
                "generate_selection_dataset",
                "generate_sft_dialogue",
                "meddies_to_pal",
            ]
        ):
            del sys.modules[key]


@pytest.fixture
def client() -> TestClient:
    """Build a TestClient with PAL wrapper initialized (stub clients)."""
    with patch.dict(
        os.environ,
        {
            "PAL_SELECTOR_ENDPOINT": "",
            "PAL_GENERATOR_ENDPOINT": "",
            "PAL_LATENCY_BUDGET_SECONDS": "10.0",
        },
        clear=False,
    ):
        import pixel_inference_service as svc

        svc.pal_wrapper = None
        importlib.reload(svc)
        with TestClient(svc.app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# PAL select endpoint tests
# ---------------------------------------------------------------------------


class TestPalSelect:
    def test_select_persona(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["persona_string"], str)
        assert len(data["persona_string"]) > 0
        assert isinstance(data["selected_index"], int)
        assert data["latency_seconds"] >= 0.0

    def test_select_empty_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_select_missing_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={})
        assert resp.status_code == 422

    def test_select_no_json_leakage(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/select", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        persona = resp.json()["persona_string"]
        for char in ("{", "}", '"', "'"):
            assert char not in persona, f"JSON leakage detected: {char} in {persona!r}"


# ---------------------------------------------------------------------------
# PAL generate endpoint tests
# ---------------------------------------------------------------------------


class TestPalGenerate:
    def test_generate_response(self, client: TestClient) -> None:
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

    def test_generate_empty_persona(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pal/generate",
            json={"persona_string": "", "dialogue_history": "Patient: hello"},
        )
        assert resp.status_code == 422

    def test_generate_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/generate", json={"persona_string": "p"})
        assert resp.status_code == 422

    def test_generate_no_json_leakage(self, client: TestClient) -> None:
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
# PAL end-to-end infer endpoint tests
# ---------------------------------------------------------------------------


class TestPalInfer:
    def test_end_to_end(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={"dialogue": "Patient: I feel tired."})
        assert resp.status_code == 200
        data = resp.json()
        assert "selection" in data
        assert "generation" in data
        assert data["selection"]["persona_string"]
        assert data["generation"]["response"]
        assert data["total_latency_seconds"] >= 0.0
        assert data["dialogue"] == "Patient: I feel tired."

    def test_empty_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={"dialogue": ""})
        assert resp.status_code == 422

    def test_missing_dialogue(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pal/infer", json={})
        assert resp.status_code == 422

    def test_vietnamese(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pal/infer",
            json={"dialogue": "B\u1ec7nh nh\u00e2n: T\u00f4i c\u1ea3m th\u1ea5y m\u1ec7t m\u1ecfi. B\u00e1c s\u0129: Bao l\u00e2u r\u1ed3i?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generation"]["response"]
        assert "m\u1ec7t" in data["dialogue"] or "m\u1ecfi" in data["dialogue"]


# ---------------------------------------------------------------------------
# Health endpoint (PAL integration)
# ---------------------------------------------------------------------------


class TestHealthPalIntegration:
    def test_health_includes_pal_info(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "pal" in data
        assert "wrapper_initialized" in data["pal"]
        assert "n_candidate_personas" in data["pal"]
        assert "latency_budget_seconds" in data["pal"]
        assert data["pal"]["wrapper_initialized"] is True
        assert data["pal"]["n_candidate_personas"] >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPalErrorHandling:
    def test_404_on_unknown(self, client: TestClient) -> None:
        resp = client.get("/api/v1/pal/nonexistent")
        assert resp.status_code == 404

    def test_method_not_allowed(self, client: TestClient) -> None:
        resp = client.get("/api/v1/pal/infer")
        assert resp.status_code == 405
