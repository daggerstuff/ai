"""
Tests for PIX-3912: Mera Hierarchical Clinical Prediction endpoint.
"""

import sys
import types
from pathlib import Path

import pytest

# Mock pixel module before importing service
_PIXEL_BASE_MOD = types.ModuleType("pixel.models.pixel_base_model")
_PIXEL_BASE_MOD.PixelBaseModel = object
sys.modules["pixel.models.pixel_base_model"] = _PIXEL_BASE_MOD
sys.modules["pixel.models"] = types.ModuleType("pixel.models")
sys.modules["pixel"] = types.ModuleType("pixel")

# Mock inference_wrapper
_inference_wrapper = types.ModuleType("inference_wrapper")
_inference_wrapper.DEFAULT_LATENCY_BUDGET_SECONDS = 30.0
_inference_wrapper.JsonLeakageError = Exception
_inference_wrapper.LatencyExceededError = Exception
_inference_wrapper.SelectionParseError = Exception
_inference_wrapper.PalInferenceWrapper = object
sys.modules["inference_wrapper"] = _inference_wrapper

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.inference.api.pixel_inference_service import (
    ClinicalPredictionRequest,
    ClinicalPredictionResponse,
    MeraClinicalPredictionEngine,
)


class TestMeraClinicalPredictionEngine:
    @pytest.fixture
    def engine(self):
        eng = MeraClinicalPredictionEngine()
        eng.initialize()
        return eng

    @pytest.mark.asyncio
    async def test_predict_returns_ranked_diagnoses(self, engine):
        req = ClinicalPredictionRequest(
            patient_presentation="Patient reports depressed mood, insomnia, poor concentration, and fatigue for 3 weeks.",
            top_k=3,
        )
        resp = await engine.predict(req)
        assert isinstance(resp, ClinicalPredictionResponse)
        assert len(resp.ranked_diagnoses) == 3
        assert resp.inference_time_ms > 0
        assert resp.hierarchy_coverage > 100

        # Check ranking order
        for i, diag in enumerate(resp.ranked_diagnoses):
            assert diag.rank == i + 1
            assert diag.condition_id
            assert diag.condition_name
            assert 0.0 <= diag.final_score <= 1.0

    @pytest.mark.asyncio
    async def test_predict_with_evidence(self, engine):
        req = ClinicalPredictionRequest(
            patient_presentation="Patient reports depressed mood, insomnia, poor concentration, and fatigue for 3 weeks.",
            top_k=5,
            include_evidence=True,
        )
        resp = await engine.predict(req)
        # At least one diagnosis should have evidence (conditions with symptom leaves)
        diagnoses_with_evidence = [d for d in resp.ranked_diagnoses if d.evidence]
        assert len(diagnoses_with_evidence) > 0
        for diag in diagnoses_with_evidence:
            for ev in diag.evidence:
                assert ev.finding_type
                assert ev.description

    @pytest.mark.asyncio
    async def test_predict_with_test_results(self, engine):
        req = ClinicalPredictionRequest(
            patient_presentation="Patient reports depressed mood, insomnia, poor concentration, and fatigue for 3 weeks.",
            top_k=2,
            test_results=[
                {"name": "sleep_study", "value": "abnormal", "flag": "abnormal"},
            ],
        )
        resp = await engine.predict(req)
        assert len(resp.ranked_diagnoses) == 2

    def test_engine_status(self, engine):
        status = engine.get_status()
        assert status["initialized"] is True
        assert status["mera_available"] is True
        assert status["hierarchy_size"] > 100
