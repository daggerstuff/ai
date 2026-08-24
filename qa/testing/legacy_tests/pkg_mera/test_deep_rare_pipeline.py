#!/usr/bin/env python3
"""Test suite for DeepRare pipeline end-to-end (PIX-3907)."""

from __future__ import annotations

import json
import tempfile

import pytest

from ai.tools.utilities.platform.deep_rare.pipeline import PipelineConfig, RareDiseasePipeline
from ai.tools.utilities.platform.deep_rare.schema import (
    EvaluationMetrics,
    PatientCase,
    SymptomProfile,
    TestResult,
)


@pytest.fixture
def pipeline() -> RareDiseasePipeline:
    return RareDiseasePipeline()


@pytest.fixture
def pompe_case() -> PatientCase:
    return PatientCase(
        case_id="PIPE-POMPE",
        patient_age=2,
        patient_sex="female",
        presenting_symptoms=[
            SymptomProfile(name="muscle weakness", category="neurological", severity="severe"),
            SymptomProfile(name="hypotonia", category="neurological"),
            SymptomProfile(name="cardiomegaly", category="cardiovascular"),
        ],
        available_tests=[
            TestResult(test_name="CK level", test_type="laboratory", status="abnormal", is_abnormal=True),
        ],
        ground_truth_diagnosis="Pompe disease",
    )


@pytest.fixture
def multiple_cases() -> list[PatientCase]:
    return [
        PatientCase(
            case_id="MC-001",
            presenting_symptoms=[
                SymptomProfile(name="muscle weakness", category="neurological"),
                SymptomProfile(name="hypotonia", category="neurological"),
            ],
            ground_truth_diagnosis="Pompe disease",
        ),
        PatientCase(
            case_id="MC-002",
            presenting_symptoms=[
                SymptomProfile(name="ataxia", category="neurological"),
                SymptomProfile(name="dysarthria", category="neurological"),
            ],
            ground_truth_diagnosis="Friedreich ataxia",
        ),
        PatientCase(
            case_id="MC-003",
            presenting_symptoms=[
                SymptomProfile(name="tall stature", category="other"),
                SymptomProfile(name="aortic dilatation", category="cardiovascular"),
            ],
            ground_truth_diagnosis="Marfan syndrome",
        ),
    ]


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.max_iterations == 10
        assert config.convergence_window == 3
        assert config.pruning_threshold == 0.01
        assert config.enable_evaluation is True

    def test_custom_values(self):
        config = PipelineConfig(max_iterations=5, convergence_window=2, pruning_threshold=0.05)
        assert config.max_iterations == 5
        assert config.convergence_window == 2

    def test_invalid_iterations(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PipelineConfig(max_iterations=0)


class TestRareDiseasePipeline:
    def test_diagnose_single(self, pipeline: RareDiseasePipeline, pompe_case: PatientCase):
        result = pipeline.diagnose(pompe_case)
        assert result.case_id == "PIPE-POMPE"
        assert result.differential is not None
        assert len(result.differential.ranked_list) > 0

    def test_diagnose_batch(self, pipeline: RareDiseasePipeline, multiple_cases: list[PatientCase]):
        results = pipeline.diagnose_batch(multiple_cases)
        assert len(results) == 3
        assert all(r.case_id.startswith("MC-") for r in results)

    def test_evaluate(self, pipeline: RareDiseasePipeline, multiple_cases: list[PatientCase]):
        metrics = pipeline.evaluate(multiple_cases)
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_cases == 3
        assert metrics.avg_time_seconds > 0
        assert 0.0 <= metrics.recall_at_1 <= 1.0

    def test_get_info(self, pipeline: RareDiseasePipeline):
        info = pipeline.get_info()
        assert info["version"] == "1.0.0"
        assert "config" in info
        assert "knowledge_base" in info
        assert "symptom_analyzer" in info["agents"]

    def test_load_cases_from_json(self, pipeline: RareDiseasePipeline, pompe_case: PatientCase):
        case_dict = pompe_case.model_dump()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([case_dict], f)
            f.flush()
            cases = pipeline.load_cases_from_file(f.name)
            assert len(cases) == 1
            assert cases[0].case_id == "PIPE-POMPE"

    def test_load_cases_from_jsonl(self, pipeline: RareDiseasePipeline, pompe_case: PatientCase):
        case_dict = pompe_case.model_dump()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(case_dict) + "\n")
            f.flush()
            cases = pipeline.load_cases_from_file(f.name)
            assert len(cases) == 1
            assert cases[0].case_id == "PIPE-POMPE"

    def test_custom_config(self):
        config = PipelineConfig(max_iterations=3, convergence_window=2)
        pipe = RareDiseasePipeline(config=config)
        info = pipe.get_info()
        assert info["config"]["max_iterations"] == 3

    def test_pompe_top_diagnosis(self, pipeline: RareDiseasePipeline, pompe_case: PatientCase):
        """Pompe disease should appear in the differential for a case with classic Pompe symptoms."""
        result = pipeline.diagnose(pompe_case)
        disease_names = [r.disease_name.lower() for r in result.differential.ranked_list]
        assert any("pompe" in name for name in disease_names), f"Pompe not in differential: {disease_names}"


if __name__ == "__main__":
    import unittest

    unittest.main()
