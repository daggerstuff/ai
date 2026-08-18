#!/usr/bin/env python3
"""Test suite for DeepRare orchestrator and sub-agents (PIX-3907)."""

from __future__ import annotations

import pytest

from ai.pkg_mera.platform.deep_rare.knowledge_base import RareDiseaseKnowledgeBase
from ai.pkg_mera.platform.deep_rare.orchestrator import ControllerOrchestrator
from ai.pkg_mera.platform.deep_rare.differential import DifferentialDiagnosisManager
from ai.pkg_mera.platform.deep_rare.schema import (
    DiagnosisResult,
    Hypothesis,
    PatientCase,
    RareDiseaseState,
    SymptomProfile,
    TestResult,
)


@pytest.fixture
def kb() -> RareDiseaseKnowledgeBase:
    return RareDiseaseKnowledgeBase()


@pytest.fixture
def orchestrator(kb: RareDiseaseKnowledgeBase) -> ControllerOrchestrator:
    return ControllerOrchestrator(kb=kb, max_iterations=5, convergence_window=3)


@pytest.fixture
def pompe_case() -> PatientCase:
    return PatientCase(
        case_id="POMPE-001",
        patient_age=3,
        patient_sex="male",
        presenting_symptoms=[
            SymptomProfile(name="muscle weakness", category="neurological", severity="severe"),
            SymptomProfile(name="hypotonia", category="neurological"),
            SymptomProfile(name="cardiomegaly", category="cardiovascular"),
            SymptomProfile(name="respiratory distress", category="respiratory"),
        ],
        available_tests=[
            TestResult(
                test_name="CK level",
                test_type="laboratory",
                status="abnormal",
                interpretation="2500 U/L",
                is_abnormal=True,
            ),
            TestResult(
                test_name="GAA enzyme assay",
                test_type="laboratory",
                status="abnormal",
                interpretation="markedly deficient",
                is_abnormal=True,
            ),
        ],
        clinical_notes="Infantile onset, severe hypotonia",
        ground_truth_diagnosis="Pompe disease",
    )


@pytest.fixture
def simple_case() -> PatientCase:
    return PatientCase(
        case_id="SIMPLE-001",
        presenting_symptoms=[
            SymptomProfile(name="muscle weakness", category="neurological"),
        ],
    )


class TestKnowledgeBase:
    def test_has_diseases(self, kb: RareDiseaseKnowledgeBase):
        assert kb.disease_count > 0

    def test_pompe_in_kb(self, kb: RareDiseaseKnowledgeBase):
        disease = kb.get_disease("Pompe Disease")
        assert disease is not None
        assert "pompe" in disease.name.lower()

    def test_search_by_symptom(self, kb: RareDiseaseKnowledgeBase):
        results = kb.search_by_symptoms(["muscle weakness"])
        assert len(results) > 0

    def test_search_by_organ(self, kb: RareDiseaseKnowledgeBase):
        results = kb.search_by_organ_system("neurological")
        assert len(results) > 0


class TestControllerOrchestrator:
    def test_diagnose_returns_result(self, orchestrator: ControllerOrchestrator, pompe_case: PatientCase):
        result = orchestrator.diagnose(pompe_case)
        assert isinstance(result, DiagnosisResult)
        assert result.case_id == "POMPE-001"
        assert result.iterations > 0

    def test_diagnose_produces_differential(self, orchestrator: ControllerOrchestrator, pompe_case: PatientCase):
        result = orchestrator.diagnose(pompe_case)
        assert result.differential is not None
        assert len(result.differential.ranked_list) > 0

    def test_diagnose_time_under_60s(self, orchestrator: ControllerOrchestrator, pompe_case: PatientCase):
        result = orchestrator.diagnose(pompe_case)
        assert result.time_seconds < 60.0, f"Expected <60s, got {result.time_seconds}s"

    def test_simple_case_runs(self, orchestrator: ControllerOrchestrator, simple_case: PatientCase):
        result = orchestrator.diagnose(simple_case)
        assert isinstance(result, DiagnosisResult)
        assert result.case_id == "SIMPLE-001"

    def test_agent_outputs_present(self, orchestrator: ControllerOrchestrator, pompe_case: PatientCase):
        result = orchestrator.diagnose(pompe_case)
        assert "symptom_analyzer" in result.agent_outputs
        assert "test_interpreter" in result.agent_outputs
        assert "literature_matcher" in result.agent_outputs

    def test_recommended_next_steps(self, orchestrator: ControllerOrchestrator, pompe_case: PatientCase):
        result = orchestrator.diagnose(pompe_case)
        assert isinstance(result.recommended_next_steps, list)


class TestDifferentialDiagnosisManager:
    def test_build_differential(self, kb: RareDiseaseKnowledgeBase):
        manager = DifferentialDiagnosisManager(kb=kb)
        state = RareDiseaseState()
        state.add_hypothesis(Hypothesis(disease_name="Disease A", posterior_probability=0.6))
        state.add_hypothesis(Hypothesis(disease_name="Disease B", posterior_probability=0.3))
        state.add_hypothesis(Hypothesis(disease_name="Disease C", posterior_probability=0.1))
        dd = manager.build_differential(state)
        assert dd.total_hypotheses_considered == 3
        top = dd.top_disease()
        assert top is not None
        assert top.disease_name == "Disease A"

    def test_pruning(self, kb: RareDiseaseKnowledgeBase):
        manager = DifferentialDiagnosisManager(kb=kb, pruning_threshold=0.2)
        state = RareDiseaseState()
        state.add_hypothesis(Hypothesis(disease_name="Keep", posterior_probability=0.5))
        state.add_hypothesis(Hypothesis(disease_name="Prune", posterior_probability=0.05))
        manager.update(state)  # normalize + prune
        dd = manager.build_differential(state)
        names = [r.disease_name for r in dd.ranked_list]
        assert "Keep" in names
        assert "Prune" not in names


if __name__ == "__main__":
    import unittest

    unittest.main()
