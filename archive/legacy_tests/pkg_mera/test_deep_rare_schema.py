#!/usr/bin/env python3
"""Test suite for DeepRare schema models (PIX-3907)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai.pkg_mera.platform.deep_rare.schema import (
    DiagnosisResult,
    DifferentialDiagnosis,
    EvaluationMetrics,
    Evidence,
    Hypothesis,
    PatientCase,
    RankedDiagnosis,
    RareDiseaseState,
    SymptomProfile,
    TestResult,
)


class TestSymptomProfile:
    def test_creation(self):
        s = SymptomProfile(name="Muscle weakness", category="neurological")
        assert s.name == "Muscle weakness"
        assert s.category == "neurological"
        assert s.onset == "unknown"
        assert s.severity == "unknown"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            SymptomProfile.model_validate({"name": "test", "category": "not_a_category"})

    def test_pathognomonic_flag(self):
        s = SymptomProfile(name="Cherry-red spot", category="ophthalmological", is_pathognomonic=True)
        assert s.is_pathognomonic is True

    def test_negative_duration_raises(self):
        with pytest.raises(ValidationError):
            SymptomProfile(name="test", category="neurological", duration_days=-1.0)


class TestTestResult:
    def test_creation(self):
        t = TestResult(test_name="CK level", test_type="laboratory", status="abnormal")
        assert t.test_name == "CK level"
        assert t.test_type == "laboratory"
        assert t.status == "abnormal"
        assert t.is_abnormal is False

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            TestResult(test_name="  ", test_type="laboratory", status="normal")


class TestEvidence:
    def test_creation(self):
        e = Evidence(source="symptom_analyzer", description="Pathognomonic match", supports=True)
        assert e.source == "symptom_analyzer"
        assert e.supports is True
        assert e.weight == 0.5

    def test_weight_out_of_range(self):
        with pytest.raises(ValidationError):
            Evidence(source="test_interpreter", description="test", supports=False, weight=1.5)

    def test_weight_boundary(self):
        e = Evidence(source="orchestrator", description="test", supports=True, weight=0.0)
        assert e.weight == 0.0
        e2 = Evidence(source="orchestrator", description="test", supports=True, weight=1.0)
        assert e2.weight == 1.0


class TestHypothesis:
    def test_creation(self):
        h = Hypothesis(disease_name="Pompe disease")
        assert h.disease_name == "Pompe disease"
        assert h.status == "active"
        assert h.prior_probability == 0.01
        assert h.evidence_list == []

    def test_add_evidence(self):
        h = Hypothesis(disease_name="Test disease")
        e = Evidence(source="symptom_analyzer", description="match", supports=True)
        h.add_evidence(e)
        assert len(h.evidence_list) == 1
        assert len(h.supporting_evidence()) == 1
        assert len(h.refuting_evidence()) == 0

    def test_refuting_evidence(self):
        h = Hypothesis(disease_name="Test")
        h.add_evidence(Evidence(source="test_interpreter", description="no match", supports=False))
        assert len(h.refuting_evidence()) == 1
        assert len(h.supporting_evidence()) == 0

    def test_invalid_probability(self):
        with pytest.raises(ValidationError):
            Hypothesis(disease_name="test", prior_probability=1.5)

    def test_status_literal(self):
        for status in ("active", "eliminated", "confirmed", "pending_verification"):
            h = Hypothesis(disease_name="test", status=status)
            assert h.status == status

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            Hypothesis.model_validate({"disease_name": "test", "status": "not_a_status"})


class TestRareDiseaseState:
    def test_empty_creation(self):
        state = RareDiseaseState()
        assert state.iteration == 0
        assert state.active_hypotheses == []
        assert state.eliminated_conditions == []
        assert state.is_converged is False

    def test_add_hypothesis(self):
        state = RareDiseaseState()
        h = Hypothesis(disease_name="Test")
        state.add_hypothesis(h)
        assert len(state.active_hypotheses) == 1

    def test_eliminate(self):
        state = RareDiseaseState()
        state.add_hypothesis(Hypothesis(disease_name="Disease A"))
        state.add_hypothesis(Hypothesis(disease_name="Disease B"))
        state.eliminate("Disease A")
        assert len(state.active_hypotheses) == 1
        assert state.active_hypotheses[0].disease_name == "Disease B"
        assert "Disease A" in state.eliminated_conditions

    def test_add_and_resolve_inquiry(self):
        state = RareDiseaseState()
        state.add_inquiry("Check CK levels")
        assert len(state.pending_inquiries) == 1
        state.resolve_inquiry("Check CK levels")
        assert len(state.pending_inquiries) == 0

    def test_convergence_false_short_history(self):
        state = RareDiseaseState(convergence_window=3)
        state.record_top_hypotheses()
        assert state.check_convergence() is False

    def test_convergence_true_stable(self):
        state = RareDiseaseState(convergence_window=3)
        state.add_hypothesis(Hypothesis(disease_name="A", posterior_probability=0.5))
        state.add_hypothesis(Hypothesis(disease_name="B", posterior_probability=0.3))
        for _ in range(3):
            state.record_top_hypotheses()
        assert state.check_convergence() is True

    def test_to_dict(self):
        state = RareDiseaseState()
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["iteration"] == 0


class TestPatientCase:
    def test_minimal_creation(self):
        case = PatientCase(case_id="CASE-001")
        assert case.case_id == "CASE-001"
        assert case.patient_sex == "unknown"
        assert case.presenting_symptoms == []

    def test_full_creation(self):
        case = PatientCase(
            case_id="CASE-002",
            patient_age=35,
            patient_sex="male",
            presenting_symptoms=[
                SymptomProfile(name="Muscle weakness", category="neurological"),
            ],
            medical_history=["hypertension"],
            family_history=["unknown"],
            presentation_complexity="moderate",
            ground_truth_diagnosis="Pompe disease",
        )
        assert case.patient_age == 35
        assert len(case.presenting_symptoms) == 1
        assert case.ground_truth_diagnosis == "Pompe disease"

    def test_invalid_age(self):
        with pytest.raises(ValidationError):
            PatientCase(case_id="t", patient_age=200)

    def test_invalid_sex(self):
        with pytest.raises(ValidationError):
            PatientCase.model_validate({"case_id": "t", "patient_sex": "not_a_sex"})


class TestRankedDiagnosis:
    def test_creation(self):
        rd = RankedDiagnosis(rank=1, disease_name="Test")
        assert rd.rank == 1
        assert rd.disease_name == "Test"
        assert rd.probability == 0.0

    def test_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            RankedDiagnosis(rank=0, disease_name="Test")


class TestDifferentialDiagnosis:
    def test_empty(self):
        dd = DifferentialDiagnosis()
        assert dd.ranked_list == []
        assert dd.top_disease() is None

    def test_top_n(self):
        dd = DifferentialDiagnosis(
            ranked_list=[
                RankedDiagnosis(rank=1, disease_name="A"),
                RankedDiagnosis(rank=2, disease_name="B"),
                RankedDiagnosis(rank=3, disease_name="C"),
            ],
        )
        top = dd.top_disease()
        assert top is not None
        assert top.disease_name == "A"
        assert len(dd.top_n(2)) == 2

    def test_to_dict(self):
        dd = DifferentialDiagnosis(ranked_list=[RankedDiagnosis(rank=1, disease_name="X")])
        d = dd.to_dict()
        ranked = d["ranked_list"]
        assert isinstance(ranked, list)
        assert ranked[0]["disease_name"] == "X"


class TestDiagnosisResult:
    def test_creation(self):
        result = DiagnosisResult(
            case_id="TEST-001",
            differential=DifferentialDiagnosis(),
            state=RareDiseaseState(),
        )
        assert result.case_id == "TEST-001"
        assert result.iterations == 0
        assert result.converged is False
        assert result.evaluation is None

    def test_to_dict(self):
        result = DiagnosisResult(
            case_id="T",
            differential=DifferentialDiagnosis(),
            state=RareDiseaseState(),
        )
        d = result.to_dict()
        assert d["case_id"] == "T"


class TestEvaluationMetrics:
    def test_defaults(self):
        m = EvaluationMetrics()
        assert m.recall_at_1 == 0.0
        assert m.total_cases == 0
        assert m.correct_cases == 0

    def test_values(self):
        m = EvaluationMetrics(recall_at_1=0.42, recall_at_5=0.71, mrr=0.55, total_cases=100, correct_cases=42)
        assert m.recall_at_1 == 0.42
        assert m.correct_cases == 42


if __name__ == "__main__":
    import unittest

    unittest.main()
