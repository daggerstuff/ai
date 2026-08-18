"""Tests for DiagnosisArenaEvaluator with Wilson CI (Phase 3 enterprise upgrade)."""

from __future__ import annotations

import pytest

from ai.pkg_mera.platform.deep_rare.evaluator import DiagnosisArenaEvaluator
from ai.pkg_mera.platform.deep_rare.knowledge_base import RareDiseaseKnowledgeBase
from ai.pkg_mera.platform.deep_rare.orchestrator import ControllerOrchestrator
from ai.pkg_mera.platform.deep_rare.schema import (
    DifferentialDiagnosis,
    DiagnosisResult,
    EvaluationMetrics,
    PatientCase,
    RankedDiagnosis,
    RareDiseaseState,
    SymptomProfile,
)


@pytest.fixture
def evaluator() -> DiagnosisArenaEvaluator:
    return DiagnosisArenaEvaluator()


@pytest.fixture
def kb() -> RareDiseaseKnowledgeBase:
    return RareDiseaseKnowledgeBase()


@pytest.fixture
def orchestrator(kb: RareDiseaseKnowledgeBase) -> ControllerOrchestrator:
    return ControllerOrchestrator(kb, max_iterations=5, convergence_window=3)


def _make_result(case_id: str, gt: str, ranked_names: list[str], converged: bool = True) -> DiagnosisResult:
    ranked = [
        RankedDiagnosis(
            rank=i + 1, disease_name=n, probability=0.9 - i * 0.1, evidence_summary="", evidence_count=1, confidence=0.5
        )
        for i, n in enumerate(ranked_names)
    ]
    return DiagnosisResult(
        case_id=case_id,
        differential=DifferentialDiagnosis(
            ranked_list=ranked,
            eliminated=[],
            total_hypotheses_considered=len(ranked),
            iterations_used=3,
            convergence_achieved=converged,
            reasoning_trace="test",
        ),
        state=RareDiseaseState(max_iterations=5, convergence_window=3),
        iterations=3,
        time_seconds=5.0,
        converged=converged,
        agent_outputs={},
        recommended_next_steps=[],
        clinical_confidence=0.7,
    )


def _make_case(case_id: str, gt: str) -> PatientCase:
    return PatientCase(
        case_id=case_id,
        patient_age=30,
        patient_sex="male",
        presenting_symptoms=[
            SymptomProfile(
                name="muscle weakness",
                category="musculoskeletal",
                onset="chronic",
                progression="worsening",
                severity="moderate",
            ),
        ],
        medical_history=[],
        family_history=[],
        current_medications=[],
        available_tests=[],
        clinical_notes="",
        ground_truth_diagnosis=gt,
    )


class TestWilsonCI:
    def test_wilson_ci_with_matches(self, evaluator: DiagnosisArenaEvaluator):
        results = [_make_result("C1", "Disease A", ["Disease A", "Disease B"])]
        cases = [_make_case("C1", "Disease A")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.recall_at_1 == 1.0
        assert 0.0 < metrics.recall_at_1_ci_lower < 1.0
        assert 0.0 < metrics.recall_at_1_ci_upper <= 1.0

    def test_wilson_ci_no_matches(self, evaluator: DiagnosisArenaEvaluator):
        results = [_make_result("C1", "Disease A", ["Disease B", "Disease C"])]
        cases = [_make_case("C1", "Disease A")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.recall_at_1 == 0.0
        assert metrics.recall_at_1_ci_lower == 0.0
        assert metrics.recall_at_1_ci_upper > 0.0

    def test_wilson_ci_bounds_valid(self, evaluator: DiagnosisArenaEvaluator):
        for n in [2, 5, 10, 20]:
            results = [_make_result(f"C{i}", f"D{i}", [f"D{i}", "Other"]) for i in range(n)]
            cases = [_make_case(f"C{i}", f"D{i}") for i in range(n)]
            metrics = evaluator.evaluate(results, cases)
            assert 0.0 <= metrics.recall_at_1_ci_lower <= 1.0
            assert 0.0 <= metrics.recall_at_1_ci_upper <= 1.0


class TestRecallAtK:
    def test_recall_at_1(self, evaluator: DiagnosisArenaEvaluator):
        results = [
            _make_result("C1", "Disease A", ["Disease A"]),
            _make_result("C2", "Disease B", ["Disease C"]),
        ]
        cases = [_make_case("C1", "Disease A"), _make_case("C2", "Disease B")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.recall_at_1 == 0.5

    def test_recall_at_5(self, evaluator: DiagnosisArenaEvaluator):
        results = [
            _make_result("C1", "Disease A", ["X", "Y", "Z", "W", "Disease A"]),
            _make_result("C2", "Disease B", ["Disease B", "X", "Y", "Z", "W"]),
        ]
        cases = [_make_case("C1", "Disease A"), _make_case("C2", "Disease B")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.recall_at_5 == 1.0

    def test_recall_at_10(self, evaluator: DiagnosisArenaEvaluator):
        names = [f"D{i}" for i in range(9)] + ["Disease A"]
        results = [_make_result("C1", "Disease A", names)]
        cases = [_make_case("C1", "Disease A")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.recall_at_10 == 1.0
        assert metrics.recall_at_1 == 0.0

    def test_mrr(self, evaluator: DiagnosisArenaEvaluator):
        results = [
            _make_result("C1", "Disease A", ["Disease A"]),
            _make_result("C2", "Disease B", ["X", "Disease B"]),
        ]
        cases = [_make_case("C1", "Disease A"), _make_case("C2", "Disease B")]
        metrics = evaluator.evaluate(results, cases)
        assert metrics.mrr == pytest.approx(0.75, abs=0.01)


class TestPerCaseErrorAnalysis:
    def test_error_cases_populated(self, evaluator: DiagnosisArenaEvaluator):
        results = [
            _make_result("C1", "Disease A", ["Disease A"]),
            _make_result("C2", "Disease B", ["Disease C", "Disease D"]),
        ]
        cases = [_make_case("C1", "Disease A"), _make_case("C2", "Disease B")]
        metrics = evaluator.evaluate(results, cases)
        assert len(metrics.error_cases) == 1
        assert metrics.error_cases[0]["case_id"] == "C2"

    def test_error_cases_empty_when_all_correct(self, evaluator: DiagnosisArenaEvaluator):
        results = [_make_result("C1", "Disease A", ["Disease A"])]
        cases = [_make_case("C1", "Disease A")]
        metrics = evaluator.evaluate(results, cases)
        assert len(metrics.error_cases) == 0


class TestSafetyTracking:
    def test_safety_violation_count(self, evaluator: DiagnosisArenaEvaluator):
        result = _make_result("C1", "Disease A", ["Disease A"])
        result.safety_violations = [{"type": "test", "message": "violation"}]
        metrics = evaluator.evaluate([result], [_make_case("C1", "Disease A")])
        assert metrics.safety_violation_count == 1

    def test_avg_clinical_confidence(self, evaluator: DiagnosisArenaEvaluator):
        result = _make_result("C1", "Disease A", ["Disease A"])
        result.clinical_confidence = 0.8
        metrics = evaluator.evaluate([result], [_make_case("C1", "Disease A")])
        assert metrics.avg_clinical_confidence == pytest.approx(0.8, abs=0.01)


class TestCompareBaseline:
    def test_compare_returns_dict(self, evaluator: DiagnosisArenaEvaluator):
        multi = evaluator.evaluate([_make_result("C1", "Disease A", ["Disease A"])], [_make_case("C1", "Disease A")])
        single = evaluator.evaluate([_make_result("C1", "Disease A", ["Disease B"])], [_make_case("C1", "Disease A")])
        diff = evaluator.compare_baseline(multi, single)
        assert isinstance(diff, dict)
        assert "recall_at_1_delta" in diff

    def test_compare_significance(self, evaluator: DiagnosisArenaEvaluator):
        multi = evaluator.evaluate(
            [_make_result(f"C{i}", f"D{i}", [f"D{i}"]) for i in range(20)],
            [_make_case(f"C{i}", f"D{i}") for i in range(20)],
        )
        single = evaluator.evaluate(
            [_make_result(f"C{i}", f"D{i}", ["Other"]) for i in range(20)],
            [_make_case(f"C{i}", f"D{i}") for i in range(20)],
        )
        diff = evaluator.compare_baseline(multi, single)
        assert "recall_at_1_significance" in diff


class TestEndToEndEvaluation:
    def test_evaluate_with_orchestrator(self, orchestrator: ControllerOrchestrator, evaluator: DiagnosisArenaEvaluator):
        case = PatientCase(
            case_id="EVAL-POMPE",
            patient_age=3,
            patient_sex="male",
            presenting_symptoms=[
                SymptomProfile(
                    name="muscle weakness",
                    category="musculoskeletal",
                    onset="infancy",
                    progression="worsening",
                    severity="severe",
                ),
                SymptomProfile(
                    name="hypotonia",
                    category="neurological",
                    onset="infancy",
                    progression="worsening",
                    severity="severe",
                ),
                SymptomProfile(
                    name="cardiomegaly",
                    category="cardiovascular",
                    onset="infancy",
                    progression="worsening",
                    severity="severe",
                ),
                SymptomProfile(
                    name="respiratory distress",
                    category="respiratory",
                    onset="infancy",
                    progression="worsening",
                    severity="moderate",
                ),
            ],
            medical_history=[],
            family_history=[],
            current_medications=[],
            available_tests=[],
            clinical_notes="Infantile onset",
            ground_truth_diagnosis="Pompe Disease",
        )
        result = orchestrator.diagnose(case)
        metrics = evaluator.evaluate([result], [case])
        assert metrics.total_cases == 1
        assert metrics.avg_time_seconds > 0

    def test_evaluate_empty(self, evaluator: DiagnosisArenaEvaluator):
        metrics = evaluator.evaluate([], [])
        assert metrics.total_cases == 0
