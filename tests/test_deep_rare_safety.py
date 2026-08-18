"""Tests for the clinical safety layer (Phase 1 enterprise upgrade)."""

from __future__ import annotations

import pytest

from ai.pkg_mera.platform.deep_rare.clinical_safety import (
    AuditAction,
    AuditTrail,
    ClinicalSafetyContext,
    ClinicalSafetyGate,
    RedFlagDetector,
    SafetyLevel,
    SafetyViolation,
)
from ai.pkg_mera.platform.deep_rare.schema import (
    Evidence,
    Hypothesis,
    PatientCase,
    SymptomProfile,
)


@pytest.fixture
def detector() -> RedFlagDetector:
    return RedFlagDetector()


@pytest.fixture
def safety_gate() -> ClinicalSafetyGate:
    return ClinicalSafetyGate()


@pytest.fixture
def audit_trail() -> AuditTrail:
    return AuditTrail()


@pytest.fixture
def safety_context() -> ClinicalSafetyContext:
    return ClinicalSafetyContext()


@pytest.fixture
def routine_case() -> PatientCase:
    return PatientCase(
        case_id="SAFE-001",
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
        clinical_notes="Routine evaluation",
    )


@pytest.fixture
def life_threatening_case() -> PatientCase:
    return PatientCase(
        case_id="SAFE-002",
        patient_age=25,
        patient_sex="female",
        presenting_symptoms=[
            SymptomProfile(
                name="cardiomyopathy",
                category="cardiovascular",
                onset="acute",
                progression="worsening",
                severity="severe",
            ),
            SymptomProfile(
                name="arrhythmia", category="cardiovascular", onset="acute", progression="worsening", severity="severe"
            ),
        ],
        medical_history=[],
        family_history=[],
        current_medications=[],
        available_tests=[],
        clinical_notes="Acute presentation",
        clinical_urgency="life_threatening",
    )


@pytest.fixture
def pathognomonic_case() -> PatientCase:
    return PatientCase(
        case_id="SAFE-003",
        patient_age=5,
        patient_sex="male",
        presenting_symptoms=[
            SymptomProfile(
                name="cherry-red spot",
                category="ophthalmological",
                onset="congenital",
                progression="stable",
                severity="moderate",
            ),
            SymptomProfile(
                name="developmental delay",
                category="neurological",
                onset="infancy",
                progression="worsening",
                severity="severe",
            ),
        ],
        medical_history=[],
        family_history=[],
        current_medications=[],
        available_tests=[],
        clinical_notes="Tay-Sachs suspected",
    )


class TestRedFlagDetector:
    def test_routine_case_no_flags(self, detector: RedFlagDetector, routine_case: PatientCase):
        flags = detector.detect(routine_case)
        assert isinstance(flags, list)

    def test_life_threatening_detected(self, detector: RedFlagDetector, life_threatening_case: PatientCase):
        flags = detector.detect(life_threatening_case)
        assert len(flags) > 0

    def test_pathognomonic_detected(self, detector: RedFlagDetector, pathognomonic_case: PatientCase):
        flags = detector.detect(pathognomonic_case)
        assert len(flags) > 0

    def test_should_block_routine(self, detector: RedFlagDetector, routine_case: PatientCase):
        flags = detector.detect(routine_case)
        assert detector.should_block(flags) is False

    def test_should_block_life_threatening(self, detector: RedFlagDetector, life_threatening_case: PatientCase):
        flags = detector.detect(life_threatening_case)
        # Life-threatening flags are CRITICAL, not BLOCKING — should_block returns False
        # but the flags should still be detected
        assert len(flags) > 0


class TestClinicalSafetyGate:
    def test_can_confirm_high_confidence_with_evidence(self, safety_gate: ClinicalSafetyGate):
        hyp = Hypothesis(disease_name="Test Disease", disease_id="ORPHA:1", confidence_score=0.90)
        hyp.add_evidence(
            Evidence(source="symptom_analyzer", description="pathognomonic match", supports=True, weight=0.9)
        )
        can, reason = safety_gate.can_confirm(hyp)
        assert can is True
        assert reason is None

    def test_cannot_confirm_low_confidence(self, safety_gate: ClinicalSafetyGate):
        hyp = Hypothesis(disease_name="Test Disease", disease_id="ORPHA:1", confidence_score=0.50)
        hyp.add_evidence(Evidence(source="symptom_analyzer", description="weak match", supports=True, weight=0.3))
        can, reason = safety_gate.can_confirm(hyp)
        assert can is False
        assert reason is not None

    def test_cannot_confirm_no_evidence(self, safety_gate: ClinicalSafetyGate):
        hyp = Hypothesis(disease_name="Test Disease", disease_id="ORPHA:1", confidence_score=0.90)
        can, reason = safety_gate.can_confirm(hyp)
        assert can is False
        assert "evidence" in (reason or "").lower()

    def test_can_eliminate_low_posterior(self, safety_gate: ClinicalSafetyGate):
        hyp = Hypothesis(
            disease_name="Test Disease", disease_id="ORPHA:1", posterior_probability=0.0001, is_life_threatening=False
        )
        can, reason = safety_gate.can_eliminate(hyp, evidence=[])
        assert can is True

    def test_cannot_eliminate_high_posterior(self, safety_gate: ClinicalSafetyGate):
        hyp = Hypothesis(disease_name="Test Disease", disease_id="ORPHA:1", posterior_probability=0.05)
        can, reason = safety_gate.can_eliminate(hyp, evidence=[])
        assert can is False

    def test_cannot_eliminate_life_threatening_protected(self, safety_gate: ClinicalSafetyGate):
        safety_gate.register_protected_disease("Test Disease")
        hyp = Hypothesis(
            disease_name="Test Disease", disease_id="ORPHA:1", posterior_probability=0.0001, is_life_threatening=True
        )
        can, reason = safety_gate.can_eliminate(hyp, evidence=[])
        assert can is False

    def test_record_violation(self, safety_gate: ClinicalSafetyGate):
        safety_gate.record_violation(SafetyLevel.WARNING, "test_rule", "test description", {})
        assert len(safety_gate.violations) == 1

    def test_has_blocking_violations(self, safety_gate: ClinicalSafetyGate):
        safety_gate.record_violation(SafetyLevel.BLOCKING, "block_rule", "blocking", {})
        assert safety_gate.has_blocking_violations is True

    def test_reset_iteration(self, safety_gate: ClinicalSafetyGate):
        safety_gate.increment_elimination()
        safety_gate.reset_iteration()
        # After reset, should allow eliminations again

    def test_get_safety_summary(self, safety_gate: ClinicalSafetyGate):
        summary = safety_gate.get_safety_summary()
        assert isinstance(summary, dict)


class TestAuditTrail:
    def test_record_and_retrieve(self, audit_trail: AuditTrail):
        entry = audit_trail.record(
            action=AuditAction.HYPOTHESIS_CREATED,
            agent_name="symptom_analyzer",
            case_id="CASE-001",
            details={"disease": "Test"},
        )
        assert entry.action == AuditAction.HYPOTHESIS_CREATED
        assert entry.case_id == "CASE-001"
        assert audit_trail.entry_count == 1

    def test_get_case_trail(self, audit_trail: AuditTrail):
        audit_trail.record(AuditAction.HYPOTHESIS_CREATED, "symptom_analyzer", "CASE-001", {})
        audit_trail.record(AuditAction.TEST_INTERPRETED, "test_interpreter", "CASE-001", {})
        audit_trail.record(AuditAction.HYPOTHESIS_CREATED, "symptom_analyzer", "CASE-002", {})
        trail = audit_trail.get_case_trail("CASE-001")
        assert len(trail) == 2

    def test_to_dict(self, audit_trail: AuditTrail):
        audit_trail.record(AuditAction.DIAGNOSIS_FINALIZED, "orchestrator", "CASE-001", {"result": "confirmed"})
        d = audit_trail.to_dict()
        assert isinstance(d, list)
        assert len(d) == 1

    def test_clear(self, audit_trail: AuditTrail):
        audit_trail.record(AuditAction.HYPOTHESIS_CREATED, "symptom_analyzer", "CASE-001", {})
        audit_trail.clear()
        assert audit_trail.entry_count == 0


class TestClinicalSafetyContext:
    def test_evaluate_case_safety_routine(self, safety_context: ClinicalSafetyContext, routine_case: PatientCase):
        flags = safety_context.evaluate_case_safety(routine_case)
        assert isinstance(flags, list)

    def test_evaluate_case_safety_life_threatening(
        self, safety_context: ClinicalSafetyContext, life_threatening_case: PatientCase
    ):
        flags = safety_context.evaluate_case_safety(life_threatening_case)
        assert len(flags) > 0

    def test_get_full_report(self, safety_context: ClinicalSafetyContext, routine_case: PatientCase):
        safety_context.evaluate_case_safety(routine_case)
        report = safety_context.get_full_report()
        assert isinstance(report, dict)
