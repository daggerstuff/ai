"""Tests for ai/core/pipelines/training_readiness_gates.py."""

from __future__ import annotations

import pytest

from ai.tools.utilities.core.pipelines.training_readiness_gates import (
    STAGE_QUALITY_THRESHOLDS,
    GateResult,
    ReadinessGate,
    ReadinessResult,
    ReadinessStatus,
    TrainingReadinessGates,
    get_stage_thresholds,
)


class TestStageThresholds:
    def test_all_stages_have_thresholds(self):
        expected_stages = [
            "stage1_foundation",
            "stage2_therapeutic_expertise",
            "stage3_edge_stress_test",
            "stage4_voice_persona",
            "supplementary",
        ]
        for stage in expected_stages:
            assert stage in STAGE_QUALITY_THRESHOLDS
            thresholds = STAGE_QUALITY_THRESHOLDS[stage]
            assert "empathy_floor" in thresholds
            assert "clinical_floor" in thresholds
            assert "safety_floor" in thresholds
            assert "dedup_retention_min" in thresholds

    def test_get_stage_thresholds(self):
        thresholds = get_stage_thresholds("stage1_foundation")
        assert thresholds["empathy_floor"] == 0.70
        assert thresholds["clinical_floor"] == 0.30
        assert thresholds["safety_floor"] == 0.70

    def test_unknown_stage_defaults_to_supplementary(self):
        thresholds = get_stage_thresholds("unknown_stage")
        assert thresholds == STAGE_QUALITY_THRESHOLDS["supplementary"]
        assert thresholds["safety_floor"] == 0.20


class TestReadinessResult:
    def test_passed_property_true_when_ready(self):
        result = ReadinessResult(
            package_id="test",
            stage_id="stage1",
            status=ReadinessStatus.READY,
        )
        assert result.passed is True
        assert result.can_promote is True

    def test_passed_property_false_when_not_ready(self):
        result = ReadinessResult(
            package_id="test",
            stage_id="stage1",
            status=ReadinessStatus.NOT_READY,
        )
        assert result.passed is False
        assert result.can_promote is False

    def test_failed_gates_list(self):
        result = ReadinessResult(
            package_id="test",
            stage_id="stage1",
            status=ReadinessStatus.NOT_READY,
        )
        result.gate_results["completeness"] = GateResult(
            gate=ReadinessGate.COMPLETENESS,
            passed=False,
            reason="Missing fields",
        )
        result.gate_results["quality"] = GateResult(
            gate=ReadinessGate.QUALITY_FLOORS,
            passed=True,
            reason="OK",
        )
        assert "completeness" in result.failed_gates
        assert "quality" not in result.failed_gates

    def test_to_dict(self):
        result = ReadinessResult(
            package_id="pkg-001",
            stage_id="stage1_foundation",
            status=ReadinessStatus.READY,
            record_count=100,
        )
        d = result.to_dict()
        assert d["package_id"] == "pkg-001"
        assert d["status"] == "ready"
        assert d["can_promote"] is True
        assert d["record_count"] == 100

    def test_get_failure_summary(self):
        result = ReadinessResult(
            package_id="test",
            stage_id="stage1",
            status=ReadinessStatus.NOT_READY,
        )
        result.gate_results["completeness"] = GateResult(
            gate=ReadinessGate.COMPLETENESS,
            passed=False,
            reason="Missing fields",
        )
        summary = result.get_failure_summary()
        assert "Missing fields" in summary
        assert "completeness" in summary


class TestTrainingReadinessGates:
    @pytest.fixture
    def gates(self):
        return TrainingReadinessGates()

    @pytest.fixture
    def valid_records(self):
        return [
            {
                "id": f"rec-{i}",
                "text": "I understand how you feel about this.",
                "stage": "stage1_foundation",
                "source": "test",
                "metadata": {"created_at": "2026-01-01"},
            }
            for i in range(10)
        ]

    @pytest.fixture
    def quality_metrics(self):
        return {
            "empathy_score": 0.75,
            "clinical_score": 0.40,
            "safety_score": 1.0,
        }

    def test_validate_empty_package_fails(self, gates):
        result = gates.validate_package("pkg-1", "stage1_foundation", [])
        assert result.passed is False
        assert "completeness" in result.failed_gates

    def test_validate_valid_package_passes(self, gates, valid_records, quality_metrics):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            metrics=quality_metrics,
        )
        assert result.passed is True
        assert result.can_promote is True
        assert result.status == ReadinessStatus.READY

    def test_validate_with_precomputed_metrics(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            metrics={"empathy_score": 0.80, "clinical_score": 0.50, "safety_score": 1.0},
        )
        assert result.passed is True
        assert result.metrics["empathy_score"] == 0.80

    def test_validate_accepts_clinical_validity_avg_alias(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            metrics={
                "empathy_score": 0.80,
                "clinical_score": 0.50,
                "clinical_validity_avg": 0.72,
            },
        )
        assert result.passed is True
        gate = result.gate_results["quality_floors"]
        assert gate.passed is True
        assert gate.details["clinical_validity"] == 0.72

    def test_validate_low_empathy_fails(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            metrics={"empathy_score": 0.50, "clinical_score": 0.40, "safety_score": 1.0},
        )
        assert result.passed is False
        assert "quality_floors" in result.failed_gates

    def test_validate_low_clinical_fails(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage2_therapeutic_expertise",
            valid_records,
            metrics={"empathy_score": 0.80, "clinical_score": 0.30, "safety_score": 1.0},
        )
        assert result.passed is False
        assert "quality_floors" in result.failed_gates

    def test_validate_low_clinical_validity_fails(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage2_therapeutic_expertise",
            valid_records,
            metrics={
                "empathy_score": 0.80,
                "clinical_score": 0.50,
                "clinical_validity_avg": 0.70,
            },
        )
        assert result.passed is False
        assert "quality_floors" in result.failed_gates

    def test_validate_blocked_privacy_fails(self, gates, valid_records):
        gate_audit = {
            "gates": {
                "gate0": {"decision": "pass"},
                "gate1": {"decision": "block", "reason": "PII not scrubbed"},
            }
        }
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            gate_audit=gate_audit,
        )
        assert result.passed is False
        assert "privacy_compliance" in result.failed_gates

    def test_validate_misassigned_records_fail(self, gates, valid_records, quality_metrics):
        records = valid_records.copy()
        records[0]["stage"] = "stage2_therapeutic_expertise"
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            records,
            metrics=quality_metrics,
        )
        assert result.passed is False
        assert "slice_boundaries" in result.failed_gates

    def test_validate_with_gate_audit_none_passes(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
            gate_audit=None,
        )
        assert result.gate_results["privacy_compliance"].passed is True

    def test_all_five_gates_present(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-1",
            "stage1_foundation",
            valid_records,
        )
        expected_gates = {
            "completeness",
            "quality_floors",
            "dedup_retention",
            "privacy_compliance",
            "slice_boundaries",
        }
        assert set(result.gate_results.keys()) == expected_gates

    def test_estimate_quality_floors_rejects_low_quality(self, gates):
        records = [
            {
                "id": "low-quality",
                "text": "random text with no therapeutic markers",
                "stage": "stage1_foundation",
                "source": "test",
                "metadata": {"created_at": "2026-01-01"},
            }
        ]
        result = gates.validate_package("pkg-1", "stage1_foundation", records)
        assert result.passed is False
        quality_gate = result.gate_results["quality_floors"]
        assert quality_gate.passed is False

    def test_edge_stage_has_lower_thresholds(self, gates):
        thresholds = get_stage_thresholds("stage3_edge_stress_test")
        assert thresholds["empathy_floor"] == 0.60
        assert thresholds["clinical_floor"] == 0.40

    def test_voice_stage_has_highest_empathy(self, gates):
        foundation = get_stage_thresholds("stage1_foundation")
        voice = get_stage_thresholds("stage4_voice_persona")
        assert voice["empathy_floor"] > foundation["empathy_floor"]

    def test_result_serialization(self, gates, valid_records):
        result = gates.validate_package(
            "pkg-001",
            "stage1_foundation",
            valid_records,
            metrics={"empathy_score": 0.75, "clinical_score": 0.40, "safety_score": 1.0},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["package_id"] == "pkg-001"
        assert "gate_results" in d
        assert "completeness" in d["gate_results"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
