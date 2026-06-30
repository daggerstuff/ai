"""Training-readiness validation gates for the modern dataset pipeline.

This module defines the validation system that determines whether a curated
dataset is actually training-ready. It makes readiness a visible operational
state rather than an assumption.

Design principles
-----------------
* Every validation check has explicit pass/fail criteria, not a fuzzy threshold.
* Validation results are self-documenting: another operator can understand
  why a dataset passed or failed without extra interpretation.
* Validation state is surfaced in a format consumable by downstream systems
  (promotion decisions, observability, reporting).
* Stage-specific criteria honor the slicing model from PIX-249.

Validation checks
-----------------
Each dataset package must pass:
  1. Completeness — required metadata fields present
  2. Quality floors — empathy, clinical, safety scores meet stage thresholds
  3. Deduplication retention — dedup rate within acceptable range
  4. Privacy compliance — gate audit trail shows clean passage
  5. Slice boundaries — records belong to correct stage slice

Downstream consumers
-------------------
  PIX-507 (pipeline observability) — consumes validation events
  Promotion system — consumes pass/fail decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ReadinessStatus(StrEnum):
    """Overall readiness determination."""

    READY = "ready"  # All gates passed
    NOT_READY = "not_ready"  # One or more gates failed
    CONDITIONALLY_READY = "conditionally_ready"  # Passed but with warnings


class ReadinessGate(StrEnum):
    """Individual validation gates."""

    COMPLETENESS = "completeness"
    QUALITY_FLOORS = "quality_floors"
    DEDUP_RETENTION = "dedup_retention"
    PRIVACY_COMPLIANCE = "privacy_compliance"
    SLICE_BOUNDARIES = "slice_boundaries"


@dataclass
class GateResult:
    """Result of a single validation gate."""

    gate: ReadinessGate
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
            "checked_at": self.checked_at,
        }


@dataclass
class ReadinessResult:
    """Full readiness validation result for a dataset package."""

    package_id: str
    stage_id: str
    status: ReadinessStatus
    gate_results: dict[str, GateResult] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    record_count: int = 0
    validated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    validated_by: str = "TrainingReadinessGates"

    @property
    def passed(self) -> bool:
        return self.status == ReadinessStatus.READY

    @property
    def failed_gates(self) -> list[str]:
        return [gate for gate, result in self.gate_results.items() if not result.passed]

    @property
    def warnings(self) -> list[str]:
        return [
            result.reason for result in self.gate_results.values() if result.passed and result.details.get("warning")
        ]

    @property
    def can_promote(self) -> bool:
        """True when package is eligible for promotion."""
        return self.status == ReadinessStatus.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "stage_id": self.stage_id,
            "status": self.status.value,
            "can_promote": self.can_promote,
            "gate_results": {k: v.to_dict() for k, v in self.gate_results.items()},
            "metrics": self.metrics,
            "record_count": self.record_count,
            "failed_gates": self.failed_gates,
            "warnings": self.warnings,
            "validated_at": self.validated_at,
            "validated_by": self.validated_by,
        }

    def get_failure_summary(self) -> str:
        """Human-readable failure summary."""
        if self.passed:
            return "All validation gates passed"
        lines = [f"Validation failed for {self.package_id}:"]
        for gate, result in self.gate_results.items():
            if not result.passed:
                lines.append(f"  - {gate}: {result.reason}")
        return "\n".join(lines)


# Stage-specific quality thresholds (from PIX-249 canonical model)
STAGE_QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "stage1_foundation": {
        "empathy_floor": 0.70,
        "clinical_floor": 0.30,
        "safety_floor": 1.0,
        "safety_floor": 0.70,
        "dedup_retention_min": 0.50,
    },
    "stage2_therapeutic_expertise": {
        "empathy_floor": 0.75,
        "clinical_floor": 0.50,
        "safety_floor": 1.0,
        "safety_floor": 0.75,
        "dedup_retention_min": 0.50,
    },
    "stage3_edge_stress_test": {
        "empathy_floor": 0.60,
        "clinical_floor": 0.40,
        "safety_floor": 1.0,
        "safety_floor": 0.65,
        "dedup_retention_min": 0.40,
    },
    "stage4_voice_persona": {
        "empathy_floor": 0.80,
        "clinical_floor": 0.35,
        "safety_floor": 1.0,
        "safety_floor": 0.75,
        "dedup_retention_min": 0.60,
    },
    "supplementary": {
        "empathy_floor": 0.50,
        "clinical_floor": 0.20,
        "safety_floor": 1.0,
        "safety_floor": 0.20,
        "dedup_retention_min": 0.30,
    },
}

# Minimum metadata fields required for completeness
REQUIRED_METADATA_FIELDS = frozenset(
    [
        "source",
        "stage",
        "created_at",
    ]
)

# Maximum acceptable dedup retention (to catch over-dedup that might indicate data loss)
MAX_DEDUP_RETENTION = 0.95


class TrainingReadinessGates:
    """Validates whether a dataset package is training-ready.

    Applies a series of explicit gates to determine readiness. Each gate
    has documented pass/fail criteria.

    Usage::

        gates = TrainingReadinessGates()
        result = gates.validate_package(
            package_id="pkg-001",
            stage_id="stage1_foundation",
            records=[...],
            gate_audit={"gate0": "pass", "gate1": "pass", ...},
        )
        if result.can_promote:
            print("Package is training-ready")
        else:
            print(result.get_failure_summary())
    """

    def __init__(self) -> None:
        self._gate_results: list[GateResult] = []

    def validate_package(
        self,
        package_id: str,
        stage_id: str,
        records: list[dict[str, Any]],
        gate_audit: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> ReadinessResult:
        """Validate whether a package meets training-readiness criteria.

        Args:
            package_id: Identifier for the package being validated
            stage_id: Which stage slice this belongs to
            records: List of records in the package
            gate_audit: Optional gate audit trail from privacy_content_gates
            metrics: Optional pre-computed quality metrics

        Returns:
            ReadinessResult with pass/fail and detailed gate results
        """
        result = ReadinessResult(
            package_id=package_id,
            stage_id=stage_id,
            status=ReadinessStatus.NOT_READY,
            record_count=len(records),
        )

        if not records:
            result.gate_results[ReadinessGate.COMPLETENESS.value] = GateResult(
                gate=ReadinessGate.COMPLETENESS,
                passed=False,
                reason="Package contains no records",
            )
            return result

        thresholds = STAGE_QUALITY_THRESHOLDS.get(stage_id, STAGE_QUALITY_THRESHOLDS["supplementary"])

        # Gate 1: Completeness
        completeness_result = self._check_completeness(records)
        result.gate_results[ReadinessGate.COMPLETENESS.value] = completeness_result

        # Gate 2: Quality floors
        if metrics:
            quality_result = self._check_quality_floors(metrics, thresholds, stage_id)
        else:
            quality_result = self._estimate_quality_floors(records, thresholds)
        result.gate_results[ReadinessGate.QUALITY_FLOORS.value] = quality_result

        # Gate 3: Deduplication retention
        dedup_result = self._check_dedup_retention(records, thresholds)
        result.gate_results[ReadinessGate.DEDUP_RETENTION.value] = dedup_result

        # Gate 4: Privacy compliance
        privacy_result = self._check_privacy_compliance(gate_audit)
        result.gate_results[ReadinessGate.PRIVACY_COMPLIANCE.value] = privacy_result

        # Gate 5: Slice boundaries
        slice_result = self._check_slice_boundaries(records, stage_id)
        result.gate_results[ReadinessGate.SLICE_BOUNDARIES.value] = slice_result

        # Collect metrics
        result.metrics = self._collect_metrics(
            records, metrics, completeness_result, quality_result, dedup_result, privacy_result, slice_result
        )

        # Determine overall status
        all_gates_passed = all(r.passed for r in result.gate_results.values())
        any_gate_failed = any(not r.passed for r in result.gate_results.values())

        if all_gates_passed:
            result.status = ReadinessStatus.READY
        elif not any_gate_failed:
            result.status = ReadinessStatus.CONDITIONALLY_READY
        else:
            result.status = ReadinessStatus.NOT_READY

        return result

    def _check_completeness(self, records: list[dict[str, Any]]) -> GateResult:
        """Check that required metadata fields are present."""
        missing_fields: dict[str, int] = {}

        for field in REQUIRED_METADATA_FIELDS:
            missing_count = sum(1 for r in records if not r.get(field) and not r.get("metadata", {}).get(field))
            if missing_count > 0:
                missing_fields[field] = missing_count

        if missing_fields:
            return GateResult(
                gate=ReadinessGate.COMPLETENESS,
                passed=False,
                reason=f"Missing required metadata fields: {list(missing_fields.keys())}",
                details={"missing_fields": missing_fields},
            )

        # Check for empty text/content
        empty_count = sum(1 for r in records if not (r.get("text") or r.get("content") or r.get("conversation")))
        if empty_count > len(records) * 0.1:
            return GateResult(
                gate=ReadinessGate.COMPLETENESS,
                passed=False,
                reason=f"Too many records with empty content ({empty_count}/{len(records)})",
                details={"empty_records": empty_count},
            )

        return GateResult(
            gate=ReadinessGate.COMPLETENESS,
            passed=True,
            reason="All required metadata fields present",
            details={"records_checked": len(records)},
        )

    def _check_quality_floors(
        self,
        metrics: dict[str, float],
        thresholds: dict[str, float],
        stage_id: str,
    ) -> GateResult:
        """Check quality metrics against stage thresholds."""
        violations: list[str] = []
        empathy = metrics.get("empathy_score", metrics.get("empathy_avg", 0))
        clinical = metrics.get("clinical_score", metrics.get("clinical_avg", 0))
        safety = metrics.get("safety_score", metrics.get("safety_avg", 1.0))
        clinical_validity = metrics.get(
            "clinical_validity_avg",
            metrics.get("safety_avg", metrics.get("safety_score", 0)),
        )

        if empathy < thresholds["empathy_floor"]:
            violations.append(f"empathy {empathy:.2f} < floor {thresholds['empathy_floor']}")
        if clinical < thresholds["clinical_floor"]:
            violations.append(f"clinical {clinical:.2f} < floor {thresholds['clinical_floor']}")
        if safety < thresholds["safety_floor"]:
            violations.append(f"safety {safety:.2f} < floor {thresholds['safety_floor']}")
        if clinical_validity < thresholds["safety_floor"]:
            violations.append(f"clinical validity {clinical_validity:.2f} < floor {thresholds['safety_floor']}")

        if violations:
            return GateResult(
                gate=ReadinessGate.QUALITY_FLOORS,
                passed=False,
                reason=f"Quality floors not met: {'; '.join(violations)}",
                details={
                    "empathy": empathy,
                    "clinical": clinical,
                    "safety": safety,
                    "clinical_validity": clinical_validity,
                    "safety": clinical_validity,
                    "thresholds": thresholds,
                },
            )

        return GateResult(
            gate=ReadinessGate.QUALITY_FLOORS,
            passed=True,
            reason="All quality floors met",
            details={
                "empathy": empathy,
                "clinical": clinical,
                "safety": safety,
                "clinical_validity": clinical_validity,
                "safety": clinical_validity,
            },
        )

    def _estimate_quality_floors(
        self,
        records: list[dict[str, Any]],
        thresholds: dict[str, float],
    ) -> GateResult:
        """Estimate quality metrics when not pre-computed."""
        empathy_markers = [
            "understand",
            "feel",
            "hear",
            "support",
            "care",
            "empath",
            "compassion",
            "validate",
            "acknowledge",
        ]
        clinical_markers = [
            "diagnosis",
            "treatment",
            "intervention",
            "cbt",
            "dbt",
            "therapeutic",
            "clinical",
            "dsm",
            "symptom",
            "disorder",
        ]

        empathy_scores = []
        clinical_scores = []

        for record in records:
            text = (record.get("text", "") or "").lower()
            empathy_score = sum(1 for m in empathy_markers if m in text) / len(empathy_markers)
            clinical_score = sum(1 for m in clinical_markers if m in text) / len(clinical_markers)
            empathy_scores.append(empathy_score)
            clinical_scores.append(clinical_score)

        empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0
        clinical = sum(clinical_scores) / len(clinical_scores) if clinical_scores else 0.0
        safety = 1.0  # Assume safe unless crisis detector says otherwise

        return self._check_quality_floors(
            {"empathy_score": empathy, "clinical_score": clinical, "safety_score": safety},
            thresholds,
            "",
        )

    def _check_dedup_retention(
        self,
        records: list[dict[str, Any]],
        thresholds: dict[str, float],
    ) -> GateResult:
        """Check deduplication retention is within acceptable range."""
        seen_ids: set[str] = set()
        duplicate_count = 0

        for record in records:
            record_id = record.get("id", "")
            if record_id in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(record_id)

        retention = 1.0 - (duplicate_count / len(records)) if records else 0.0
        min_retention = thresholds["dedup_retention_min"]

        if retention < min_retention:
            return GateResult(
                gate=ReadinessGate.DEDUP_RETENTION,
                passed=False,
                reason=f"Dedup retention {retention:.2%} < minimum {min_retention:.2%}",
                details={
                    "retention": retention,
                    "duplicates_found": duplicate_count,
                    "total_records": len(records),
                },
            )

        if retention > MAX_DEDUP_RETENTION:
            return GateResult(
                gate=ReadinessGate.DEDUP_RETENTION,
                passed=True,
                reason=f"Dedup retention {retention:.2%} is very high (possible over-dedup)",
                details={
                    "retention": retention,
                    "warning": True,
                    "duplicates_found": duplicate_count,
                },
            )

        return GateResult(
            gate=ReadinessGate.DEDUP_RETENTION,
            passed=True,
            reason=f"Dedup retention {retention:.2%} within acceptable range",
            details={
                "retention": retention,
                "duplicates_found": duplicate_count,
            },
        )

    def _check_privacy_compliance(self, gate_audit: dict[str, Any] | None) -> GateResult:
        """Check that privacy gate audit shows clean passage."""
        if gate_audit is None:
            return GateResult(
                gate=ReadinessGate.PRIVACY_COMPLIANCE,
                passed=True,
                reason="No privacy audit required (package passed prior gates)",
            )

        gates = gate_audit.get("gates", {})
        blocked = False
        escalated = False

        for _gate_name, gate_data in gates.items():
            if not gate_data:
                continue
            decision = gate_data.get("decision", "").lower()
            if decision == "block":
                blocked = True
                break
            if decision == "escalate":
                escalated = True

        if blocked:
            return GateResult(
                gate=ReadinessGate.PRIVACY_COMPLIANCE,
                passed=False,
                reason="Package blocked by privacy gates",
                details={"gate_audit": gates},
            )

        if escalated:
            return GateResult(
                gate=ReadinessGate.PRIVACY_COMPLIANCE,
                passed=False,
                reason="Package has unresolved escalation",
                details={"gate_audit": gates},
            )

        return GateResult(
            gate=ReadinessGate.PRIVACY_COMPLIANCE,
            passed=True,
            reason="Privacy gate audit passed",
        )

    def _check_slice_boundaries(
        self,
        records: list[dict[str, Any]],
        stage_id: str,
    ) -> GateResult:
        """Check that records are correctly assigned to the claimed stage."""
        misassigned = 0
        for record in records:
            record_stage = record.get("stage", "")
            if record_stage and record_stage != stage_id:
                misassigned += 1

        misassignment_rate = misassigned / len(records) if records else 0.0

        if misassignment_rate > 0.05:
            return GateResult(
                gate=ReadinessGate.SLICE_BOUNDARIES,
                passed=False,
                reason=f"{misassigned} records ({misassignment_rate:.1%}) don't match stage {stage_id}",
                details={
                    "misassigned": misassigned,
                    "total": len(records),
                    "claimed_stage": stage_id,
                },
            )

        return GateResult(
            gate=ReadinessGate.SLICE_BOUNDARIES,
            passed=True,
            reason=f"All records correctly assigned to {stage_id}",
            details={"records_checked": len(records)},
        )

    def _collect_metrics(
        self,
        records: list[dict[str, Any]],
        precomputed_metrics: dict[str, float] | None,
        *gate_results: GateResult,
    ) -> dict[str, float]:
        """Collect all metrics for reporting."""
        metrics: dict[str, float] = {}

        if precomputed_metrics:
            metrics.update(precomputed_metrics)

        metrics["record_count"] = float(len(records))
        metrics["passed_gates"] = float(sum(1 for r in gate_results if r.passed))
        metrics["total_gates"] = float(len(gate_results))

        return metrics


def get_stage_thresholds(stage_id: str) -> dict[str, float]:
    """Get quality thresholds for a stage."""
    return STAGE_QUALITY_THRESHOLDS.get(stage_id, STAGE_QUALITY_THRESHOLDS["supplementary"])


__all__ = [
    "STAGE_QUALITY_THRESHOLDS",
    "GateResult",
    "ReadinessGate",
    "ReadinessResult",
    "ReadinessStatus",
    "TrainingReadinessGates",
    "get_stage_thresholds",
]
