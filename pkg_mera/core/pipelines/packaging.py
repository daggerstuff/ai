"""Training-ready dataset packaging for the modern dataset pipeline.

This module defines how curated stage-sliced datasets are packaged into
promotion-ready artifacts. It integrates with the training readiness gates
(PIX-506) to produce validated, self-describing dataset packages.

Package Model
-------------
Each package is a self-contained bundle containing:
- manifest.json: Package metadata and validation status
- data.jsonl: Serialized records in canonical format
- metrics.json: Quality metrics snapshot
- readiness_report.json: PIX-506 validation output
- promotion_token.json: Promotion eligibility proof (if can_promote=True)

Downstream Integration
---------------------
- PIX-507 (observability): Consumes package creation events
- Promotion system: Polls for packages with promotion_token.json
- Training consumers: Read manifest.json for readiness status
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .training_readiness_gates import (
    ReadinessResult,
    TrainingReadinessGates,
)


class PackageStatus(StrEnum):
    """Package lifecycle status."""

    CREATED = "created"  # Package bundle created
    VALIDATED = "validated"  # Readiness gates applied
    READY = "ready"  # Can be promoted (all gates passed)
    BLOCKED = "blocked"  # Failed validation gates
    PROMOTED = "promoted"  # Successfully promoted to training


@dataclass
class DatasetManifest:
    """Manifest schema for training-ready dataset packages.

    This is the canonical metadata format that makes readiness
    operationally visible rather than implied.
    """

    name: str
    stage: str
    created_at: str
    record_count: int
    stage_thresholds: dict[str, float]
    actual_metrics: dict[str, float]
    validation_gates: dict[str, str]  # gate_name -> "PASS" | "FAIL"
    promotion_status: str  # "READY" | "NOT_READY" | "BLOCKED"

    # Optional fields
    package_id: str = ""
    data_hash: str = ""
    readiness_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "stage": self.stage,
            "created_at": self.created_at,
            "record_count": self.record_count,
            "stage_thresholds": self.stage_thresholds,
            "actual_metrics": self.actual_metrics,
            "validation_gates": self.validation_gates,
            "promotion_status": self.promotion_status,
            "package_id": self.package_id,
            "data_hash": self.data_hash,
            "readiness_result": self.readiness_result,
        }

    @classmethod
    def from_readiness_result(
        cls,
        readiness_result: ReadinessResult,
        stage_thresholds: dict[str, float],
    ) -> DatasetManifest:
        """Create manifest from PIX-506 validation result.

        Args:
            readiness_result: Output from TrainingReadinessGates.validate_package()
            stage_thresholds: Stage-specific thresholds from PIX-249

        Returns:
            DatasetManifest ready for serialization
        """
        # Extract actual metrics from readiness result
        actual_metrics = readiness_result.metrics.copy()

        # Build validation gate results
        validation_gates = {}
        for gate_name, gate_result in readiness_result.gate_results.items():
            validation_gates[gate_name] = "PASS" if gate_result.passed else "FAIL"

        # Determine promotion status
        if readiness_result.can_promote:
            promotion_status = "READY"
        elif readiness_result.failed_gates:
            promotion_status = "BLOCKED"
        else:
            promotion_status = "NOT_READY"

        return cls(
            name=f"{readiness_result.stage_id}-slice-v1",
            stage=readiness_result.stage_id,
            created_at=readiness_result.validated_at,
            record_count=readiness_result.record_count,
            stage_thresholds=stage_thresholds,
            actual_metrics=actual_metrics,
            validation_gates=validation_gates,
            promotion_status=promotion_status,
            package_id=readiness_result.package_id,
            readiness_result=readiness_result.to_dict(),
        )


@dataclass
class PackageBundle:
    """Complete package bundle ready for export."""

    manifest: DatasetManifest
    data_path: Path
    metrics_path: Path
    readiness_path: Path
    promotion_token_path: Path | None  # None if not promotable

    @property
    def is_promotable(self) -> bool:
        """Whether this package can be promoted."""
        return self.promotion_token_path is not None


class DatasetPackager:
    """Creates training-ready dataset packages.

    Usage:
        packager = DatasetPackager(output_dir="datasets/")
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=[...],
            gate_audit={...},
            metrics={...},
        )
        if bundle.is_promotable:
            logger.info(f"Package ready for promotion: {bundle.manifest.name}")
    """

    import logging

    logger = logging.getLogger(__name__)

    def __init__(self, output_dir: str = "datasets/") -> None:
        """Initialize packager.

        Args:
            output_dir: Base directory for package output
        """
        self.output_dir = Path(output_dir)
        self.readiness_gates = TrainingReadinessGates()

    def create_package(
        self,
        stage_id: str,
        records: list[dict[str, Any]],
        gate_audit: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        package_id: str | None = None,
    ) -> PackageBundle:
        """Create a training-ready dataset package.

        Args:
            stage_id: Stage slice identifier (e.g., "stage1_foundation")
            records: Records to include in package
            gate_audit: Optional audit trail from privacy gates
            metrics: Optional pre-computed quality metrics
            package_id: Optional package identifier

        Returns:
            PackageBundle with all package artifacts
        """
        # Generate package ID if not provided
        if package_id is None:
            package_id = self._generate_package_id(stage_id, records)

        # Run readiness validation (PIX-506)
        readiness_result = self.readiness_gates.validate_package(
            package_id=package_id,
            stage_id=stage_id,
            records=records,
            gate_audit=gate_audit,
            metrics=metrics,
        )

        # Get stage thresholds (from PIX-249)
        from .training_readiness_gates import STAGE_QUALITY_THRESHOLDS

        stage_thresholds = STAGE_QUALITY_THRESHOLDS.get(
            stage_id,
            STAGE_QUALITY_THRESHOLDS["supplementary"],
        )

        # Create manifest
        manifest = DatasetManifest.from_readiness_result(
            readiness_result=readiness_result,
            stage_thresholds=stage_thresholds,
        )

        # Create package directory
        stage_dir = self.output_dir / stage_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Write data.jsonl
        data_path = stage_dir / "data.jsonl"
        with open(data_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        # Compute data hash
        data_hash = self._compute_file_hash(data_path)
        manifest.data_hash = data_hash

        # Write metrics.json
        metrics_path = stage_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(readiness_result.metrics, f, indent=2)

        # Write readiness_report.json
        readiness_path = stage_dir / "readiness_report.json"
        with open(readiness_path, "w") as f:
            json.dump(readiness_result.to_dict(), f, indent=2)

        # Write manifest.json
        manifest_path = stage_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Write promotion_token.json if promotable
        promotion_token_path = None
        if readiness_result.can_promote:
            promotion_token_path = stage_dir / "promotion_token.json"
            token = {
                "package_id": package_id,
                "promoted_at": datetime.now(UTC).isoformat(),
                "status": "READY_FOR_PROMOTION",
                "validation_hash": data_hash,
            }
            with open(promotion_token_path, "w") as f:
                json.dump(token, f, indent=2)

        return PackageBundle(
            manifest=manifest,
            data_path=data_path,
            metrics_path=metrics_path,
            readiness_path=readiness_path,
            promotion_token_path=promotion_token_path,
        )

    def _generate_package_id(self, stage_id: str, records: list[dict]) -> str:
        """Generate unique package ID."""
        timestamp = datetime.now(UTC).isoformat()
        content = f"{stage_id}-{timestamp}-{len(records)}"
        return f"pkg-{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def create_training_package(
    stage_id: str,
    records: list[dict[str, Any]],
    output_dir: str = "datasets/",
    gate_audit: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> PackageBundle:
    """Convenience function to create a training package.

    Usage:
        bundle = create_training_package(
            stage_id="stage1_foundation",
            records=my_records,
            output_dir="datasets/",
        )
        if bundle.is_promotable:
            logger.info(f"Ready: {bundle.manifest.name}")
    """
    packager = DatasetPackager(output_dir=output_dir)
    return packager.create_package(
        stage_id=stage_id,
        records=records,
        gate_audit=gate_audit,
        metrics=metrics,
    )
