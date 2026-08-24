"""Promotion eligibility and token validation for training-ready packages.

This module handles the promotion workflow for dataset packages that have
passed training readiness gates. It validates promotion tokens and manages
the promotion state machine.

Promotion Workflow
------------------
1. Package created with promotion_token.json (if can_promote=True)
2. Promotion service validates token and manifest
3. Package exported to training pipeline
4. Promotion status updated to PROMOTED
5. Observability events emitted (PIX-507)

Token Validation
---------------
- Token must exist and be valid JSON
- Token package_id must match manifest package_id
- Token validation_hash must match data.jsonl hash
- Token promoted_at must be recent (within 24h)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .packaging import DatasetManifest


class PromotionStatus(StrEnum):
    """Promotion lifecycle status."""

    ELIGIBLE = "eligible"  # Has valid token, can be promoted
    PROMOTING = "promoting"  # Currently being promoted
    PROMOTED = "promoted"  # Successfully promoted to training
    FAILED = "failed"  # Promotion failed
    EXPIRED = "expired"  # Token expired


class PromotionError(Exception):
    """Promotion validation or execution error."""


@dataclass
class PromotionToken:
    """Promotion eligibility token from package creation."""

    package_id: str
    promoted_at: str
    status: str
    validation_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "package_id": self.package_id,
            "promoted_at": self.promoted_at,
            "status": self.status,
            "validation_hash": self.validation_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionToken:
        return cls(
            package_id=data["package_id"],
            promoted_at=data["promoted_at"],
            status=data["status"],
            validation_hash=data["validation_hash"],
        )


@dataclass
class PromotionResult:
    """Result of promotion validation."""

    status: PromotionStatus
    package_id: str
    stage_id: str
    token: PromotionToken | None = None
    manifest: DatasetManifest | None = None
    error_message: str = ""
    validated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "package_id": self.package_id,
            "stage_id": self.stage_id,
            "token": self.token.to_dict() if self.token else None,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "error_message": self.error_message,
            "validated_at": self.validated_at,
        }


class PromotionService:
    """Validates and executes package promotion.

    Usage:
        service = PromotionService()
        result = service.validate_promotion(
            package_path=Path("datasets/stage1_foundation/"),
        )
        if result.status == PromotionStatus.ELIGIBLE:
            export_to_training(result)
    """

    def __init__(self, token_expiry_hours: int = 24) -> None:
        """Initialize promotion service.

        Args:
            token_expiry_hours: Hours after which promotion tokens expire
        """
        self.token_expiry_hours = token_expiry_hours

    def validate_promotion(
        self,
        package_path: Path,
    ) -> PromotionResult:
        """Validate a package for promotion.

        Args:
            package_path: Path to package directory

        Returns:
            PromotionResult with validation status
        """
        # Load promotion token
        token_path = package_path / "promotion_token.json"
        if not token_path.exists():
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id="",
                stage_id="",
                error_message="No promotion token found",
            )

        # Reject already-promoted packages
        promoted_path = package_path / "promoted.json"
        if promoted_path.exists():
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id="",
                stage_id="",
                error_message="Package already promoted — re-export is blocked",
            )

        try:
            with open(token_path) as f:
                token_data = json.load(f)
            token = PromotionToken.from_dict(token_data)
        except (json.JSONDecodeError, KeyError) as e:
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id="",
                stage_id="",
                error_message=f"Invalid token: {e}",
            )

        # Load manifest
        manifest_path = package_path / "manifest.json"
        if not manifest_path.exists():
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id="",
                token=token,
                error_message="No manifest found",
            )

        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            manifest = DatasetManifest(
                name=manifest_data["name"],
                stage=manifest_data["stage"],
                created_at=manifest_data["created_at"],
                record_count=manifest_data["record_count"],
                stage_thresholds=manifest_data["stage_thresholds"],
                actual_metrics=manifest_data["actual_metrics"],
                validation_gates=manifest_data["validation_gates"],
                promotion_status=manifest_data["promotion_status"],
                package_id=manifest_data.get("package_id", ""),
                data_hash=manifest_data.get("data_hash", ""),
                readiness_result=manifest_data.get("readiness_result"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id="",
                token=token,
                error_message=f"Invalid manifest: {e}",
            )

        # Validate token package_id matches manifest
        if token.package_id != manifest.package_id:
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id=manifest.stage,
                token=token,
                manifest=manifest,
                error_message=f"Package ID mismatch: token={token.package_id}, manifest={manifest.package_id}",
            )

        # Validate data hash
        data_path = package_path / "data.jsonl"
        if not data_path.exists():
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id=manifest.stage,
                token=token,
                manifest=manifest,
                error_message="data.jsonl not found in package",
            )
        actual_hash = self._compute_file_hash(data_path)
        if actual_hash != token.validation_hash:
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id=manifest.stage,
                token=token,
                manifest=manifest,
                error_message="Data hash mismatch - package may be corrupted",
            )

        # Check token expiry
        try:
            promoted_at = datetime.fromisoformat(token.promoted_at)
            now = datetime.now(UTC)
            if now - promoted_at > timedelta(hours=self.token_expiry_hours):
                return PromotionResult(
                    status=PromotionStatus.EXPIRED,
                    package_id=token.package_id,
                    stage_id=manifest.stage,
                    token=token,
                    manifest=manifest,
                    error_message=f"Token expired ({self.token_expiry_hours}h limit)",
                )
        except ValueError as e:
            return PromotionResult(
                status=PromotionStatus.FAILED,
                package_id=token.package_id,
                stage_id=manifest.stage,
                token=token,
                manifest=manifest,
                error_message=f"Invalid token timestamp: {e}",
            )

        # All validations passed
        return PromotionResult(
            status=PromotionStatus.ELIGIBLE,
            package_id=token.package_id,
            stage_id=manifest.stage,
            token=token,
            manifest=manifest,
        )

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def mark_promoted(
        self,
        package_path: Path,
        training_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark a package as promoted.

        Args:
            package_path: Path to package directory
            training_run_id: Optional training run identifier

        Returns:
            Promotion metadata dictionary
        """
        token_path = package_path / "promotion_token.json"
        promoted_path = package_path / "promoted.json"

        # Load existing token
        with open(token_path) as f:
            token_data = json.load(f)

        # Write promoted status
        promoted_data = {
            **token_data,
            "training_run_id": training_run_id,
            "marked_promoted_at": datetime.now(UTC).isoformat(),
        }

        with open(promoted_path, "w") as f:
            json.dump(promoted_data, f, indent=2)

        return promoted_data


def check_promotion_eligibility(package_path: Path) -> PromotionResult:
    """Convenience function to check promotion eligibility.

    Usage:
        result = check_promotion_eligibility(Path("datasets/stage1_foundation/"))
        if result.status == PromotionStatus.ELIGIBLE:
            logger.info(f"Package {result.package_id} ready for promotion")
    """
    import logging

    logger = logging.getLogger(__name__)
    service = PromotionService()
    return service.validate_promotion(package_path)


__all__ = [
    "PromotionError",
    "PromotionResult",
    "PromotionService",
    "PromotionStatus",
    "PromotionToken",
    "check_promotion_eligibility",
]
