"""Runtime bootstrap helpers for dataset assembly workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.orchestration.runtime_policy_service import (
    RuntimePolicyService,
)
from ai.pipelines.orchestrator.storage_config import (
    StorageBackend,
    StorageConfig,
    get_storage_config,
)
from ai.pipelines.orchestrator.storage_manager import StorageManager
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.pipeline_runtime_bootstrap")


@dataclass(frozen=True)
class PipelineRuntimeBundle:
    runtime_policy_service: RuntimePolicyService
    stage_distribution: dict[str, float]
    stage_quality_profiles: dict[str, Any]
    stage_drift_waivers: dict[str, Any]
    storage: StorageManager | None


class PipelineRuntimeBootstrap:
    """Own strict-mode overrides, runtime policy setup, and storage bootstrap."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        stage_drift_tolerance: float,
        storage_config: StorageConfig | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.stage_drift_tolerance = stage_drift_tolerance
        self.storage_config = storage_config or get_storage_config()

    def bootstrap(
        self,
        *,
        config: Any,
        warning_sink: Any,
        source_paths: list[str],
    ) -> PipelineRuntimeBundle:
        self.apply_strict_mode_overrides(config)
        runtime_policy_service = RuntimePolicyService(
            manifest_path=self.manifest_path,
            warning_sink=warning_sink,
            default_stage_distribution=config.stage_distribution,
            default_stage_drift_tolerance=self.stage_drift_tolerance,
        )
        runtime_policy_service.hydrate_tracker_config(config)
        stage_policy = runtime_policy_service.load_stage_policy()
        config.stage_distribution = stage_policy.stage_distribution
        storage = self.initialize_storage_manager(source_paths)
        return PipelineRuntimeBundle(
            runtime_policy_service=runtime_policy_service,
            stage_distribution=stage_policy.stage_distribution,
            stage_quality_profiles=stage_policy.quality_profiles,
            stage_drift_waivers=stage_policy.drift_waivers,
            storage=storage,
        )

    def apply_strict_mode_overrides(self, config: Any) -> None:
        strict_mode_env = os.getenv("TRAINING_STRICT_MODE", "true").strip().lower()
        is_strict = strict_mode_env in {"1", "true", "yes", "on"}

        if not is_strict:
            logger.warning(
                "⚠️  STRICT MODE DISABLED via TRAINING_STRICT_MODE=false. "
                "This may produce partial-quality datasets. Use only for testing."
            )
            config.fail_on_stage_drift = False
            config.fail_on_missing_stage_artifacts = False
            return

        allow_missing = (
            os.getenv("TRAINING_ALLOW_MISSING_ARTIFACTS", "false").strip().lower()
        )
        if allow_missing in {"1", "true", "yes", "on"}:
            logger.warning(
                "⚠️  TRAINING_ALLOW_MISSING_ARTIFACTS=true. "
                "Stage 3/4 artifacts may be missing. Dataset quality may be reduced."
            )
            config.fail_on_missing_stage_artifacts = False

        allow_drift = os.getenv("TRAINING_ALLOW_STAGE_DRIFT", "false").strip().lower()
        if allow_drift in {"1", "true", "yes", "on"}:
            logger.warning(
                "⚠️  TRAINING_ALLOW_STAGE_DRIFT=true. "
                "Stage distribution may drift from targets. Curriculum balance may be affected."
            )
            config.fail_on_stage_drift = False

        logger.info("✅ STRICT MODE ENABLED (production default)")
        logger.info("   - fail_on_stage_drift: %s", config.fail_on_stage_drift)
        logger.info(
            "   - fail_on_missing_stage_artifacts: %s",
            config.fail_on_missing_stage_artifacts,
        )

    def initialize_storage_manager(
        self,
        source_paths: list[str],
    ) -> StorageManager | None:
        storage_config = self.storage_config

        if storage_config.backend != StorageBackend.LOCAL:
            return StorageManager(storage_config)

        if any(path.startswith(("gdrive:", "datasets/")) for path in source_paths):
            rclone_config = replace(storage_config, backend=StorageBackend.RCLONE)
            return StorageManager(rclone_config)

        if any(path.startswith("s3://") for path in source_paths):
            s3_config = replace(storage_config, backend=StorageBackend.S3)
            return StorageManager(s3_config)

        return None


__all__ = ["PipelineRuntimeBootstrap", "PipelineRuntimeBundle"]
