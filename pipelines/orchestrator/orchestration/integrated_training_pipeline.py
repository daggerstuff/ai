#!/usr/bin/env python3
"""
Integrated Training Pipeline Orchestrator
Combines ALL data sources for comprehensive therapeutic AI training
"""

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.configs.stages import get_all_stages
from ai.pipelines.orchestrator.data_splitter import DataSplitter
from ai.pipelines.orchestrator.configs.intake_routing import CONTINUITY_HOLDOUT_LANE
from ai.pipelines.orchestrator.ingestion.intake_gates import OrchestratorIntakeGates
from ai.pipelines.orchestrator.ingestion.intake_routing_adapter import (
    apply_intake_routing,
)
from ai.pipelines.orchestrator.storage_config import StorageBackend, get_storage_config
from ai.pipelines.orchestrator.storage_manager import StorageManager
from ai.pipelines.orchestrator.orchestration.pipeline_service_factory import (
    build_pipeline_services,
)
from ai.pipelines.orchestrator.orchestration.runtime_policy_service import (
    RuntimePolicyService,
)
from ai.pipelines.orchestrator.orchestration.storage_resolver import (
    StorageResolver,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.integrated_training_pipeline")
STAGE_MANIFEST_PATH = Path("ai/data/training_policy_manifest.json")
STAGE_DRIFT_TOLERANCE = 0.02


@dataclass
class DataSourceConfig:
    """Configuration for each data source"""

    enabled: bool = True
    target_percentage: float = 0.0  # Target percentage of final dataset
    max_samples: int | None = None
    source_path: str | None = None
    fallback_paths: tuple[str, ...] = ()


@dataclass
class IntegratedPipelineConfig:
    """Configuration for integrated training pipeline"""

    # Data source configurations
    edge_cases: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.25,  # 25% edge cases
            # Updated 2026-03-19: Use rclone path from dataset_registry.json
            source_path="drive:backups/S3-Complete/gdrive/processed/edge_cases/edge_cases_training_format.jsonl",
        )
    )

    pixel_voice: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.20,  # 20% voice-derived
            # Updated 2026-03-19: Use rclone path from dataset_registry.json
            source_path="drive:backups/S3-Complete/voice/exports",
        )
    )

    psychology_knowledge: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.15,  # 15% psychology knowledge
            # Updated 2026-03-19: Use rclone path from dataset_registry.json
            source_path="drive:backups/S3-Complete/datasets/consolidated/datasets/psychology_dataset.json",
        )
    )

    dual_persona: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.10,  # 10% dual persona
            # Updated 2026-03-19: Keep local path - loader generates synthetic data
            source_path="ai/pipelines/dual_persona",
        )
    )

    standard_therapeutic: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.30,  # 30% standard conversations
            # Updated 2026-03-19: Use rclone path - ULTIMATE_FINAL_DATASET
            # has 420K samples
            source_path=(
                "drive:backups/S3-Complete/processed_ready/"
                "ULTIMATE_FINAL_DATASET_processed.jsonl"
            ),
            fallback_paths=(
                "ai/lightning/pixelated-training/training_dataset.json",
                "ai/pipelines/orchestrator/pixelated-training/training_dataset.json",
            ),
        )
    )

    output_dir: str = "ai/lightning"
    output_filename: str = "training_dataset.json"
    target_total_samples: int = 8000
    stage_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "stage1_foundation": 0.40,
            "stage2_therapeutic_expertise": 0.25,
            "stage3_edge_stress_test": 0.20,
            "stage4_voice_persona": 0.15,
        }
    )

    # Quality settings
    enable_bias_detection: bool = True
    enable_quality_validation: bool = True
    min_quality_score: float = 0.7
    fail_on_stage_drift: bool = True  # STRICT MODE: Default to True for production
    fail_on_missing_stage_artifacts: bool = (
        True  # STRICT MODE: Default to True for production
    )

    # Progress tracking integration
    enable_progress_tracking: bool = True
    progress_tracker_path: str = "ai/lightning/therapeutic_progress_tracker.py"
    enable_tracker_sync: bool = True
    tracker_sync_output_path: str = "ai/lightning/training_run_checklist.json"
    tracker_sync_state_output_path: str = "ai/lightning/tracker_sync_state.json"
    enable_asana_sync: bool = True
    enable_beads_sync: bool = True
    enable_jira_sync: bool = True
    enable_linear_sync: bool = True
    enable_dataset_asana_sync: bool = True  # New: Dataset inventory sync to Asana
    asana_project_gid: str | None = None
    asana_section_gid: str | None = None
    asana_dataset_section_gid: str | None = None  # New: Section for dataset tasks
    asana_parent_task_gid: str | None = None
    asana_task_gid_output_path: str = "ai/lightning/training_run_asana_task_gid.txt"
    asana_task_key_mapping_output_path: str = "ai/lightning/asana_task_key_mapping.json"
    asana_task_transition_output_path: str = (
        "ai/lightning/asana_task_transition_results.json"
    )
    asana_dataset_mapping_output_path: str = (
        "ai/lightning/asana_dataset_task_mapping.json"  # New: Dataset task mapping
    )
    dataset_scan_directory: str = "ai/datasets"  # New: Directory to scan for datasets
    dataset_file_patterns: list[str] = field(
        default_factory=lambda: [".jsonl", ".json", ".csv", ".parquet"]
    )  # New: File patterns to include
    dataset_task_prefix: str = "DATASET"  # New: Prefix for dataset task names
    stage_health_report_output_path: str = (
        "ai/lightning/integrated_stage_health_report.json"
    )
    closure_pack_output_path: str = "ai/lightning/mtgc_closure_pack.json"


@dataclass
class IntegrationStats:
    """Statistics from pipeline integration"""

    total_samples: int = 0
    samples_by_source: dict[str, int] = field(default_factory=dict)
    samples_by_category: dict[str, int] = field(default_factory=dict)
    samples_by_stage: dict[str, int] = field(default_factory=dict)
    stage_balance: dict[str, dict[str, float | int]] = field(default_factory=dict)
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    quality_scores: dict[str, float] = field(default_factory=dict)
    stage_policy_enforcement: dict[str, Any] = field(default_factory=dict)
    bias_detection_results: dict[str, Any] = field(default_factory=dict)
    integration_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IntegratedTrainingPipeline:
    """
    Orchestrates integration of all data sources into unified training dataset
    """

    def __init__(self, config: IntegratedPipelineConfig | None = None):
        self.config = config or IntegratedPipelineConfig()
        self._apply_strict_mode_overrides()
        self.stats = IntegrationStats()
        self.runtime_policy_service = RuntimePolicyService(
            manifest_path=STAGE_MANIFEST_PATH,
            warning_sink=self.stats,
            default_stage_distribution=self.config.stage_distribution,
            default_stage_drift_tolerance=STAGE_DRIFT_TOLERANCE,
        )
        self.runtime_policy_service.hydrate_tracker_config(self.config)
        stage_policy = self.runtime_policy_service.load_stage_policy()
        self.config.stage_distribution = stage_policy.stage_distribution
        self.stage_quality_profiles = stage_policy.quality_profiles
        self.stage_drift_waivers = stage_policy.drift_waivers
        self.intake_gates = OrchestratorIntakeGates()
        self.storage: StorageManager | None = None
        self.storage = self._initialize_storage_manager()
        self.storage_resolver = StorageResolver(self.storage)
        services = build_pipeline_services(
            config=self.config,
            stats=self.stats,
            runtime_policy_service=self.runtime_policy_service,
            stage_drift_tolerance=STAGE_DRIFT_TOLERANCE,
            stage_drift_waivers=self.stage_drift_waivers,
            stage_quality_profiles=self.stage_quality_profiles,
            storage_resolver=self.storage_resolver,
            cache_data=self._cache_data,
            apply_intake_routing=self._apply_intake_routing,
        )
        self.asana_client = services.asana_client
        self.curriculum_enforcement_service = services.curriculum_enforcement_service
        self.asana_progress_service = services.asana_progress_service
        self.dataset_asana_sync_service = services.dataset_asana_sync_service
        self.standard_therapeutic_loader_service = (
            services.standard_therapeutic_loader_service
        )
        self.source_loader_service = services.source_loader_service
        self.dataset_quality_service = services.dataset_quality_service
        self.dataset_output_service = services.dataset_output_service
        self.run_artifact_service = services.run_artifact_service
        self.checklist_tracker_sync_service = services.checklist_tracker_sync_service
        self.dataset_assembler = services.dataset_assembler
        self.data_ingestion = services.data_ingestion

        # Initialize stage_balance with the four-stage ladder from configs/stages.py
        for stage in get_all_stages():
            if stage.id not in self.stats.stage_balance:
                self.stats.stage_balance[stage.id] = {
                    "target_share": stage.target_share,
                    "actual_samples": 0,
                }

    def _configured_source_paths(self) -> list[str]:
        return [
            self.config.edge_cases.source_path or "",
            self.config.pixel_voice.source_path or "",
            self.config.psychology_knowledge.source_path or "",
            self.config.dual_persona.source_path or "",
            self.config.standard_therapeutic.source_path or "",
        ]

    def _initialize_storage_manager(self) -> StorageManager | None:
        """Initialize storage for the configured source paths when needed."""
        storage_config = get_storage_config()
        source_paths = self._configured_source_paths()

        if storage_config.backend != StorageBackend.LOCAL:
            return StorageManager(storage_config)

        if any(path.startswith(("drive:", "datasets/")) for path in source_paths):
            rclone_config = replace(storage_config, backend=StorageBackend.RCLONE)
            return StorageManager(rclone_config)

        if any(path.startswith("s3://") for path in source_paths):
            s3_config = replace(storage_config, backend=StorageBackend.S3)
            return StorageManager(s3_config)

        return None

    def _apply_strict_mode_overrides(self) -> None:
        """
        Apply strict mode overrides from environment variables.

        STRICT MODE (default=True for production):
        - fail_on_stage_drift: Fails if stage distribution drifts >2% from targets
        - fail_on_missing_stage_artifacts: Fails if required Stage 3/4
          assets are missing

        Override with environment variables (development/testing only):
        - TRAINING_STRICT_MODE=false: Disable all strict checks (logs warnings instead)
        - TRAINING_ALLOW_MISSING_ARTIFACTS=true: Allow missing Stage 3/4 artifacts
        - TRAINING_ALLOW_STAGE_DRIFT=true: Allow stage distribution drift

        WARNING: Non-strict mode may produce partial-quality datasets.
        Use only for testing.
        """
        strict_mode_env = os.getenv("TRAINING_STRICT_MODE", "true").strip().lower()
        is_strict = strict_mode_env in {"1", "true", "yes", "on"}

        if not is_strict:
            logger.warning(
                "⚠️  STRICT MODE DISABLED via TRAINING_STRICT_MODE=false. "
                "This may produce partial-quality datasets. Use only for testing."
            )
            self.config.fail_on_stage_drift = False
            self.config.fail_on_missing_stage_artifacts = False
            return

        # Strict mode is enabled; check for specific overrides
        allow_missing = (
            os.getenv("TRAINING_ALLOW_MISSING_ARTIFACTS", "false").strip().lower()
        )
        if allow_missing in {"1", "true", "yes", "on"}:
            logger.warning(
                "⚠️  TRAINING_ALLOW_MISSING_ARTIFACTS=true. "
                "Stage 3/4 artifacts may be missing. Dataset quality may be reduced."
            )
            self.config.fail_on_missing_stage_artifacts = False

        allow_drift = os.getenv("TRAINING_ALLOW_STAGE_DRIFT", "false").strip().lower()
        if allow_drift in {"1", "true", "yes", "on"}:
            logger.warning(
                "⚠️  TRAINING_ALLOW_STAGE_DRIFT=true. "
                "Stage distribution may drift from targets."
                " Curriculum balance may be affected."
            )
            self.config.fail_on_stage_drift = False

        if is_strict:
            logger.info("✅ STRICT MODE ENABLED (production default)")
            logger.info("   - fail_on_stage_drift: %s", self.config.fail_on_stage_drift)
            logger.info(
                "   - fail_on_missing_stage_artifacts: %s",
                self.config.fail_on_missing_stage_artifacts,
            )

    def _cache_data(self, source_path: str | None) -> Path | None:
        """Download data from storage to local cache if needed."""
        return self.storage_resolver.cache_data(source_path)

    def run(self) -> dict:
        """
        Run the complete integrated pipeline

        Returns:
            Dictionary with training data and statistics
        """
        logger.info("🚀 Starting Integrated Training Pipeline (Cloud Ready)")
        logger.info("=" * 60)

        start_time = datetime.now(timezone.utc)
        all_training_data = []

        # Preflight validation for required Stage 3/4 corpora and artifacts.
        self.curriculum_enforcement_service.validate_required_stage_artifacts()
        all_training_data = self.data_ingestion.load_all_sources()

        assembly = self.dataset_assembler.assemble(all_training_data)
        balanced_data = assembly["training_data"]

        # MTGC-07: Explicit stage ratio validation
        self.curriculum_enforcement_service.validate_final_stage_balance(balanced_data)

        output_path = assembly["output_path"]
        report = assembly["report"]
        stage_health_report = assembly["stage_health_report"]
        closure_pack = assembly["closure_pack"]
        self.stats.integration_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        logger.info("=" * 60)
        logger.info("✅ Integration Complete!")
        logger.info(f"📊 Total samples: {self.stats.total_samples}")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"⏱️  Time: {self.stats.integration_time:.2f}s")

        return {
            "training_data": balanced_data,
            "statistics": self.stats,
            "output_path": output_path,
            "report": report,
            "stage_health_report": stage_health_report,
            "closure_pack": closure_pack,
        }

    def _load_standard_therapeutic(self) -> list[dict]:
        """Load standard therapeutic conversations with robust error handling"""
        return self.standard_therapeutic_loader_service.load()

    def _apply_intake_routing(
        self, records: list[dict], source_family: str
    ) -> list[dict]:
        """Apply canonical intake routing metadata to a batch of records."""
        return apply_intake_routing(
            records,
            source_family=source_family,
            intake_gates=self.intake_gates,
        )

    def _split_records_with_preferences(self, records: list[dict[str, Any]]) -> Any:
        """Compatibility seam while callers migrate off pipeline-owned split helpers."""
        return self.dataset_output_service.split_records_with_preferences(records)


def run_integrated_pipeline(config: IntegratedPipelineConfig | None = None) -> dict:
    """
    Convenience function to run the integrated training pipeline

    Args:
        config: Optional pipeline configuration

    Returns:
        Dictionary with training data and statistics
    """
    pipeline = IntegratedTrainingPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    # Run the integrated pipeline
    logger.info("🚀 Integrated Training Pipeline")
    logger.info("=" * 60)

    result = run_integrated_pipeline()

    logger.info("\n📊 Integration Report:")
    logger.info(json.dumps(result["report"], indent=2))
