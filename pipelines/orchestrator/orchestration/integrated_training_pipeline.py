#!/usr/bin/env python3
"""
Integrated Training Pipeline Orchestrator
Combines ALL data sources for comprehensive therapeutic AI training
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.configs.stages import get_all_stages
from ai.pipelines.orchestrator.data_splitter import DataSplitter
from ai.pipelines.orchestrator.ingestion.dual_persona_loader import (
    DualPersonaLoader,
)
from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import (
    EdgeCaseJSONLLoader,
)
from ai.pipelines.orchestrator.ingestion.pixel_voice_loader import (
    PixelVoiceLoader,
)
from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import (
    PsychologyKnowledgeLoader,
)
from ai.pipelines.orchestrator.quality.evidence_based_practice_validator import (
    validate_bias,
)
from ai.pipelines.orchestrator.storage_config import get_storage_config
from ai.pipelines.orchestrator.storage_manager import StorageManager
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.integrated_training_pipeline")
STAGE_MANIFEST_PATH = Path("ai/data/training_policy_manifest.json")
STAGE_DRIFT_TOLERANCE = 0.02
TASK_KEY_PATTERN = re.compile(r"\bMTGC-\d{2}\b")


@dataclass
class DataSourceConfig:
    """Configuration for each data source"""

    enabled: bool = True
    target_percentage: float = 0.0  # Target percentage of final dataset
    max_samples: int | None = None
    source_path: str | None = None


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
    enable_asana_sync: bool = True
    asana_project_gid: str | None = None
    asana_section_gid: str | None = None
    asana_parent_task_gid: str | None = None
    asana_task_gid_output_path: str = "ai/lightning/training_run_asana_task_gid.txt"
    asana_task_key_mapping_output_path: str = "ai/lightning/asana_task_key_mapping.json"
    asana_task_transition_output_path: str = (
        "ai/lightning/asana_task_transition_results.json"
    )
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
        self._hydrate_asana_config_from_env()
        self._apply_stage_distribution_from_manifest()
        self.stage_quality_profiles = self._load_stage_quality_profiles_from_manifest()
        self.stage_drift_waivers = self._load_stage_drift_waivers_from_manifest()
        self.stats = IntegrationStats()
        self._asana_access_token_cache: str | None = None
        self._asana_access_token_expiry_epoch: float = 0.0

        # Initialize storage manager for cloud access

        # Ensure we have a valid config for S3
        storage_config = get_storage_config()
        backend_env = os.getenv("DATASET_STORAGE_BACKEND")
        if (
            backend_env
            and backend_env.lower() == "s3"
            and not storage_config.s3_bucket
            and os.getenv("USER") == "vivi"
        ):
            # Convenience default for a known VPS environment when S3 is
            # explicitly selected. Do not force S3 when the backend isn't
            # explicitly configured.

            self.storage = StorageManager(storage_config)

        # Initialize stage_balance with the four-stage ladder from configs/stages.py
        for stage in get_all_stages():
            if stage.id not in self.stats.stage_balance:
                self.stats.stage_balance[stage.id] = {
                    "target_share": stage.target_share,
                    "actual_samples": 0,
                }

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

    def _load_stage_quality_profiles_from_manifest(self) -> dict[str, dict[str, Any]]:
        """Load per-stage quality profile constraints from policy manifest."""
        if not STAGE_MANIFEST_PATH.exists():
            return {}

        try:
            with open(STAGE_MANIFEST_PATH, encoding="utf-8") as handle:
                manifest = json.load(handle)

            manifest_stages = manifest.get("stages", {})
            if not isinstance(manifest_stages, dict):
                return {}

            profiles: dict[str, dict[str, Any]] = {}
            for stage_id, stage_config in manifest_stages.items():
                if not isinstance(stage_config, dict):
                    continue
                quality_profile = stage_config.get("quality_profile", {})
                if isinstance(quality_profile, dict):
                    profiles[stage_id] = quality_profile

            if profiles:
                logger.info(
                    "Loaded stage quality profiles from %s for stages: %s",
                    STAGE_MANIFEST_PATH,
                    sorted(profiles.keys()),
                )

            return profiles
        except Exception as exc:
            logger.warning(
                "Failed to load stage quality profiles from manifest: %s", exc
            )
            return {}

    def _load_stage_drift_waivers_from_manifest(self) -> dict[str, dict[str, Any]]:
        """Load optional stage drift waivers from policy manifest."""
        if not STAGE_MANIFEST_PATH.exists():
            return {}

        try:
            with open(STAGE_MANIFEST_PATH, encoding="utf-8") as handle:
                manifest = json.load(handle)

            waivers = manifest.get("stage_drift_waivers", {})
            if not isinstance(waivers, dict):
                return {}

            parsed: dict[str, dict[str, Any]] = {}
            for stage, waiver in waivers.items():
                if not isinstance(waiver, dict):
                    continue

                max_drift = waiver.get("max_drift")
                if not isinstance(max_drift, (int, float)):
                    continue

                parsed[stage] = {
                    "max_drift": float(max_drift),
                    "reason": waiver.get("reason", ""),
                    "approved_by": waiver.get("approved_by", ""),
                    "expires_at": waiver.get("expires_at"),
                }

            if parsed:
                logger.info(
                    "Loaded stage drift waivers from %s for stages: %s",
                    STAGE_MANIFEST_PATH,
                    sorted(parsed.keys()),
                )

            return parsed
        except Exception as exc:
            logger.warning("Failed to load stage drift waivers from manifest: %s", exc)
            return {}

    @staticmethod
    def _parse_iso_timestamp(value: Any) -> datetime | None:
        """Parse ISO-8601 timestamp with optional Z suffix."""
        if not isinstance(value, str) or not value.strip():
            return None

        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    def _resolve_stage_drift_tolerance(self, stage: str) -> tuple[float, bool]:
        """Resolve active drift tolerance for a stage, applying waivers when valid."""
        waiver = self.stage_drift_waivers.get(stage, {})
        if not isinstance(waiver, dict):
            return STAGE_DRIFT_TOLERANCE, False

        max_drift = waiver.get("max_drift")
        if not isinstance(max_drift, (int, float)):
            return STAGE_DRIFT_TOLERANCE, False

        expires_at_raw = waiver.get("expires_at")
        expires_at = self._parse_iso_timestamp(expires_at_raw)
        if expires_at is not None and datetime.now(timezone.utc) > expires_at:
            warning = (
                f"Drift waiver for stage '{stage}' is expired at "
                f"{expires_at.isoformat()}"
            )
            logger.warning(warning)
            self.stats.warnings.append(warning)
            return STAGE_DRIFT_TOLERANCE, False

        return float(max_drift), True

    def _hydrate_asana_config_from_env(self) -> None:
        """Load Asana configuration from environment variables when not set."""
        if self.config.asana_project_gid is None:
            self.config.asana_project_gid = os.getenv("ASANA_PROJECT_GID")
        if self.config.asana_section_gid is None:
            self.config.asana_section_gid = os.getenv("ASANA_SECTION_GID")
        if self.config.asana_parent_task_gid is None:
            self.config.asana_parent_task_gid = os.getenv("ASANA_PARENT_TASK_GID")

        enable_asana_env = os.getenv("ENABLE_ASANA_SYNC")
        if enable_asana_env is not None:
            self.config.enable_asana_sync = enable_asana_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

    def _validate_required_stage_artifacts(self) -> None:
        """
        Preflight required Stage 3/4 artifacts before training run.

        STRICT MODE (default): Raises RuntimeError if required artifacts are missing.
        NON-STRICT MODE: Logs warnings and continues (may produce partial-quality
        datasets).
        """
        required_artifacts = {
            "stage3_edge_stress_test": [
                Path("ai/pipelines/edge_case/output/edge_cases_training_format.jsonl"),
                Path("ai/pipelines/orchestrator/prompt_corpus"),
            ],
            "stage4_voice_persona": [
                Path("ai/data/tim_fletcher_voice"),
                Path("ai/training_data_consolidated/transcripts"),
            ],
        }

        missing: dict[str, list[str]] = {}
        for stage, paths in required_artifacts.items():
            if self.config.stage_distribution.get(stage, 0.0) <= 0:
                continue
            if stage_missing := [str(path) for path in paths if not path.exists()]:
                missing[stage] = stage_missing

        if not missing:
            logger.info("✅ All required stage artifacts present")
            return

        # Log all missing artifacts
        for stage, paths in missing.items():
            message = f"Missing required artifacts for {stage}: {', '.join(paths)}"
            logger.warning(message)
            self.stats.warnings.append(message)

        # In strict mode, fail immediately; in non-strict, continue with warning
        if self.config.fail_on_missing_stage_artifacts:
            missing_str = " | ".join(
                f"{stage}: {', '.join(paths)}" for stage, paths in missing.items()
            )
            error_msg = (
                "STRICT MODE: Required stage artifacts missing. "
                "To override, set TRAINING_ALLOW_MISSING_ARTIFACTS=true "
                "(development only). Missing: " + missing_str
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.warning(
                "⚠️  NON-STRICT MODE: Continuing despite missing artifacts. "
                "Dataset quality may be reduced."
            )

    def _apply_stage_distribution_from_manifest(self) -> None:
        """Load stage target percentages from policy manifest when available."""
        if not STAGE_MANIFEST_PATH.exists():
            return

        try:
            with open(STAGE_MANIFEST_PATH, encoding="utf-8") as handle:
                manifest = json.load(handle)

            manifest_stages = manifest.get("stages", {})
            if not isinstance(manifest_stages, dict):
                return

            stage_distribution: dict[str, float] = {}
            for stage_id, stage_config in manifest_stages.items():
                if not isinstance(stage_config, dict):
                    continue

                target = stage_config.get("target_percentage")
                if isinstance(target, (int, float)) and target > 0:
                    stage_distribution[stage_id] = float(target)

            total = sum(stage_distribution.values())
            if stage_distribution and abs(total - 1.0) < 1e-6:
                self.config.stage_distribution = stage_distribution
                logger.info(
                    "Loaded stage distribution from %s: %s",
                    STAGE_MANIFEST_PATH,
                    self.config.stage_distribution,
                )
            elif stage_distribution:
                logger.warning(
                    "Ignoring manifest stage distribution (sum %.4f != 1.0): %s",
                    total,
                    stage_distribution,
                )
        except Exception as exc:
            logger.warning("Failed to load stage distribution from manifest: %s", exc)

    def _resolve_s3_path(self, manifest_path: str) -> str:
        """Resolve legacy local paths to storage URIs (S3 or rclone)."""
        if manifest_path.startswith("s3://"):
            return manifest_path
        if manifest_path.startswith("drive:"):
            return manifest_path

        # Map legacy VPS/Local paths to S3 structure
        # ~/datasets/consolidated/ -> s3://pixel-data/datasets/consolidated/
        if "consolidated" in manifest_path:
            # Strip home directory or relative prefixes
            clean_path = manifest_path.replace("~/", "").replace("../", "")
            if clean_path.startswith("datasets/consolidated/"):
                return (
                    f"datasets/consolidated/"
                    f"{clean_path.split('datasets/consolidated/')[1]}"
                )

        return manifest_path

    def _cache_data(self, source_path: str | None) -> Path | None:
        """Download data from storage to local cache if needed."""
        if not source_path:
            return None

        storage_path = self._resolve_s3_path(source_path)

        # If it's a local path that exists, return it
        if not storage_path.startswith(("s3://", "drive:", "datasets/")):
            local_p = Path(os.path.expanduser(source_path))
            if local_p.exists():
                return local_p
            return None

        # Define cache location
        cache_dir = Path.home() / ".cache" / "pixelated" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique filename based on the path to avoid collisions
        safe_name = (
            storage_path.replace("/", "_").replace("s3:__", "").replace(":", "_")
        )
        cached_file = cache_dir / safe_name

        if cached_file.exists():
            logger.info(f"Using cached file: {cached_file}")
            return cached_file

        logger.info(f"Downloading {storage_path} to cache...")
        try:
            # Handle different storage backends
            if storage_path.startswith("drive:"):
                # Rclone path - pass directly to storage manager with full path
                # Storage manager expects the full rclone path
                # e.g., "drive:backups/S3-Complete/..."
                self.storage.download_file(storage_path, cached_file)
                return cached_file
            elif storage_path.startswith("s3://"):
                download_key = storage_path
                if storage_path.startswith(f"s3://{self.storage.config.s3_bucket}/"):
                    download_key = storage_path.replace(
                        f"s3://{self.storage.config.s3_bucket}/", ""
                    )
                self.storage.download_file(download_key, cached_file)
                return cached_file
            elif storage_path.startswith("datasets/"):
                # Relative path - use as-is
                self.storage.download_file(storage_path, cached_file)
                return cached_file

            return None
        except Exception as e:
            logger.error(f"Failed to download {storage_path}: {e}")
            return None

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
        self._validate_required_stage_artifacts()

        # 1. Load Edge Case Data
        if self.config.edge_cases.enabled:
            cached_path = self._cache_data(self.config.edge_cases.source_path)
            edge_data = self._load_edge_cases(
                cached_path
            )  # Modified to pass cached_path
            all_training_data.extend(edge_data)
            self.stats.samples_by_source["edge_cases"] = len(edge_data)
            logger.info(f"✅ Loaded {len(edge_data)} edge case examples")

        # 2. Load Pixel Voice Data
        if self.config.pixel_voice.enabled:
            cached_path = self._cache_data(self.config.pixel_voice.source_path)
            voice_data = self._load_pixel_voice(
                cached_path
            )  # Modified to pass cached_path
            all_training_data.extend(voice_data)
            self.stats.samples_by_source["pixel_voice"] = len(voice_data)
            logger.info(f"✅ Loaded {len(voice_data)} voice-derived examples")

        # 3. Load Psychology Knowledge
        if self.config.psychology_knowledge.enabled:
            cached_path = self._cache_data(self.config.psychology_knowledge.source_path)
            psych_data = self._load_psychology_knowledge(
                cached_path
            )  # Modified to pass cached_path
            all_training_data.extend(psych_data)
            self.stats.samples_by_source["psychology_knowledge"] = len(psych_data)
            logger.info(f"✅ Loaded {len(psych_data)} psychology knowledge examples")

        # 4. Load Dual Persona Data
        if self.config.dual_persona.enabled:
            cached_path = self._cache_data(self.config.dual_persona.source_path)
            persona_data = self._load_dual_persona(
                cached_path
            )  # Modified to pass cached_path
            all_training_data.extend(persona_data)
            self.stats.samples_by_source["dual_persona"] = len(persona_data)
            logger.info(f"✅ Loaded {len(persona_data)} dual persona examples")

        # 5. Load Standard Therapeutic Conversations
        if self.config.standard_therapeutic.enabled:
            # Standard therapeutic loader handles its own path resolution
            # and caching internally as it tries multiple paths.
            standard_data = self._load_standard_therapeutic()
            all_training_data.extend(standard_data)
            self.stats.samples_by_source["standard_therapeutic"] = len(standard_data)
            logger.info(f"✅ Loaded {len(standard_data)} standard therapeutic examples")

        # 6. Balance dataset according to target percentages
        balanced_data, stage_segments = self._balance_dataset(all_training_data)

        # 7. Run bias detection if enabled
        if self.config.enable_bias_detection:
            balanced_data = self._run_bias_detection(balanced_data)

        # 8. Run quality validation if enabled
        if self.config.enable_quality_validation:
            balanced_data = self._run_quality_validation(balanced_data)

        # Recompute stage counts after all filters and record drift.
        self._validate_final_stage_balance(balanced_data)

        # 9. Save integrated dataset
        output_path = self._save_dataset(balanced_data)
        self._write_stage_outputs(stage_segments)
        self._write_split_outputs(balanced_data)

        # 10. Generate integration report
        self.stats.total_samples = len(balanced_data)
        self.stats.samples_by_category = dict(self.stats.samples_by_stage)
        self.stats.integration_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        report = self._generate_report()
        stage_health_report = self._build_stage_health_report(report)
        self._write_stage_health_report(stage_health_report)
        self._sync_run_checklist(report)
        closure_pack = self._build_mtgc_closure_pack(report, stage_health_report)
        self._write_mtgc_closure_pack(closure_pack)

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

    def _sync_run_checklist(self, report: dict[str, Any]) -> None:
        """Persist checklist payload and optionally emit to tracker webhook."""
        if not self.config.enable_tracker_sync:
            return

        stage_balance = report.get("stage_balance", {})
        drift_failures = []
        for stage, metrics in stage_balance.items():
            if not isinstance(metrics, dict):
                continue
            drift = metrics.get("drift_vs_target")
            if isinstance(drift, (int, float)) and abs(drift) > STAGE_DRIFT_TOLERANCE:
                drift_failures.append(stage)

        checklist = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": report.get("total_samples", 0),
            "stage_drift_within_tolerance": not drift_failures,
            "stage_drift_failures": drift_failures,
            "split_counts": report.get("split_counts", {}),
            "ops_freshness": self._collect_ops_freshness(),
            "stage_health_report_path": self.config.stage_health_report_output_path,
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
            "report": report,
        }

        output_path = Path(self.config.tracker_sync_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(checklist, handle, indent=2)

        logger.info("🧾 Training run checklist saved to %s", output_path)

        webhook_url = os.getenv("TRAINING_CHECKLIST_WEBHOOK_URL", "").strip()
        if not webhook_url:
            self._sync_to_asana(checklist, output_path)
            return

        try:
            request = urllib.request.Request(
                webhook_url,
                data=json.dumps(checklist).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                logger.info(
                    "📡 Checklist webhook sent (status=%s)",
                    response.status,
                )
        except (urllib.error.URLError, TimeoutError) as exc:
            warning = f"Checklist webhook sync failed: {exc}"
            logger.warning(warning)
            self.stats.warnings.append(warning)

        self._sync_to_asana(checklist, output_path)

    @staticmethod
    def _is_valid_gid(value: str | None) -> bool:
        """Asana gids are numeric strings."""
        return isinstance(value, str) and bool(re.fullmatch(r"\d+", value))

    def _asana_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> Any:
        """Issue a request to Asana API v1.0 and return parsed response data."""
        token = os.getenv("ASANA_ACCESS_TOKEN", "").strip()
        token = self._get_asana_access_token()

        url = f"https://app.asana.com/api/1.0{path}"
        if query_params:
            encoded_params = urllib.parse.urlencode(query_params, doseq=True)
            url = f"{url}?{encoded_params}"

        body = None
        if payload is not None:
            body = json.dumps({"data": payload}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            if not isinstance(response_data, dict) or "data" not in response_data:
                raise RuntimeError("Invalid Asana response payload")
            return response_data["data"]

    def _get_asana_access_token(self) -> str:
        """Get Asana bearer token from direct access token or OAuth refresh flow."""
        now = time.time()
        if (
            self._asana_access_token_cache
            and self._asana_access_token_expiry_epoch > now + 30
        ):
            return self._asana_access_token_cache

        direct_token = os.getenv("ASANA_ACCESS_TOKEN", "").strip()
        if direct_token:
            self._asana_access_token_cache = direct_token
            # No known expiry for manually-provided tokens; keep short cache window.
            self._asana_access_token_expiry_epoch = now + 300
            return direct_token

        client_id = os.getenv("ASANA_CLIENT_ID", os.getenv("ASANA_CID", "")).strip()
        client_secret = os.getenv(
            "ASANA_CLIENT_SECRET", os.getenv("ASANA_CS", "")
        ).strip()
        refresh_token = os.getenv("ASANA_REFRESH_TOKEN", "").strip()

        # Optional one-time bootstrap using auth code if refresh token is not yet set.
        # Expected env vars: ASANA_AUTH_CODE and ASANA_REDIRECT_URI
        if client_id and client_secret and not refresh_token:
            auth_code = os.getenv("ASANA_AUTH_CODE", "").strip()
            redirect_uri = os.getenv("ASANA_REDIRECT_URI", "").strip()
            if auth_code and redirect_uri:
                token_request_body = urllib.parse.urlencode(
                    {
                        "grant_type": "authorization_code",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect_uri,
                        "code": auth_code,
                    }
                ).encode("utf-8")

                request = urllib.request.Request(
                    "https://app.asana.com/-/oauth_token",
                    data=token_request_body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(request, timeout=15) as response:
                        token_response = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    details = exc.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"Asana OAuth auth-code exchange failed: {details}"
                    ) from exc
                except urllib.error.URLError as exc:
                    raise RuntimeError(
                        f"Asana OAuth auth-code exchange failed: {exc}"
                    ) from exc

                if not isinstance(token_response, dict):
                    raise RuntimeError(
                        "Asana OAuth auth-code exchange returned invalid payload"
                    )

                exchanged_access = str(token_response.get("access_token", "")).strip()
                exchanged_refresh = str(token_response.get("refresh_token", "")).strip()
                expires_in_raw = token_response.get("expires_in", 3600)

                if not exchanged_access:
                    raise RuntimeError(
                        "Asana OAuth auth-code exchange returned no access_token"
                    )

                if exchanged_refresh:
                    refresh_token = exchanged_refresh
                    # Persist bootstrap tokens for operator visibility.
                    token_out = Path("ai/lightning/asana_oauth_tokens.json")
                    token_out.parent.mkdir(parents=True, exist_ok=True)
                    with open(token_out, "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                                "note": (
                                    "Move refresh_token into ASANA_REFRESH_TOKEN "
                                    "and clear ASANA_AUTH_CODE."
                                ),
                                "refresh_token": exchanged_refresh,
                            },
                            handle,
                            indent=2,
                        )
                    logger.info("Asana OAuth bootstrap tokens saved to %s", token_out)

                try:
                    expires_in = float(expires_in_raw)
                except (TypeError, ValueError):
                    expires_in = 3600.0

                self._asana_access_token_cache = exchanged_access
                self._asana_access_token_expiry_epoch = now + max(expires_in, 60.0)
                return exchanged_access

        if not (client_id and client_secret and refresh_token):
            raise RuntimeError(
                "Set ASANA_ACCESS_TOKEN or OAuth vars "
                "(ASANA_CLIENT_ID/ASANA_CLIENT_SECRET/ASANA_REFRESH_TOKEN), "
                "or bootstrap with ASANA_AUTH_CODE + ASANA_REDIRECT_URI."
            )

        token_request_body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            "https://app.asana.com/-/oauth_token",
            data=token_request_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                token_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Asana OAuth token refresh failed: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Asana OAuth token refresh failed: {exc}") from exc

        if not isinstance(token_response, dict):
            raise RuntimeError("Asana OAuth token refresh returned invalid payload")

        access_token = str(token_response.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError("Asana OAuth token refresh returned no access_token")

        expires_in_raw = token_response.get("expires_in", 3600)
        try:
            expires_in = float(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600.0

        self._asana_access_token_cache = access_token
        self._asana_access_token_expiry_epoch = now + max(expires_in, 60.0)
        return access_token

    def _sync_to_asana(self, checklist: dict[str, Any], checklist_path: Path) -> None:
        """Create/update Asana task linkage for this training checklist run."""
        if not self.config.enable_asana_sync:
            return

        project_gid = self.config.asana_project_gid
        if not self._is_valid_gid(project_gid):
            warning = "Asana sync skipped: ASANA_PROJECT_GID missing or invalid"
            logger.warning(warning)
            self.stats.warnings.append(warning)
            return

        section_gid = self.config.asana_section_gid
        parent_gid = self.config.asana_parent_task_gid
        if section_gid and not self._is_valid_gid(section_gid):
            warning = (
                "Asana sync skipped section assignment: ASANA_SECTION_GID is invalid"
            )
            logger.warning(warning)
            self.stats.warnings.append(warning)
            section_gid = None
        if parent_gid and not self._is_valid_gid(parent_gid):
            warning = (
                "Asana sync skipped parent linkage: ASANA_PARENT_TASK_GID is invalid"
            )
            logger.warning(warning)
            self.stats.warnings.append(warning)
            parent_gid = None

        timestamp = checklist.get(
            "generated_at", datetime.now(timezone.utc).isoformat()
        )
        drift_ok = checklist.get("stage_drift_within_tolerance", False)
        total_samples = checklist.get("total_samples", 0)
        status_icon = "✅" if drift_ok else "⚠️"
        task_name = f"{status_icon} Training Checklist {timestamp}"

        notes_lines = [
            "Automated training checklist sync from integrated pipeline.",
            f"Generated at: {timestamp}",
            f"Total samples: {total_samples}",
            f"Stage drift within tolerance: {drift_ok}",
            f"Checklist artifact: {checklist_path}",
        ]
        drift_failures = checklist.get("stage_drift_failures", [])
        if drift_failures:
            notes_lines.append("Stage drift failures: " + ", ".join(drift_failures))

        task_payload: dict[str, Any] = {
            "name": task_name,
            "notes": "\n".join(notes_lines),
            "projects": [project_gid],
        }
        if section_gid:
            task_payload["memberships"] = [
                {"project": project_gid, "section": section_gid}
            ]

        task_data: dict[str, Any]
        try:
            if parent_gid:
                task_data = self._asana_request(
                    "POST",
                    f"/tasks/{parent_gid}/subtasks",
                    task_payload,
                )
            else:
                task_data = self._asana_request("POST", "/tasks", task_payload)

            run_task_gid = task_data.get("gid")
            if not self._is_valid_gid(run_task_gid):
                raise RuntimeError("Asana returned invalid task gid")

            stage_balance = checklist.get("report", {}).get("stage_balance", {})
            for stage, metrics in stage_balance.items():
                if not isinstance(metrics, dict):
                    continue
                actual = metrics.get("final_actual", metrics.get("actual", 0))
                drift = metrics.get("drift_vs_target", 0)
                subtask_name = f"{stage}: samples={actual}, drift={drift}"
                self._asana_request(
                    "POST",
                    f"/tasks/{run_task_gid}/subtasks",
                    {"name": subtask_name},
                )

            self._asana_request(
                "POST",
                f"/tasks/{run_task_gid}/stories",
                {
                    "text": (
                        "Linked checklist generated by integrated training pipeline.\n"
                        f"Artifact path: {checklist_path}"
                    )
                },
            )

            gid_path = Path(self.config.asana_task_gid_output_path)
            gid_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gid_path, "w", encoding="utf-8") as gid_handle:
                gid_handle.write(str(run_task_gid))

            logger.info(
                "📎 Asana linkage created for checklist task gid=%s", run_task_gid
            )

            self._sync_mtgc_task_transitions(checklist, project_gid)
        except Exception as exc:
            warning = f"Asana sync failed: {exc}"
            logger.warning(warning)
            self.stats.warnings.append(warning)

    def _has_asana_auth_context(self) -> bool:
        """Return True when any valid auth path is configured for Asana API use."""
        if os.getenv("ASANA_ACCESS_TOKEN", "").strip():
            return True

        client_id = os.getenv("ASANA_CLIENT_ID", os.getenv("ASANA_CID", "")).strip()
        client_secret = os.getenv(
            "ASANA_CLIENT_SECRET", os.getenv("ASANA_CS", "")
        ).strip()
        refresh_token = os.getenv("ASANA_REFRESH_TOKEN", "").strip()

        return bool(client_id and client_secret and refresh_token)

    @staticmethod
    def _extract_task_key(name: str | None) -> str | None:
        """Extract MTGC task key from an Asana task name."""
        if not isinstance(name, str):
            return None
        match = TASK_KEY_PATTERN.search(name)
        if not match:
            return None
        return match.group(0)

    def _load_task_key_gid_mapping(self, project_gid: str) -> dict[str, str]:
        """Load MTGC task-key -> Asana gid mapping from file and project tasks."""
        mapping: dict[str, str] = {}
        mapping_path = Path(self.config.asana_task_key_mapping_output_path)

        if mapping_path.exists():
            try:
                with open(mapping_path, encoding="utf-8") as handle:
                    existing = json.load(handle)
                if isinstance(existing, dict):
                    for task_key, gid in existing.items():
                        if isinstance(task_key, str) and self._is_valid_gid(str(gid)):
                            mapping[task_key] = str(gid)
            except Exception as exc:
                warning = f"Failed to read task key mapping file: {exc}"
                logger.warning(warning)
                self.stats.warnings.append(warning)

        try:
            tasks = self._asana_request(
                "GET",
                f"/projects/{project_gid}/tasks",
                payload=None,
                query_params={"limit": 100, "opt_fields": "gid,name"},
            )
            if isinstance(tasks, list):
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    task_key = self._extract_task_key(task.get("name"))
                    gid = str(task.get("gid", "")).strip()
                    if task_key and self._is_valid_gid(gid):
                        mapping[task_key] = gid
        except Exception as exc:
            warning = f"Failed to resolve MTGC task keys from Asana project: {exc}"
            logger.warning(warning)
            self.stats.warnings.append(warning)

        try:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_path, "w", encoding="utf-8") as handle:
                json.dump(mapping, handle, indent=2)
            logger.info("🗺️  Asana task key mapping saved to %s", mapping_path)
        except Exception as exc:
            warning = f"Failed to write task key mapping file: {exc}"
            logger.warning(warning)
            self.stats.warnings.append(warning)

        return mapping

    def _derive_mtgc_status_targets(
        self, checklist: dict[str, Any], task_key_gid_map: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        """Map checklist signals to MTGC task key completion targets."""
        split_counts = checklist.get("split_counts", {})
        split_aggregate = (
            split_counts.get("aggregate") if isinstance(split_counts, dict) else None
        )
        ops_freshness = checklist.get("ops_freshness", {})
        ops_all_fresh = (
            bool(ops_freshness.get("all_fresh", False))
            if isinstance(ops_freshness, dict)
            else False
        )

        mapping_resolved = all(
            task_key in task_key_gid_map
            for task_key in ("MTGC-09", "MTGC-10", "MTGC-12")
        )

        signal_map: dict[str, dict[str, Any]] = {
            "asana_sync.authenticated": {
                "task_keys": ["MTGC-09"],
                "passed": self._has_asana_auth_context(),
            },
            "checklist_mapping.resolved": {
                "task_keys": ["MTGC-10"],
                "passed": mapping_resolved,
            },
            "ops_freshness.all_fresh": {
                "task_keys": ["MTGC-12"],
                "passed": ops_all_fresh,
            },
            "stage_drift_within_tolerance": {
                "task_keys": ["MTGC-01", "MTGC-08"],
                "passed": bool(checklist.get("stage_drift_within_tolerance", False)),
            },
            "split_counts.aggregate_present": {
                "task_keys": ["MTGC-06"],
                "passed": isinstance(split_aggregate, dict),
            },
        }

        targets: dict[str, dict[str, Any]] = {}
        for signal_key, signal in signal_map.items():
            task_keys = signal.get("task_keys", [])
            passed = bool(signal.get("passed", False))
            for task_key in task_keys:
                entry = targets.setdefault(
                    task_key,
                    {
                        "passed": True,
                        "signal_keys": [],
                    },
                )
                entry["passed"] = bool(entry["passed"]) and passed
                entry["signal_keys"].append(signal_key)

        return targets

    def _sync_mtgc_task_transitions(
        self, checklist: dict[str, Any], project_gid: str
    ) -> None:
        """Apply authenticated Asana task transitions from checklist-derived signals."""
        task_key_gid_map = self._load_task_key_gid_mapping(project_gid)
        targets = self._derive_mtgc_status_targets(checklist, task_key_gid_map)
        generated_at = str(
            checklist.get("generated_at", datetime.now(timezone.utc).isoformat())
        )

        transition_results: dict[str, Any] = {
            "generated_at": generated_at,
            "task_key_gid_map": task_key_gid_map,
            "targets": targets,
            "updates": {},
        }

        for task_key, target in targets.items():
            gid = task_key_gid_map.get(task_key)
            if not self._is_valid_gid(gid):
                warning = (
                    f"Asana transition skipped for {task_key}: missing gid mapping"
                )
                logger.warning(warning)
                self.stats.warnings.append(warning)
                transition_results["updates"][task_key] = {
                    "updated": False,
                    "reason": "missing_gid_mapping",
                    "signals": target.get("signal_keys", []),
                }
                continue

            should_complete = bool(target.get("passed", False))
            signal_keys = target.get("signal_keys", [])

            try:
                self._asana_request(
                    "PUT",
                    f"/tasks/{gid}",
                    payload={"completed": should_complete},
                )
                self._asana_request(
                    "POST",
                    f"/tasks/{gid}/stories",
                    payload={
                        (
                        f"Checklist transition update ({generated_at}): "
                        f"completed={should_complete}; signals={', '.join(signal_keys)}"
                        )
                    },
                )
                transition_results["updates"][task_key] = {
                    "updated": True,
                    "gid": gid,
                    "completed": should_complete,
                    "signals": signal_keys,
                }
            except Exception as exc:
                warning = f"Asana transition failed for {task_key} ({gid}): {exc}"
                logger.warning(warning)
                self.stats.warnings.append(warning)
                transition_results["updates"][task_key] = {
                    "updated": False,
                    "gid": gid,
                    "completed": should_complete,
                    "signals": signal_keys,
                    "error": str(exc),
                }

        transition_path = Path(self.config.asana_task_transition_output_path)
        try:
            transition_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transition_path, "w", encoding="utf-8") as handle:
                json.dump(transition_results, handle, indent=2)
            logger.info("🔁 Asana task transition results saved to %s", transition_path)
        except Exception as exc:
            warning = f"Failed to write Asana transition results: {exc}"
            logger.warning(warning)
            self.stats.warnings.append(warning)

    def _collect_ops_freshness(self) -> dict[str, Any]:
        """Collect ops freshness for checklist sync."""
        threshold_hours_raw = os.getenv("TRAINING_OPS_FRESHNESS_HOURS", "24").strip()
        try:
            threshold_hours = float(threshold_hours_raw)
        except ValueError:
            threshold_hours = 24.0

        now = datetime.now(timezone.utc)
        targets = {
            "inventory": Path(
                os.getenv(
                    "TRAINING_OPS_INVENTORY_PATH",
                    "ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json",
                )
            ),
            "prompt_mirror": Path(
                os.getenv(
                    "TRAINING_OPS_PROMPT_MIRROR_PATH",
                    "ai/pipelines/orchestrator/prompt_corpus",
                )
            ),
            "voice_export": Path(
                os.getenv(
                    "TRAINING_OPS_VOICE_EXPORT_PATH",
                    "ai/training_data_consolidated/transcripts",
                )
            ),
        }

        checks: dict[str, Any] = {}
        all_fresh = True

        for key, path in targets.items():
            if not path.exists():
                checks[key] = {
                    "path": str(path),
                    "exists": False,
                    "fresh": False,
                    "reason": "missing",
                }
                all_fresh = False
                continue

            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_hours = (now - modified_at).total_seconds() / 3600.0
            fresh = age_hours <= threshold_hours
            checks[key] = {
                "path": str(path),
                "exists": True,
                "fresh": fresh,
                "modified_at": modified_at.isoformat(),
                "age_hours": round(age_hours, 3),
                "threshold_hours": threshold_hours,
            }
            if not fresh:
                all_fresh = False

        return {
            "checked_at": now.isoformat(),
            "threshold_hours": threshold_hours,
            "all_fresh": all_fresh,
            "checks": checks,
        }

    def _load_edge_cases(self, file_path: Path | None = None) -> list[dict]:
        """Load edge case training data"""
        try:
            loader = EdgeCaseJSONLLoader(file_path=file_path)

            if not loader.check_pipeline_output_exists():
                warning = "Edge case data not found. Run edge case pipeline first."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format(loader.load_edge_cases())

        except Exception as e:
            error = f"Failed to load edge cases: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_pixel_voice(self, file_path: Path | None = None) -> list[dict]:
        """Load Pixel Voice pipeline data"""
        try:
            loader = PixelVoiceLoader(file_path=file_path)

            if not loader.check_pipeline_output_exists():
                warning = "Pixel Voice data not found. Run Pixel Voice pipeline first."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format(loader.load_therapeutic_pairs())

        except Exception as e:
            error = f"Failed to load Pixel Voice data: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_psychology_knowledge(self, file_path: Path | None = None) -> list[dict]:
        """Load psychology knowledge base"""
        try:
            loader = PsychologyKnowledgeLoader(file_path=file_path)

            if not loader.check_knowledge_base_exists():
                warning = "Psychology knowledge base not found."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format(loader.load_concepts())

        except Exception as e:
            error = f"Failed to load psychology knowledge: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_dual_persona(self, file_path: Path | None = None) -> list[dict]:
        """Load dual persona training data"""
        try:
            loader = DualPersonaLoader(file_path=file_path)

            # Dual persona loader will generate synthetic data if none exists
            return loader.convert_to_training_format(loader.load_dialogues())

        except Exception as e:
            error = f"Failed to load dual persona data: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_standard_therapeutic(self) -> list[dict]:
        """Load standard therapeutic conversations with robust error handling"""
        source_root = self.config.standard_therapeutic.source_path or ""

        # Check if source is a rclone JSONL file
        if source_root.startswith("drive:") and source_root.endswith(".jsonl"):
            cached_path = self._cache_data(source_root)
            if cached_path and cached_path.exists():
                return self._load_jsonl_file(cached_path)
            warning = f"Could not cache rclone file: {source_root}"
            logger.warning(warning)
            self.stats.warnings.append(warning)
            return []

        # Try multiple file locations for local paths
        possible_files = [
            Path(source_root) / "training_dataset.json",
            Path("ai/lightning/pixelated-training/training_dataset.json"),
            Path("ai/pipelines/orchestrator/pixelated-training/training_dataset.json"),
        ]

        # Try each file until one loads successfully
        raw_conversations = []
        last_error = None

        for standard_file in possible_files:
            if not standard_file.exists():
                continue

            logger.info(f"Attempting to load from: {standard_file}")
            try:
                raw_conversations = self._try_load_json_file(standard_file)
                if raw_conversations:
                    break
            except Exception as e:
                last_error = e
                continue

        if not raw_conversations:
            self._handle_load_error(possible_files, last_error)
            return []

        return self._normalize_conversations(raw_conversations)

    def _load_jsonl_file(
        self, file_path: Path, max_samples: int | None = None
    ) -> list[dict]:
        """Load a JSONL file and return list of conversations"""
        conversations = []
        try:
            with open(file_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # Convert to training format if it has messages
                        if "messages" in record:
                            conversations.append(record)
                        elif "conversation" in record:
                            conversations.append(record)
                        else:
                            conversations.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line {i + 1}: {e}")
                        continue
            logger.info(
                f"✅ Loaded {len(conversations)} conversations from JSONL: {file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")
        return conversations

    def _try_load_json_file(self, file_path: Path) -> list:
        """Helper to try loading a JSON file and return list of conversations"""
        try:
            with open(file_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                logger.info(
                    f"✅ Loaded {len(raw_data)} conversations from {file_path} "
                    f"(list format)"
                )
                return raw_data
            if isinstance(raw_data, dict):
                conversations = raw_data.get("conversations", [])
                if conversations:
                    logger.info(
                        f"✅ Loaded {len(conversations)} conversations from "
                        f"{file_path} (dict format)"
                    )
                    return conversations
                logger.warning(f"File {file_path} loaded but no conversations found")
            else:
                logger.warning(f"Unexpected data type in {file_path}: {type(raw_data)}")
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON parsing error in {file_path} at position {e.pos}: {e.msg}"
            )
            raise e
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            raise e
        return []

    def _handle_load_error(
        self, possible_files: list[Path], last_error: Exception | None
    ):
        """Helper to handle load errors"""
        if last_error:
            error_msg = (
                f"Failed to load standard therapeutic data. Last error: {last_error}"
            )
        else:
            file_list = [str(f) for f in possible_files]
            error_msg = f"Standard therapeutic data not found in: {file_list}"
        logger.error(error_msg)
        self.stats.errors.append(error_msg)

    def _normalize_conversations(self, conversations: list) -> list[dict]:
        """Normalize raw conversations to training format"""
        training_data = []
        for conv in conversations:
            if not isinstance(conv, dict):
                continue

            text = self._extract_text_from_conv(conv)
            if text:
                training_data.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": "standard_therapeutic",
                            "is_edge_case": False,
                        },
                    }
                )

        logger.info(
            f"✅ Converted {len(training_data)} standard therapeutic examples "
            f"to training format"
        )
        return training_data

    def _extract_text_from_conv(self, conv: dict) -> str:
        """Extract text content from a conversation dict"""
        text = conv.get("text", "")
        if text:
            return text

        # Check for 'conversation' key (list format)
        conversation_array = conv.get("conversation", [])
        if conversation_array:
            parts = self._parts_from_messages(conversation_array)
            if parts:
                return "\n".join(parts)

        # Try messages format
        messages = conv.get("messages", [])
        if messages:
            parts = self._parts_from_messages(messages)
            if parts:
                return "\n".join(parts)

        return conv.get("content", "")

    def _parts_from_messages(self, messages: list) -> list[str]:
        """Extract parts from a list of message dicts"""
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    parts.append(f"{role.capitalize()}: {content}")
        return parts

    def _balance_dataset(
        self, data: list[dict]
    ) -> tuple[list[dict], dict[str, list[dict]]]:
        """Balance dataset according to stage distribution."""
        logger.info("⚖️  Balancing dataset by stage...")

        stage_buckets: dict[str, list[dict]] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            stage_buckets.setdefault(stage, []).append(item)

        balanced: list[dict] = []
        stage_segments: dict[str, list[dict]] = {}

        for stage, percentage in self.config.stage_distribution.items():
            target_count = int(self.config.target_total_samples * percentage)
            bucket = stage_buckets.get(stage, [])

            if not bucket:
                warning = f"No data found for stage '{stage}' (target: {target_count})."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                self.stats.stage_balance[stage] = {
                    "target": target_count,
                    "available": 0,
                    "actual": 0,
                }
                continue

            if len(bucket) <= target_count:
                stage_sample = bucket
                if len(bucket) < target_count:
                    warning = (
                        f"Stage '{stage}' has only {len(bucket)} samples "
                        f"(target: {target_count})."
                    )
                    logger.warning(warning)
                    self.stats.warnings.append(warning)
            else:
                stage_sample = random.sample(bucket, target_count)

            balanced.extend(stage_sample)
            stage_segments[stage] = stage_sample
            actual = len(stage_sample)
            self.stats.samples_by_stage[stage] = actual
            self.stats.stage_balance[stage] = {
                "target": target_count,
                "available": len(bucket),
                "actual": actual,
            }

        logger.info(f"   Stage-balanced to {len(balanced)} samples")
        return balanced, stage_segments

    def _validate_final_stage_balance(self, data: list[dict]) -> None:
        """Record final stage percentages and warn when drift exceeds tolerance."""
        if not data:
            return

        counts: dict[str, int] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            counts[stage] = counts.get(stage, 0) + 1

        total = len(data)
        self.stats.samples_by_stage = counts
        drift_violations: list[str] = []

        for stage, target in self.config.stage_distribution.items():
            actual_count = counts.get(stage, 0)
            actual_share = actual_count / total if total else 0.0
            drift = actual_share - target
            tolerance, waiver_applied = self._resolve_stage_drift_tolerance(stage)

            balance = self.stats.stage_balance.get(stage, {})
            balance.update(
                {
                    "final_actual": actual_count,
                    "final_share": round(actual_share, 6),
                    "drift_vs_target": round(drift, 6),
                    "drift_tolerance": round(tolerance, 6),
                    "drift_waiver_applied": waiver_applied,
                }
            )
            self.stats.stage_balance[stage] = balance

            if abs(drift) > tolerance:
                warning = (
                    f"Stage '{stage}' drift {drift:+.3f} exceeds tolerance "
                    f"{tolerance:.2f} (target={target:.2%}, actual={actual_share:.2%}, "
                    f"waiver_applied={waiver_applied})"
                )
                logger.warning(warning)
                self.stats.warnings.append(warning)
                drift_violations.append(stage)

        if self.config.fail_on_stage_drift and drift_violations:
            raise RuntimeError(
                "Stage balance drift exceeds tolerance: " + ", ".join(drift_violations)
            )

    def _run_bias_detection(self, data: list[dict]) -> list[dict]:
        """Run bias detection on training data"""
        logger.info("🔍 Running bias detection...")

        try:
            return self._extracted_from__run_bias_detection_6(data)
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            return data

    # TODO Rename this here and in `_run_bias_detection`
    def _extracted_from__run_bias_detection_6(self, data):
        flagged_count = 0
        filtered_data = []

        for item in data:
            text = item.get("text", "")
            if validate_bias(text):
                filtered_data.append(item)
            else:
                flagged_count += 1

        self.stats.bias_detection_results = {
            "total_checked": len(data),
            "flagged": flagged_count,
            "passed": len(filtered_data),
        }

        logger.info(f"   Flagged {flagged_count} items for bias")
        return filtered_data

    def _run_quality_validation(self, data: list[dict]) -> list[dict]:
        """Run quality validation on training data using Quality Scoring v1"""
        logger.info("✓ Running quality validation...")

        try:
            from ai.pipelines.orchestrator.quality.quality_filter_v1 import (
                QualityFilterV1,
            )

            # Use Quality Scoring v1 for validation
            quality_filter = QualityFilterV1(
                min_decision="curate",
                min_composite=0.6,  # Configurable threshold
                enabled=True,
            )

            filtered, scoring_results = quality_filter.filter_batch(data)

            # Add quality scores to metadata
            for item, result in zip(data, scoring_results):
                if "metadata" not in item:
                    item["metadata"] = {}
                item["metadata"]["quality_scoring_v1"] = result

            if self.stage_quality_profiles:
                filtered, stage_filtered_count = self._apply_stage_quality_profiles(
                    filtered
                )
                if stage_filtered_count > 0:
                    logger.info(
                        "   Stage policy filters removed %s additional samples",
                        stage_filtered_count,
                    )

            logger.info(
                f"   Validated {len(data)} samples, "
                f"filtered to {len(filtered)} high-quality samples"
            )
            return filtered

        except ImportError:
            logger.warning(
                "Quality Scoring v1 not available, skipping quality validation"
            )
            logger.info(f"   Validated {len(data)} samples (no filtering)")
            return data

    def _apply_stage_quality_profiles(
        self,
        data: list[dict],
    ) -> tuple[list[dict], int]:
        """Apply stage-specific quality profile constraints from manifest policies."""
        kept: list[dict] = []
        removed_count = 0
        removed_by_stage: dict[str, int] = {}
        crisis_override_by_stage: dict[str, int] = {}
        fail_reasons_by_stage: dict[str, dict[str, int]] = {}

        for item in data:
            metadata = item.get("metadata", {})
            stage = metadata.get("stage", "stage1_foundation")
            profile = self.stage_quality_profiles.get(stage)
            if not profile:
                kept.append(item)
                continue

            score = metadata.get("quality_scoring_v1", {})
            signals = score.get("signals", {}) if isinstance(score, dict) else {}
            empathy = float(signals.get("empathy", 0.0))
            safety = 1.0 - float(signals.get("harm", 0.0))
            bias_score = metadata.get("bias_score")
            if isinstance(bias_score, (int, float)):
                bias = float(bias_score)
            else:
                bias = 0.0

            min_empathy = profile.get("min_empathy")
            min_safety = profile.get("min_safety")
            bias_max = profile.get("bias_max")
            requires_reasoning_metadata = bool(
                profile.get("requires_reasoning_metadata", False)
            )
            requires_voice_signature = bool(
                profile.get("requires_voice_signature", False)
            )
            allow_crisis_override = bool(profile.get("allow_crisis_override", False))

            if (
                allow_crisis_override
                and metadata.get("crisis_intensity") == "very_high"
            ):
                crisis_override_by_stage[stage] = (
                    crisis_override_by_stage.get(stage, 0) + 1
                )
                kept.append(item)
                continue

            fail_reasons: list[str] = []
            if isinstance(min_empathy, (int, float)) and empathy < float(min_empathy):
                fail_reasons.append(
                    f"empathy {empathy:.3f} < min_empathy {float(min_empathy):.3f}"
                )

            if isinstance(min_safety, (int, float)) and safety < float(min_safety):
                fail_reasons.append(
                    f"safety {safety:.3f} < min_safety {float(min_safety):.3f}"
                )

            if isinstance(bias_max, (int, float)) and bias > float(bias_max):
                fail_reasons.append(f"bias {bias:.3f} > bias_max {float(bias_max):.3f}")

            if requires_reasoning_metadata:
                reasoning_keys = ("chain_of_thought", "summary", "technique")
                if not any(
                    key in metadata and metadata.get(key) for key in reasoning_keys
                ):
                    fail_reasons.append("missing required reasoning metadata")

            if requires_voice_signature:
                if not metadata.get("voice_signature"):
                    fail_reasons.append("missing required voice_signature")
                if not metadata.get("persona_id"):
                    fail_reasons.append("missing required persona_id")

            if fail_reasons:
                removed_count += 1
                removed_by_stage[stage] = removed_by_stage.get(stage, 0) + 1
                stage_fail_reasons = fail_reasons_by_stage.setdefault(stage, {})
                for reason in fail_reasons:
                    stage_fail_reasons[reason] = stage_fail_reasons.get(reason, 0) + 1
                warning = (
                    f"Dropped sample from {stage} due to stage quality profile: "
                    + "; ".join(fail_reasons)
                )
                self.stats.warnings.append(warning)
                continue

            kept.append(item)

        self.stats.stage_policy_enforcement = {
            "removed_total": removed_count,
            "removed_by_stage": removed_by_stage,
            "crisis_overrides_by_stage": crisis_override_by_stage,
            "failure_reasons_by_stage": fail_reasons_by_stage,
        }

        return kept, removed_count

    def _save_dataset(self, data: list[dict]) -> str:
        """Save integrated dataset"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / self.config.output_filename

        # Convert to expected format
        output_data = {
            "conversations": data,
            "metadata": {
                "total_conversations": len(data),
                "sources": list(self.stats.samples_by_source.keys()),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "1.0",
                "stage_metrics": self.stats.stage_balance,
                "integration_stats": {
                    "samples_by_source": self.stats.samples_by_source,
                    "samples_by_stage": self.stats.samples_by_stage,
                    "warnings": self.stats.warnings,
                    "errors": self.stats.errors,
                },
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"💾 Saved dataset to {output_path}")
        return str(output_path)

    def _write_stage_outputs(self, stage_segments: dict[str, list[dict]]) -> None:
        """Persist per-stage datasets and manifest for downstream tracking."""
        stage_dir = Path("ai/training_data_consolidated/final")
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }

        for stage in self.config.stage_distribution:
            records = stage_segments.get(stage, [])
            stage_file = stage_dir / f"MASTER_{stage}.jsonl"
            with open(stage_file, "w") as stage_handle:
                for record in records:
                    stage_handle.write(json.dumps(record) + "\n")

            balance_stats = self.stats.stage_balance.get(stage, {})
            manifest["stages"][stage] = {
                "samples": len(records),
                "target": balance_stats.get("target"),
                "available": balance_stats.get("available"),
                "output_path": str(stage_file),
            }

        manifest_path = stage_dir / "MASTER_STAGE_MANIFEST.json"
        with open(manifest_path, "w") as manifest_handle:
            json.dump(manifest, manifest_handle, indent=2)

        logger.info(f"🗂️  Stage manifest updated at {manifest_path}")

    def _write_split_outputs(self, data: list[dict]) -> None:
        """Write aggregate and per-stage train/val/test split artifacts."""
        splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        split_root = Path("ai/training_data_consolidated/final/splits")
        split_root.mkdir(parents=True, exist_ok=True)

        aggregate_split = splitter.split(list(data), shuffle=True, seed=42)
        aggregate_counts = {
            "train": len(aggregate_split.train),
            "val": len(aggregate_split.val),
            "test": len(aggregate_split.test),
        }
        self._write_jsonl_split_files(
            split_root, aggregate_split.train, aggregate_split.val, aggregate_split.test
        )
        self.stats.split_counts["aggregate"] = aggregate_counts

        by_stage: dict[str, list[dict]] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            by_stage.setdefault(stage, []).append(item)

        for stage in self.config.stage_distribution:
            records = by_stage.get(stage, [])
            stage_dir = split_root / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_split = splitter.split(list(records), shuffle=True, seed=42)
            self._write_jsonl_split_files(
                stage_dir,
                stage_split.train,
                stage_split.val,
                stage_split.test,
            )
            self.stats.split_counts[stage] = {
                "train": len(stage_split.train),
                "val": len(stage_split.val),
                "test": len(stage_split.test),
            }

        logger.info(
            "🧪 Wrote aggregate and per-stage split artifacts to %s", split_root
        )

    @staticmethod
    def _write_jsonl_split_files(
        output_dir: Path,
        train_data: list[dict],
        val_data: list[dict],
        test_data: list[dict],
    ) -> None:
        """Write train/val/test JSONL files to output directory."""
        split_map = {
            "train.jsonl": train_data,
            "val.jsonl": val_data,
            "test.jsonl": test_data,
        }

        for filename, records in split_map.items():
            with open(output_dir / filename, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

    def _generate_report(self) -> dict:
        """Generate integration report"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples": self.stats.total_samples,
            "samples_by_source": self.stats.samples_by_source,
            "stage_distribution_targets": self.config.stage_distribution,
            "fail_on_missing_stage_artifacts": (
                self.config.fail_on_missing_stage_artifacts
            ),
            "stage_balance": self.stats.stage_balance,
            "actual_stage_percentages": {
                stage: count / self.stats.total_samples
                if self.stats.total_samples > 0
                else 0
                for stage, count in self.stats.samples_by_stage.items()
            },
            "split_counts": self.stats.split_counts,
            "integration_time_seconds": self.stats.integration_time,
            "warnings": self.stats.warnings,
            "errors": self.stats.errors,
            "bias_detection": self.stats.bias_detection_results,
            "stage_policy_enforcement": self.stats.stage_policy_enforcement,
            "stage_drift_waivers": self.stage_drift_waivers,
        }

    def _build_stage_health_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Build MTGC-11 integrated stage health report payload."""
        stage_balance = report.get("stage_balance", {})
        split_counts = report.get("split_counts", {})
        enforcement = report.get("stage_policy_enforcement", {})

        removed_by_stage = {}
        failure_reasons_by_stage = {}
        if isinstance(enforcement, dict):
            removed_by_stage = enforcement.get("removed_by_stage", {})
            failure_reasons_by_stage = enforcement.get("failure_reasons_by_stage", {})

        drift_failures: list[str] = []
        if isinstance(stage_balance, dict):
            for stage, metrics in stage_balance.items():
                if not isinstance(metrics, dict):
                    continue
                drift = metrics.get("drift_vs_target")
                if (
                    isinstance(drift, (int, float))
                    and abs(drift) > STAGE_DRIFT_TOLERANCE
                ):
                    drift_failures.append(stage)

        validator_status_by_stage: dict[str, Any] = {}
        for stage in self.config.stage_distribution:
            removed_count = 0
            if isinstance(removed_by_stage, dict):
                value = removed_by_stage.get(stage, 0)
                if isinstance(value, int):
                    removed_count = value

            reasons = {}
            if isinstance(failure_reasons_by_stage, dict):
                stage_reasons = failure_reasons_by_stage.get(stage, {})
                if isinstance(stage_reasons, dict):
                    reasons = stage_reasons

            validator_status_by_stage[stage] = {
                "passed": stage not in drift_failures and removed_count == 0,
                "removed_count": removed_count,
                "failure_reasons": reasons,
                "drift_within_tolerance": stage not in drift_failures,
            }

        blockers: list[str] = []
        if report.get("errors"):
            blockers.append("pipeline_errors_present")

        aggregate = (
            split_counts.get("aggregate") if isinstance(split_counts, dict) else None
        )
        if not isinstance(aggregate, dict):
            blockers.append("aggregate_splits_missing")
        else:
            train = aggregate.get("train", 0)
            val = aggregate.get("val", 0)
            test = aggregate.get("test", 0)
            if all(isinstance(v, int) for v in (train, val, test)):
                if train + val + test == 0:
                    blockers.append("aggregate_splits_empty")
            else:
                blockers.append("aggregate_split_counts_invalid")

        for stage in drift_failures:
            blockers.append(f"stage_drift_exceeds_tolerance:{stage}")

        for stage, status in validator_status_by_stage.items():
            if not isinstance(status, dict):
                continue
            if not status.get("passed", False):
                if status.get("removed_count", 0):
                    blockers.append(f"validator_failures:{stage}")

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": report.get("total_samples", 0),
            "integration_time_seconds": report.get("integration_time_seconds", 0.0),
            "stage_distribution_targets": report.get("stage_distribution_targets", {}),
            "stage_balance": stage_balance,
            "split_counts": split_counts,
            "validator_status_by_stage": validator_status_by_stage,
            "blockers": sorted(set(blockers)),
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
            "pass": len(blockers) == 0,
        }

    def _write_stage_health_report(self, stage_health_report: dict[str, Any]) -> None:
        """Write MTGC-11 integrated stage health report artifact."""
        output_path = Path(self.config.stage_health_report_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(stage_health_report, handle, indent=2)

        logger.info("📊 Integrated stage health report saved to %s", output_path)

    def _build_mtgc_closure_pack(
        self,
        report: dict[str, Any],
        stage_health_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build MTGC-13 closure pack summarizing status and evidence artifacts."""
        checklist_path = Path(self.config.tracker_sync_output_path)
        asana_mapping_path = Path(self.config.asana_task_key_mapping_output_path)
        asana_transition_path = Path(self.config.asana_task_transition_output_path)
        stage_health_path = Path(self.config.stage_health_report_output_path)

        ops_freshness_all_fresh = False
        if checklist_path.exists():
            try:
                with open(checklist_path, encoding="utf-8") as handle:
                    checklist_payload = json.load(handle)
                ops_payload = checklist_payload.get("ops_freshness", {})
                if isinstance(ops_payload, dict):
                    ops_freshness_all_fresh = bool(ops_payload.get("all_fresh", False))
            except Exception as exc:
                warning = f"Failed to read checklist for closure pack: {exc}"
                logger.warning(warning)
                self.stats.warnings.append(warning)

        stage_health_pass = bool(stage_health_report.get("pass", False))
        stage_blockers = stage_health_report.get("blockers", [])
        if not isinstance(stage_blockers, list):
            stage_blockers = []
        drift_ok = not any(
            str(blocker).startswith("stage_drift_exceeds_tolerance:")
            for blocker in stage_blockers
        )
        split_counts = report.get("split_counts", {})
        aggregate_split = (
            split_counts.get("aggregate") if isinstance(split_counts, dict) else None
        )
        split_artifacts_present = isinstance(aggregate_split, dict)

        success_criteria = {
            "stage_drift_within_tolerance": {
                "passed": bool(stage_health_pass) and bool(drift_ok),
                "evidence": "stage_health_report.blockers + report.stage_balance",
            },
            "manifest_and_report_generated": {
                "passed": (
                    Path(
                        "ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json"
                    ).exists()
                    and stage_health_path.exists()
                ),
                "evidence": (
                    "MASTER_STAGE_MANIFEST.json + integrated_stage_health_report.json"
                ),
            },
            "stage_3_4_inputs_checked": {
                "passed": any(
                    "missing required artifacts for stage3_edge_stress_test"
                    in str(item).lower()
                    or "missing required artifacts for stage4_voice_persona"
                    in str(item).lower()
                    for item in report.get("warnings", [])
                )
                or bool(self.config.fail_on_missing_stage_artifacts),
                "evidence": "pipeline warnings + strict artifact validation gate",
            },
            "aggregate_and_stage_split_artifacts_emitted": {
                "passed": split_artifacts_present,
                "evidence": "report.split_counts.aggregate + split directories",
            },
            "asana_task_graph_evidence_available": {
                "passed": asana_mapping_path.exists()
                or not self.config.enable_asana_sync,
                "evidence": "asana_task_key_mapping.json",
            },
            "ops_freshness_reflected": {
                "passed": ops_freshness_all_fresh,
                "evidence": "training_run_checklist.json:ops_freshness",
            },
            "asana_transition_results_recorded": {
                "passed": asana_transition_path.exists()
                or not self.config.enable_asana_sync,
                "evidence": "asana_task_transition_results.json",
            },
        }

        completion_pass = all(
            bool(entry.get("passed", False)) for entry in success_criteria.values()
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_key": "MTGC-13",
            "overall_pass": completion_pass,
            "success_criteria": success_criteria,
            "artifact_paths": {
                "checklist": str(checklist_path),
                "stage_health_report": str(stage_health_path),
                "stage_manifest": (
                    "ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json"
                ),
                "asana_task_key_mapping": str(asana_mapping_path),
                "asana_task_transition_results": str(asana_transition_path),
            },
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
        }

    def _write_mtgc_closure_pack(self, closure_pack: dict[str, Any]) -> None:
        """Write MTGC-13 closure pack artifact."""
        output_path = Path(self.config.closure_pack_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(closure_pack, handle, indent=2)

        logger.info("📦 MTGC closure pack saved to %s", output_path)


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
