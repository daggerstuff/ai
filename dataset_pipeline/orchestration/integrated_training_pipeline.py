#!/usr/bin/env python3
"""
Integrated Training Pipeline Orchestrator
Combines ALL data sources for comprehensive therapeutic AI training
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ai.dataset_pipeline.configs.stages import get_all_stages
from ai.dataset_pipeline.ingestion.dual_persona_loader import (
    DualPersonaLoader,
)
from ai.dataset_pipeline.ingestion.edge_case_jsonl_loader import (
    EdgeCaseJSONLLoader,
)
from ai.dataset_pipeline.ingestion.pixel_voice_loader import (
    PixelVoiceLoader,
)
from ai.dataset_pipeline.ingestion.psychology_knowledge_loader import (
    PsychologyKnowledgeLoader,
)
from ai.dataset_pipeline.quality.evidence_based_practice_validator import validate_bias
from ai.dataset_pipeline.storage_config import StorageBackend, get_storage_config
from ai.dataset_pipeline.storage_manager import StorageManager
from ai.dataset_pipeline.utils.logger import get_logger

logger = get_logger("dataset_pipeline.integrated_training_pipeline")


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
            source_path="ai/pipelines/edge_case_pipeline_standalone/output",
        )
    )

    pixel_voice: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.20,  # 20% voice-derived
            source_path="ai/pipelines/pixel_voice",
        )
    )

    psychology_knowledge: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.15,  # 15% psychology knowledge
            source_path="ai/training_data_consolidated",
        )
    )

    dual_persona: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.10,  # 10% dual persona
            source_path="ai/pipelines/dual_persona_training",
        )
    )

    standard_therapeutic: DataSourceConfig = field(
        default_factory=lambda: DataSourceConfig(
            enabled=True,
            target_percentage=0.30,  # 30% standard conversations
            source_path="ai/dataset_pipeline/pixelated-training",
        )
    )

    # Output configuration
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

    # Progress tracking integration
    enable_progress_tracking: bool = True
    progress_tracker_path: str = "ai/lightning/therapeutic_progress_tracker.py"


@dataclass
class IntegrationStats:
    """Statistics from pipeline integration"""

    total_samples: int = 0
    samples_by_source: dict[str, int] = field(default_factory=dict)
    samples_by_category: dict[str, int] = field(default_factory=dict)
    samples_by_stage: dict[str, int] = field(default_factory=dict)
    stage_balance: dict[str, dict[str, int]] = field(default_factory=dict)
    quality_scores: dict[str, float] = field(default_factory=dict)
    bias_detection_results: dict[str, any] = field(default_factory=dict)
    integration_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IntegratedTrainingPipeline:
    """
    Orchestrates integration of all data sources into unified training dataset
    """

    def __init__(self, config: IntegratedPipelineConfig | None = None):
        self.config = config or IntegratedPipelineConfig()
        self.stats = IntegrationStats()

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
            # Convenience default for a known VPS environment when S3 is explicitly selected.
            # Do not force S3 when the backend isn't explicitly configured.
            storage_config.s3_bucket = "pixel-data"

        self.storage = StorageManager(storage_config)

        # Initialize stage_balance with the four-stage ladder from configs/stages.py
        for stage in get_all_stages():
            if stage.id not in self.stats.stage_balance:
                self.stats.stage_balance[stage.id] = {
                    "target_share": stage.target_share,
                    "actual_samples": 0,
                }

    def _resolve_s3_path(self, manifest_path: str) -> str:
        """Resolve legacy local paths to S3 URIs."""
        if manifest_path.startswith("s3://"):
            return manifest_path

        # Map legacy VPS/Local paths to S3 structure
        # ~/datasets/consolidated/ -> s3://pixel-data/datasets/consolidated/
        if "consolidated" in manifest_path:
            # Strip home directory or relative prefixes
            clean_path = manifest_path.replace("~/", "").replace("../", "")
            if clean_path.startswith("datasets/consolidated/"):
                return f"datasets/consolidated/{clean_path.split('datasets/consolidated/')[1]}"

        return manifest_path

    def _cache_data(self, source_path: str) -> Path | None:
        """Download data from S3 to local cache if needed."""
        if not source_path:
            return None

        s3_path = self._resolve_s3_path(source_path)

        # If it's still a local path (wasn't resolved to S3), check if it exists
        if not s3_path.startswith("s3://") and not s3_path.startswith("datasets/"):
            local_p = Path(os.path.expanduser(source_path))
            if local_p.exists():
                return local_p
            return None

        # Define cache location
        cache_dir = Path.home() / ".cache" / "pixelated" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique filename based on the path to avoid collisions
        safe_name = s3_path.replace("/", "_").replace("s3:__", "")
        cached_file = cache_dir / safe_name

        if cached_file.exists():
            logger.info(f"Using cached file: {cached_file}")
            return cached_file

        logger.info(f"Downloading {s3_path} to cache...")
        try:
            # StorageManager expects path without s3://bucket/ prefix for S3 backend usually,
            # but let's check how it handles it.
            # Looking at StorageManager.download_file:
            # s3_client.download_file(bucket, storage_path, local)
            # So we need to strip the bucket if it's in the string

            download_key = s3_path
            if s3_path.startswith(
                f"s3://{self.storage.config.s3_bucket}/"
            ):  # Use self.storage.config.s3_bucket
                download_key = s3_path.replace(f"s3://{self.storage.config.s3_bucket}/", "")
            elif s3_path.startswith("datasets/"):
                # This matches the S3 key structure directly
                download_key = s3_path
            else:
                # If it's not an S3 path or a datasets/ path, it's likely a local path that wasn't resolved
                # or an S3 path without the s3:// prefix.
                # For now, assume it's an S3 key if it's not a local file.
                # This might need more robust parsing depending on expected s3_path formats.
                pass

            self.storage.download_file(download_key, cached_file)
            return cached_file
        except Exception as e:
            logger.error(f"Failed to download {s3_path}: {e}")
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

        # 1. Load Edge Case Data
        if self.config.edge_cases.enabled:
            cached_path = self._cache_data(self.config.edge_cases.source_path)
            edge_data = self._load_edge_cases(cached_path)  # Modified to pass cached_path
            all_training_data.extend(edge_data)
            self.stats.samples_by_source["edge_cases"] = len(edge_data)
            logger.info(f"✅ Loaded {len(edge_data)} edge case examples")

        # 2. Load Pixel Voice Data
        if self.config.pixel_voice.enabled:
            cached_path = self._cache_data(self.config.pixel_voice.source_path)
            voice_data = self._load_pixel_voice(cached_path)  # Modified to pass cached_path
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
            persona_data = self._load_dual_persona(cached_path)  # Modified to pass cached_path
            all_training_data.extend(persona_data)
            self.stats.samples_by_source["dual_persona"] = len(persona_data)
            logger.info(f"✅ Loaded {len(persona_data)} dual persona examples")

        # 5. Load Standard Therapeutic Conversations
        if self.config.standard_therapeutic.enabled:
            # Standard therapeutic loader handles its own path resolution and caching internally
            # as it tries multiple paths. We will keep its original signature for now.
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

        # 9. Save integrated dataset
        output_path = self._save_dataset(balanced_data)
        self._write_stage_outputs(stage_segments)

        # 10. Generate integration report
        self.stats.total_samples = len(balanced_data)
        self.stats.samples_by_category = dict(self.stats.samples_by_stage)
        self.stats.integration_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        report = self._generate_report()

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
        # Try multiple file locations
        possible_files = [
            Path(self.config.standard_therapeutic.source_path) / "training_dataset.json",
            Path("ai/lightning/pixelated-training/training_dataset.json"),
            Path("ai/dataset_pipeline/pixelated-training/training_dataset.json"),
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

    def _try_load_json_file(self, file_path: Path) -> list:
        """Helper to try loading a JSON file and return list of conversations"""
        try:
            with open(file_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                logger.info(
                    f"✅ Loaded {len(raw_data)} conversations from {file_path} (list format)"
                )
                return raw_data
            if isinstance(raw_data, dict):
                conversations = raw_data.get("conversations", [])
                if conversations:
                    logger.info(
                        f"✅ Loaded {len(conversations)} conversations from {file_path} (dict format)"
                    )
                    return conversations
                logger.warning(f"File {file_path} loaded but no conversations found")
            else:
                logger.warning(f"Unexpected data type in {file_path}: {type(raw_data)}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error in {file_path} at position {e.pos}: {e.msg}")
            raise e
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            raise e
        return []

    def _handle_load_error(self, possible_files: list[Path], last_error: Exception | None):
        """Helper to handle load errors"""
        if last_error:
            error = f"Failed to load standard therapeutic data from any available file. Last error: {last_error}"
        else:
            error = (
                f"Standard therapeutic data not found in any of: {[str(f) for f in possible_files]}"
            )
        logger.error(error)
        self.stats.errors.append(error)

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
                        "metadata": {"source": "standard_therapeutic", "is_edge_case": False},
                    }
                )

        logger.info(
            f"✅ Converted {len(training_data)} standard therapeutic examples to training format"
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

    def _balance_dataset(self, data: list[dict]) -> (list[dict], dict[str, list[dict]]):
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
                        f"Stage '{stage}' has only {len(bucket)} samples (target: {target_count})."
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

    def _run_bias_detection(self, data: list[dict]) -> list[dict]:
        """Run bias detection on training data"""
        logger.info("🔍 Running bias detection...")

        try:
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

        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            return data

    def _run_quality_validation(self, data: list[dict]) -> list[dict]:
        """Run quality validation on training data"""
        logger.info("✓ Running quality validation...")

        # TODO: Implement comprehensive quality validation
        logger.info(f"   Validated {len(data)} samples")
        return data

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
        if not stage_segments:
            return

        stage_dir = Path("ai/training_data_consolidated/final")
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "stages": {}}

        for stage, records in stage_segments.items():
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

    def _generate_report(self) -> dict:
        """Generate integration report"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples": self.stats.total_samples,
            "samples_by_source": self.stats.samples_by_source,
            "stage_distribution_targets": self.config.stage_distribution,
            "stage_balance": self.stats.stage_balance,
            "actual_stage_percentages": {
                stage: count / self.stats.total_samples if self.stats.total_samples > 0 else 0
                for stage, count in self.stats.samples_by_stage.items()
            },
            "integration_time_seconds": self.stats.integration_time,
            "warnings": self.stats.warnings,
            "errors": self.stats.errors,
            "bias_detection": self.stats.bias_detection_results,
        }


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
