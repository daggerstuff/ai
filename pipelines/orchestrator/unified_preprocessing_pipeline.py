"""
Unified Preprocessing Pipeline for Pixelated Empathy AI Training

This module orchestrates the integration of all data sources into a unified training dataset:
- ULTIMATE_FINAL_DATASET.jsonl (2.6GB, 608,497 conversations)
- Psychology knowledge base (4,867 concepts)
- YouTube transcripts from expert creators
- Crisis intervention scenarios
- Therapeutic counseling conversations
- Medical consultation dialogues
"""

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canonical dataset locations (S3-first)
from ai.common.dataset_registry import iter_dataset_refs, load_registry
from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root
from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CRISIS_ESCALATION_LEVELS = {"high", "very_high"}


@dataclass
class DataSource:
    """Represents a data source with metadata"""

    name: str
    path: str
    format: str
    size_bytes: int
    record_count: int | None = None
    quality_score: float | None = None
    source_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None


@dataclass
class StagePolicy:
    """Defines guard-rail expectations per training stage."""

    name: str
    min_empathy: float
    min_safety: float
    allow_crisis_override: bool = False
    requires_voice_signature: bool = False
    dedup_priority: int = 1


def get_default_stage_policies() -> dict[str, StagePolicy]:
    """Return baseline policy definitions for all stages."""
    return {
        "stage1_foundation": StagePolicy(
            name="stage1_foundation", min_empathy=0.55, min_safety=0.7, dedup_priority=1
        ),
        "stage2_therapeutic_expertise": StagePolicy(
            name="stage2_therapeutic_expertise",
            min_empathy=0.5,
            min_safety=0.68,
            dedup_priority=2,
        ),
        "stage3_edge_stress_test": StagePolicy(
            name="stage3_edge_stress_test",
            min_empathy=0.35,
            min_safety=0.55,
            allow_crisis_override=True,
            dedup_priority=3,
        ),
        "stage4_voice_persona": StagePolicy(
            name="stage4_voice_persona",
            min_empathy=0.6,
            min_safety=0.75,
            requires_voice_signature=True,
            dedup_priority=4,
        ),
    }


class StageCatalog:
    """Loads manifest metadata and infers stage assignments."""

    def __init__(
        self,
        registry_path: Path = Path("ai/data/dataset_registry.json"),
        manifest_path: Path | None = Path("ai/data/training_policy_manifest.json"),
    ):
        self.registry_path = registry_path
        # Backwards compatibility: older docs/code referenced master_dataset_manifest.json
        if manifest_path is not None and not manifest_path.exists():
            legacy = Path("ai/data/master_dataset_manifest.json")
            self.manifest_path = legacy if legacy.exists() else manifest_path
        else:
            self.manifest_path = manifest_path
        self.stage_map: dict[str, str] = {}
        self.stage_priorities: dict[str, int] = {
            "stage1_foundation": 1,
            "stage2_therapeutic_expertise": 2,
            "stage3_edge_stress_test": 3,
            "stage4_voice_persona": 4,
        }
        self._load_registry()
        self._load_manifest()

    def _load_registry(self) -> None:
        """
        Load stage assignments from the S3-first dataset registry.

        This is the authoritative source for dataset stage metadata.
        """
        if not self.registry_path.exists():
            return

        try:
            registry = load_registry(self.registry_path)
        except Exception as exc:
            logger.warning(f"Unable to load dataset registry for stage catalog: {exc}")
            return

        for dataset_ref in iter_dataset_refs(registry):
            if not dataset_ref.stage:
                continue
            stage = dataset_ref.stage
            self.stage_map[dataset_ref.key.lower()] = stage
            self.stage_map[dataset_ref.s3_path.lower()] = stage
            self.stage_map[Path(dataset_ref.s3_path).stem.lower()] = stage
            for fallback in dataset_ref.fallback_paths.values():
                self.stage_map[fallback.lower()] = stage
                self.stage_map[Path(fallback).stem.lower()] = stage

    def _load_manifest(self) -> None:
        # `master_dataset_manifest.json` is policy + historical reference. Stage mapping
        # should come primarily from the dataset registry.
        if self.manifest_path is None or not self.manifest_path.exists():
            return

        try:
            with open(self.manifest_path) as manifest_file:
                data = json.load(manifest_file)
        except Exception as exc:
            logger.warning(f"Unable to load manifest for stage catalog: {exc}")
            return

        datasets = data.get("datasets", {})
        for section in datasets.values():
            if not isinstance(section, dict):
                continue
            for key, dataset in section.items():
                if not isinstance(dataset, dict):
                    continue
                stage = dataset.get("stage")
                if not stage:
                    continue
                self.stage_map[key.lower()] = stage
                for field_name in ("path", "gdrive_path"):
                    if raw_path := dataset.get(field_name):
                        stem = Path(str(raw_path)).stem.lower()
                        self.stage_map[stem] = stage

    def lookup(self, source_name: str, path: str | None, source_type: str | None) -> str:
        """Return the best stage label for the provided source metadata."""
        candidates = [source_name or "", source_type or ""]
        if path:
            candidates.extend([path, Path(path).stem])

        for candidate in candidates:
            if not candidate:
                continue
            if stage := self.stage_map.get(candidate.lower()):
                return stage

        return self._infer_fallback(source_name or "", source_type or "")

    def _infer_fallback(self, source_name: str, source_type: str) -> str:
        """Heuristic stage inference when manifest metadata is missing."""
        text = f"{source_name} {source_type}".lower()
        if any(
            token in text for token in ["edge_case", "reddit", "suicide", "kaggle_tf", "nightmare"]
        ):
            return "stage3_edge_stress_test"
        if any(token in text for token in ["voice", "tim_fletcher", "persona", "transcript"]):
            return "stage4_voice_persona"
        if any(token in text for token in ["cot", "reasoning", "memo", "knowledge"]):
            return "stage2_therapeutic_expertise"
        return "stage1_foundation"

    def get_priority(self, stage: str) -> int:
        return self.stage_priorities.get(stage, 1)


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""

    target_quality_threshold: float = 0.8
    deduplication_enabled: bool = True
    validation_enabled: bool = True
    safety_filtering_enabled: bool = True
    psychology_integration_enabled: bool = True
    youtube_rag_integration_enabled: bool = True
    crisis_scenario_weight: float = 1.5
    therapeutic_conversation_weight: float = 1.0
    knowledge_base_weight: float = 1.2
    stage_policy_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class UnifiedPreprocessingPipeline:
    """Main preprocessing pipeline orchestrator"""

    PROGRESS_LOG_INTERVAL = 10000

    def __init__(self, config: ProcessingConfig | None = None):
        self.config = config or ProcessingConfig()
        self.data_sources: list[DataSource] = []
        self.processed_records = 0
        self.quality_filtered_records = 0
        self.safety_filtered_records = 0
        self.final_dataset_path = None
        self.stage_catalog = StageCatalog()
        self.stage_policies = self._build_stage_policies()
        self._s3_loader: S3DatasetLoader | None = None
        self._pii_patterns = [
            re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
            re.compile(r"\b\(?\d{3}\)?\s*\d{3}[-.\s]?\d{4}\b"),
            re.compile(r"\b\d{9}\b"),
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ]

    def _is_crisis_override_active(self, metadata: dict[str, Any], policy: StagePolicy) -> bool:
        crisis_flag = metadata.get("crisis_intensity")
        if not crisis_flag or not policy.allow_crisis_override:
            return False
        if isinstance(crisis_flag, str):
            crisis_flag = crisis_flag.lower()
        return crisis_flag in CRISIS_ESCALATION_LEVELS

    def _build_stage_policies(self) -> dict[str, StagePolicy]:
        policies = get_default_stage_policies()
        for stage, overrides in self.config.stage_policy_overrides.items():
            base_policy = policies.get(stage)
            if base_policy:
                merged = {**base_policy.__dict__, **overrides}
            else:
                merged = {"name": stage, **overrides}
            policies[stage] = StagePolicy(**merged)
        return policies

    def get_stage_policy(self, stage: str) -> StagePolicy:
        return self.stage_policies.get(stage, self.stage_policies["stage1_foundation"])

    def register_data_source(self, source: DataSource):
        """Register a data source for processing"""
        if not source.stage:
            source.stage = self.stage_catalog.lookup(source.name, source.path, source.source_type)
        source.metadata.setdefault("stage", source.stage)
        self.data_sources.append(source)
        logger.info(f"Registered data source: {source.name} ({source.format})")

    def discover_data_sources(self):
        """Discover and register all available data sources"""
        self._discover_registry_sources()
        self._discover_local_sources()
        logger.info(f"Discovered {len(self.data_sources)} data sources")

    def _discover_registry_sources(self) -> None:
        """Discover and register datasets from the canonical S3-first registry."""
        try:
            registry = load_registry()
            for dataset_ref in iter_dataset_refs(registry):
                resolved_s3_path = self._resolve_registry_s3_path(
                    dataset_ref.s3_path,
                    getattr(dataset_ref, "legacy_paths", []),
                )
                if dataset_ref.s3_path.endswith("/"):
                    for s3_path in self._list_s3_objects(dataset_ref.s3_path):
                        suffix = Path(s3_path).suffix.lower().lstrip(".")
                        self.register_data_source(
                            DataSource(
                                name=f"{dataset_ref.key}:{Path(s3_path).name}",
                                path=s3_path,
                                format=suffix,
                                size_bytes=0,
                                source_type=dataset_ref.type or "registry",
                                stage=dataset_ref.stage,
                                metadata={
                                    "quality_profile": dataset_ref.quality_profile,
                                    "focus": dataset_ref.focus,
                                    "registry_parent": dataset_ref.key,
                                },
                            )
                        )
                    continue

                suffix = Path(resolved_s3_path).suffix.lower()
                if suffix not in [".json", ".jsonl"]:
                    continue

                self.register_data_source(
                    DataSource(
                        name=dataset_ref.key,
                        path=resolved_s3_path,
                        format=suffix.lstrip("."),
                        size_bytes=0,
                        source_type=dataset_ref.type or "registry",
                        stage=dataset_ref.stage,
                        metadata={
                            "quality_profile": dataset_ref.quality_profile,
                            "focus": dataset_ref.focus,
                        },
                    )
                )
        except Exception as exc:
            logger.warning(f"Skipping S3 registry discovery (will fallback to local): {exc}")

    def _discover_local_sources(self) -> None:
        """Discover and register local datasets bundled in the repo/worktree."""
        self._discover_local_main_datasets()
        self._discover_local_psychology_knowledge()
        self._discover_local_youtube_transcripts()
        self._discover_local_main_datasets()
        self._discover_local_psychology_knowledge()
        self._discover_local_youtube_transcripts()
        self._discover_local_conversations()
        self._discover_local_generated_synthetic()
        self._discover_local_nightmares()

    def _discover_local_main_datasets(self) -> None:
        datasets_dir = Path("ai/training_data_consolidated/final_datasets")
        if not datasets_dir.exists():
            return

        for file_path in datasets_dir.glob("*.*"):
            if file_path.suffix not in [".jsonl", ".json"]:
                continue
            self.register_data_source(
                DataSource(
                    name=file_path.stem,
                    path=str(file_path),
                    format=file_path.suffix.lstrip("."),
                    size_bytes=file_path.stat().st_size,
                    source_type="training_dataset",
                )
            )

    def _discover_local_psychology_knowledge(self) -> None:
        psych_dir = Path("ai/training_data_consolidated/psychology_knowledge")
        if not psych_dir.exists():
            return

        for file_path in psych_dir.glob("*.json"):
            self.register_data_source(
                DataSource(
                    name=f"psychology_{file_path.stem}",
                    path=str(file_path),
                    format="json",
                    size_bytes=file_path.stat().st_size,
                    source_type="knowledge_base",
                )
            )

    def _discover_local_youtube_transcripts(self) -> None:
        transcripts_dir = Path("ai/training_data_consolidated/transcripts")
        if not transcripts_dir.exists():
            return

        if not (transcript_files := list(transcripts_dir.glob("*.md"))):
            return

        self.register_data_source(
            DataSource(
                name="youtube_transcripts",
                path=str(transcripts_dir),
                format="markdown",
                size_bytes=sum(f.stat().st_size for f in transcript_files),
                record_count=len(transcript_files),
                source_type="youtube_transcripts",
            )
        )

    def _discover_local_conversations(self) -> None:
        conversations_dir = Path("ai/training_data_consolidated/conversations")
        if not conversations_dir.exists():
            return

        for file_path in conversations_dir.glob("*.jsonl"):
            self.register_data_source(
                DataSource(
                    name=f"conversations_{file_path.stem}",
                    path=str(file_path),
                    format="jsonl",
                    size_bytes=file_path.stat().st_size,
                    source_type="conversations",
                )
            )

    def _discover_local_generated_synthetic(self) -> None:
        """Discover locally generated synthetic datasets (NeMo, Edge Cases)."""
        generated_dir = Path("ai/training_ready/data/generated")
        if not generated_dir.exists():
            return

        # Edge Case Synthetic
        edge_case_path = generated_dir / "edge_case_synthetic.jsonl"
        if edge_case_path.exists():
            self.register_data_source(
                DataSource(
                    name="edge_case_synthetic",
                    path=str(edge_case_path),
                    format="jsonl",
                    size_bytes=edge_case_path.stat().st_size,
                    source_type="synthetic_edge_cases",
                    stage="stage3_edge_stress_test",
                    metadata={"synthetic_source": "template_generator"},
                )
            )

        # NeMo Synthetic
        nemo_path = generated_dir / "nemo_synthetic" / "nemo_synthetic_dataset.jsonl"
        if nemo_path.exists():
            self.register_data_source(
                DataSource(
                    name="nemo_synthetic_therapeutic",
                    path=str(nemo_path),
                    format="jsonl",
                    size_bytes=nemo_path.stat().st_size,
                    source_type="synthetic_nemo",
                    stage="stage2_therapeutic_expertise",  # High quality synthetic
                    metadata={"synthetic_source": "nemo_data_designer"},
                )
            )

    def _discover_local_nightmares(self) -> None:
        """Discover generated Ultra Nightmare datasets."""
        nightmares_dir = Path("ai/training_ready/data/generated/ultra_nightmares")
        if not nightmares_dir.exists():
            return

        for file_path in nightmares_dir.glob("*.jsonl"):
            self.register_data_source(
                DataSource(
                    name=f"ultra_nightmare_{file_path.stem}",
                    path=str(file_path),
                    format="jsonl",
                    size_bytes=file_path.stat().st_size,
                    source_type="synthetic_nightmares",
                    stage="stage3_edge_stress_test",
                    metadata={"synthetic_source": "ultra_nightmare_generator"},
                )
            )

    def _get_s3_loader(self) -> S3DatasetLoader | None:
        if self._s3_loader is not None:
            return self._s3_loader
        try:
            self._s3_loader = S3DatasetLoader()
        except Exception as exc:
            logger.warning(f"S3 loader unavailable: {exc}")
            self._s3_loader = None
        return self._s3_loader

    def _list_s3_objects(self, s3_prefix: str) -> list[str]:
        """
        List concrete dataset files under an S3 prefix.

        Some registry entries point to a directory/prefix (ending in `/`) when a
        dataset is stored as a folder containing multiple files. This helper
        expands that prefix into concrete .json/.jsonl objects for ingestion.
        """

        loader = self._get_s3_loader()
        if loader is None:
            return []

        bucket, key_prefix = loader._parse_s3_path(s3_prefix)
        if key_prefix and not key_prefix.endswith("/"):
            key_prefix = f"{key_prefix}/"

        results: list[str] = []
        paginator = loader.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not isinstance(key, str):
                    continue
                if key.endswith((".json", ".jsonl")):
                    results.append(f"s3://{bucket}/{key}")

        return results

    def _resolve_registry_s3_path(self, s3_path: str, legacy_paths: list[str]) -> str:
        """
        Prefer the canonical registry path, but fall back to legacy S3 paths if
        the canonical object doesn't exist yet (e.g., before a rename/copy).
        """

        loader = self._get_s3_loader()
        if loader is None:
            return s3_path

        if loader.object_exists(s3_path):
            return s3_path

        resolved = next(
            (
                legacy
                for legacy in legacy_paths
                if legacy and legacy.startswith("s3://") and loader.object_exists(legacy)
            ),
            None,
        )
        return resolved or s3_path

    def validate_data_source(self, source: DataSource) -> bool:
        """Validate a data source"""
        try:
            if source.path.startswith("s3://"):
                loader = self._get_s3_loader()
                if loader is None:
                    logger.warning("Cannot validate S3 path without S3 credentials/loader")
                    return False
                if not loader.object_exists(source.path):
                    logger.warning(f"S3 object does not exist: {source.path}")
                    return False
                return True

            if not os.path.exists(source.path):
                logger.warning(f"Data source path does not exist: {source.path}")
                return False

            if source.format == "jsonl":
                # Check if it's a valid JSONL file
                with open(source.path) as f:
                    if line := f.readline():
                        json.loads(line)
            elif source.format == "json":
                # Check if it's a valid JSON file
                with open(source.path) as f:
                    json.load(f)

            return True
        except Exception as e:
            logger.error(f"Validation failed for {source.name}: {e!s}")
            return False

    def process_dataset(self, source: DataSource) -> list[dict[str, Any]]:
        """Process a single dataset"""
        logger.info(f"Processing dataset: {source.name}")

        try:
            if source.format == "jsonl":
                return self._process_jsonl(source)
            if source.format == "json":
                return self._process_json(source)
            logger.warning(f"Unsupported format '{source.format}' for {source.name}")
            return []
        except Exception as e:
            logger.error(f"Error processing {source.name}: {e!s}")
            return []

    def _process_jsonl(self, source: DataSource) -> list[dict[str, Any]]:
        """Process a JSONL format dataset"""
        records = []
        processed_count = 0

        if source.path.startswith("s3://"):
            loader = self._get_s3_loader()
            if loader is None:
                logger.error("S3 loader unavailable, cannot process S3 JSONL source")
                return []

            for record in loader.stream_jsonl(source.path):
                if processed_record := self._process_single_record(record, source):
                    records.append(processed_record)
                    processed_count += 1
                    if processed_count % self.PROGRESS_LOG_INTERVAL == 0:
                        logger.info(f"Processed {processed_count} records from {source.name}")
        else:
            with open(source.path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        if processed_record := self._process_single_record(record, source):
                            records.append(processed_record)
                            processed_count += 1
                            if processed_count % self.PROGRESS_LOG_INTERVAL == 0:
                                logger.info(
                                    f"Processed {processed_count} records from {source.name}"
                                )
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} of {source.name}: {e!s}")

        logger.info(f"Completed processing {source.name}: {len(records)} valid records")
        return records

    def _process_json(self, source: DataSource) -> list[dict[str, Any]]:
        """Process a JSON format dataset"""
        if source.path.startswith("s3://"):
            loader = self._get_s3_loader()
            if loader is None:
                logger.error("S3 loader unavailable, cannot process S3 JSON source")
                return []
            data = loader.load_json(source.path)
        else:
            with open(source.path) as f:
                data = json.load(f)

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "conversations" in data:
            items = data["conversations"]

        records = []
        for item in items:
            if processed_record := self._process_single_record(item, source):
                records.append(processed_record)

        logger.info(f"Completed processing {source.name}: {len(records)} valid records")
        return records

    def _process_single_record(
        self, record: dict[str, Any], source: DataSource
    ) -> dict[str, Any] | None:
        """Process a single record: enhance and validate"""
        enhanced = self.enhance_record(record, source)
        return enhanced if self.validate_record(enhanced) else None

    def enhance_record(self, record: dict[str, Any], source: DataSource) -> dict[str, Any]:
        """Enhance a record with metadata and source information"""
        # Add source tracking
        record["_source"] = source.name
        record["_source_type"] = source.source_type
        record["_processed_at"] = datetime.now(timezone.utc).isoformat()

        # Add quality scoring if not present
        if "metadata" not in record:
            record["metadata"] = {}

        # Try Quality Scoring v1 first, fallback to estimate
        if "quality_score" not in record.get("metadata", {}):
            try:
                from ai.pipelines.orchestrator.quality.quality_scoring_v1 import (
                    QualityScoringV1,
                )

                if not hasattr(self, "_quality_scoring"):
                    self._quality_scoring = QualityScoringV1(enabled=True)

                # Extract text for scoring
                text = ""
                if "text" in record:
                    text = record["text"]
                elif "messages" in record:
                    text = " ".join(
                        msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        for msg in record["messages"]
                    )

                if text:
                    scoring_result = self._quality_scoring.score_conversation_text(text)
                    record["metadata"]["quality_score"] = scoring_result.get("composite", 0.5)
                    record["metadata"]["quality_scoring_v1"] = {
                        "signals": scoring_result.get("signals", {}),
                        "decision": scoring_result.get("decision", "curate"),
                    }
                else:
                    record["metadata"]["quality_score"] = self.estimate_quality_score(record)

            except ImportError:
                # Fallback to estimate if Quality Scoring v1 not available
                record["metadata"]["quality_score"] = self.estimate_quality_score(record)

        # Ensure stage metadata is populated
        self.resolve_stage_for_record(record, source)

        return record

    def resolve_stage_for_record(self, record: dict[str, Any], source: DataSource) -> str:
        """Populate and return the stage associated with this record."""
        metadata = record.setdefault("metadata", {})
        stage = (
            metadata.get("stage")
            or source.metadata.get("stage")
            or source.stage
            or self.stage_catalog.lookup(source.name, source.path, source.source_type)
        )
        metadata["stage"] = stage
        return stage

    def estimate_quality_score(self, record: dict[str, Any]) -> float:
        """Estimate quality score for a record"""
        score = 0.5  # Base score

        # Check for content length
        content_length = 0
        if "text" in record:
            content_length = len(record["text"])
        elif "messages" in record:
            for msg in record["messages"]:
                if "content" in msg:
                    content_length += len(msg["content"])

        if content_length > 100:
            score += 0.2
        if content_length > 500:
            score += 0.1

        # Check for proper structure
        if "messages" in record and len(record["messages"]) >= 2:
            score += 0.2

        # Check for metadata
        if "metadata" in record:
            score += 0.1

        return min(score, 1.0)

    def validate_record(self, record: dict[str, Any]) -> bool:
        """Validate a single record"""
        if not record:
            return False

        metadata = record.get("metadata", {})
        stage = metadata.get("stage", "stage1_foundation")
        policy = self.get_stage_policy(stage)

        # Basic validation
        if self.config.validation_enabled and (
            ("messages" not in record and "text" not in record) or self._content_length(record) < 10
        ):
            return False

        # Quality filtering
        if self.config.target_quality_threshold > 0:
            quality_score = record.get("metadata", {}).get("quality_score", 0.5)
            if quality_score < self.config.target_quality_threshold:
                return False

        empathy_score = metadata.get("empathy_score", 0.5)
        safety_score = metadata.get("safety_score", 0.7)
        crisis_override_active = self._is_crisis_override_active(metadata, policy)
        if (
            empathy_score < policy.min_empathy or safety_score < policy.min_safety
        ) and not crisis_override_active:
            return False

        return bool(not policy.requires_voice_signature or metadata.get("voice_signature"))

    def _content_length(self, record: dict[str, Any]) -> int:
        """Compute a minimal content length for validation."""
        text = record.get("text")
        if isinstance(text, str):
            return len(text)
        messages = record.get("messages")
        if isinstance(messages, list):
            return sum(len(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
        return 0

    def deduplicate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate records"""
        if not self.config.deduplication_enabled:
            return records

        hash_map: OrderedDict[str, dict[str, Any]] = OrderedDict()
        duplicates_removed = 0
        replacements = 0

        for record in records:
            # Create content hash for deduplication
            content_to_hash = ""
            if "text" in record:
                content_to_hash = record["text"]
            elif "messages" in record:
                content_to_hash = "".join([msg.get("content", "") for msg in record["messages"]])

            if content_to_hash:
                content_hash = hashlib.md5(content_to_hash.encode()).hexdigest()
                stage = record.get("metadata", {}).get("stage", "stage1_foundation")
                priority = self.stage_catalog.get_priority(stage)
                existing = hash_map.get(content_hash)

                if existing is None:
                    hash_map[content_hash] = {"record": record, "priority": priority}
                elif priority > existing["priority"]:
                    hash_map[content_hash] = {"record": record, "priority": priority}
                    replacements += 1
                else:
                    duplicates_removed += 1

        logger.info(
            "Removed %s duplicate records (%s higher-priority replacements)",
            duplicates_removed,
            replacements,
        )
        return [entry["record"] for entry in hash_map.values()]

    def apply_safety_filtering(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply safety filtering to records"""
        if not self.config.safety_filtering_enabled:
            return records

        safe_records = []
        unsafe_filtered = 0

        for record in records:
            metadata = record.get("metadata", {})
            stage = metadata.get("stage", "stage1_foundation")
            policy = self.get_stage_policy(stage)

            content = self._collect_record_content(record)
            safety_score = metadata.get("safety_score", 0.7)

            if self._contains_pii(content):
                unsafe_filtered += 1
                continue

            crisis_override_active = self._is_crisis_override_active(metadata, policy)

            if policy.allow_crisis_override:
                # Drop non-crisis records that fail safety thresholds even in lenient mode
                if safety_score < policy.min_safety and not crisis_override_active:
                    unsafe_filtered += 1
                    continue
            elif safety_score < policy.min_safety or self._contains_disallowed_keywords(content):
                unsafe_filtered += 1
                continue

            safe_records.append(record)

        self.safety_filtered_records += unsafe_filtered
        logger.info(f"Filtered {unsafe_filtered} unsafe records (stage-aware)")
        return safe_records

    def _collect_record_content(self, record: dict[str, Any]) -> str:
        if "text" in record:
            return str(record["text"]).lower()
        if "messages" in record:
            return " ".join([msg.get("content", "") for msg in record["messages"]]).lower()
        return ""

    def _contains_disallowed_keywords(self, content: str) -> bool:
        unsafe_keywords = ["explicit", "nsfw", "inappropriate"]
        return any(keyword in content for keyword in unsafe_keywords)

    def _contains_pii(self, content: str) -> bool:
        if not content:
            return False
        return any(pattern.search(content) for pattern in self._pii_patterns)

    def integrate_psychology_knowledge(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Integrate psychology knowledge base concepts into records"""
        if not self.config.psychology_integration_enabled:
            return records

        # Load psychology knowledge base
        psych_knowledge = {}
        psych_dir = Path("ai/training_data_consolidated/psychology_knowledge")
        if psych_dir.exists():
            for file_path in psych_dir.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            psych_knowledge |= data
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "concept_id" in item:
                                    psych_knowledge[item["concept_id"]] = item
                except Exception as e:
                    logger.warning(f"Failed to load psychology knowledge from {file_path}: {e!s}")

        if not psych_knowledge:
            logger.warning("No psychology knowledge base found for integration")
            return records

        # Apply psychology concepts to records
        enhanced_records = []
        for record in records:
            # Add psychology metadata
            if "metadata" not in record:
                record["metadata"] = {}

            record["metadata"]["psychology_concepts"] = self.extract_psychology_concepts(
                record, psych_knowledge
            )
            enhanced_records.append(record)

        logger.info(f"Integrated psychology knowledge into {len(enhanced_records)} records")
        return enhanced_records

    def extract_psychology_concepts(
        self, record: dict[str, Any], psych_knowledge: dict[str, Any]
    ) -> list[str]:
        """Extract relevant psychology concepts from a record"""
        concepts = []
        content = ""

        if "text" in record:
            content = record["text"].lower()
        elif "messages" in record:
            content = " ".join([msg.get("content", "").lower() for msg in record["messages"]])

        # Simple keyword matching for psychology concepts
        for concept_id, concept_data in psych_knowledge.items():
            concept_terms = []
            if isinstance(concept_data, dict):
                concept_terms = [
                    concept_data.get("category", ""),
                    concept_data.get("content", ""),
                ]
            elif isinstance(concept_data, str):
                concept_terms = [concept_data]

            for term in concept_terms:
                if term and term.lower() in content:
                    concepts.append(concept_id)
                    break

        return list(set(concepts))[:10]  # Limit to top 10 concepts

    def execute_pipeline(self) -> str:
        """Execute the complete preprocessing pipeline"""
        logger.info("Starting unified preprocessing pipeline execution")

        # Discover data sources
        self.discover_data_sources()

        if not self.data_sources:
            raise ValueError("No data sources found for processing")

        # Process all data sources
        all_records = []
        for source in self.data_sources:
            if self.validate_data_source(source):
                records = self.process_dataset(source)
                all_records.extend(records)
                self.processed_records += len(records)
                logger.info(f"Added {len(records)} records from {source.name}")
            else:
                logger.warning(f"Skipping invalid data source: {source.name}")

        logger.info(f"Total records processed: {len(all_records)}")

        # Apply preprocessing steps
        if self.config.deduplication_enabled:
            all_records = self.deduplicate_records(all_records)

        if self.config.safety_filtering_enabled:
            all_records = self.apply_safety_filtering(all_records)

        if self.config.psychology_integration_enabled:
            all_records = self.integrate_psychology_knowledge(all_records)

        # Final validation
        final_records = [record for record in all_records if self.validate_record(record)]

        logger.info(f"Final dataset contains {len(final_records)} records")

        # Save final dataset
        output_dir = get_dataset_pipeline_output_root() / "final_output"
        output_dir.mkdir(exist_ok=True)

        final_dataset_path = (
            output_dir
            / f"unified_training_dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        with open(final_dataset_path, "w") as f:
            for record in final_records:
                f.write(json.dumps(record) + "\n")

        self.final_dataset_path = str(final_dataset_path)

        # Generate summary report
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sources_processed": len(self.data_sources),
            "total_records_processed": self.processed_records,
            "final_record_count": len(final_records),
            "deduplication_enabled": self.config.deduplication_enabled,
            "safety_filtering_enabled": self.config.safety_filtering_enabled,
            "psychology_integration_enabled": self.config.psychology_integration_enabled,
            "final_dataset_path": self.final_dataset_path,
            "final_dataset_size_bytes": final_dataset_path.stat().st_size
            if final_dataset_path.exists()
            else 0,
        }

        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Pipeline execution completed. Final dataset saved to: {self.final_dataset_path}"
        )
        return self.final_dataset_path


# Convenience functions
def create_default_pipeline() -> UnifiedPreprocessingPipeline:
    """Create a pipeline with default configuration"""
    config = ProcessingConfig(
        target_quality_threshold=0.7,
        deduplication_enabled=True,
        validation_enabled=True,
        safety_filtering_enabled=True,
        psychology_integration_enabled=True,
        youtube_rag_integration_enabled=True,
    )
    return UnifiedPreprocessingPipeline(config)


def run_pipeline() -> str:
    """Run the complete preprocessing pipeline"""
    pipeline = create_default_pipeline()
    return pipeline.execute_pipeline()


if __name__ == "__main__":
    # Example usage
    try:
        final_dataset_path = run_pipeline()
        logger.info("Pipeline completed successfully. Dataset saved to: %s", final_dataset_path)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise
