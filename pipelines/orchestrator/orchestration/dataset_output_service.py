"""Dataset and split artifact persistence for assembled training data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai.pipelines.orchestrator.ingestion.intake_routing_adapter import (
    split_records_with_preferences,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.dataset_output_service")


class DatasetOutputConfigProtocol(Protocol):
    output_dir: str
    output_filename: str
    stage_distribution: dict[str, float]


class DatasetOutputStatsProtocol(Protocol):
    samples_by_source: dict[str, int]
    samples_by_stage: dict[str, int]
    warnings: list[str]
    errors: list[str]
    stage_balance: dict[str, dict[str, Any]]
    split_counts: dict[str, dict[str, int]]


class DatasetOutputService:
    """Own dataset JSON export plus per-stage and split artifact emission."""

    def __init__(
        self,
        *,
        config: DatasetOutputConfigProtocol,
        stats: DatasetOutputStatsProtocol,
    ) -> None:
        self.config = config
        self.stats = stats
        self._cached_split_source_id: int | None = None
        self._cached_aggregate_split: Any | None = None
        self._cached_stage_split_map: dict[str, Any] = {}

    def save_dataset(self, data: list[dict[str, Any]]) -> str:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config.output_filename
        self._prime_split_cache(data)

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

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write('{"conversations":[')
            for index, record in enumerate(data):
                if index:
                    handle.write(",")
                handle.write(json.dumps(record, separators=(",", ":")))
            handle.write('],"metadata":')
            handle.write(json.dumps(output_data["metadata"], separators=(",", ":")))
            handle.write("}")

        logger.info("💾 Saved dataset to %s", output_path)
        return str(output_path)

    def write_stage_outputs(self, stage_segments: dict[str, list[dict[str, Any]]]) -> None:
        stage_dir = Path("ai/training_data_consolidated/final")
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "stages": {}}
        stage_names = sorted(set(self.config.stage_distribution) | set(stage_segments))
        for stage in stage_names:
            records = stage_segments.get(stage, [])
            stage_file = stage_dir / f"MASTER_{stage}.jsonl"
            self._write_jsonl(stage_file, records)
            balance_stats = self.stats.stage_balance.get(stage, {})
            manifest["stages"][stage] = {
                "samples": len(records),
                "target": balance_stats.get("target"),
                "available": balance_stats.get("available"),
                "output_path": str(stage_file),
            }

        manifest_path = stage_dir / "MASTER_STAGE_MANIFEST.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        logger.info("🗂️  Stage manifest updated at %s", manifest_path)

    def write_split_outputs(self, data: list[dict[str, Any]]) -> None:
        split_root = Path("ai/training_data_consolidated/final/splits")
        split_root.mkdir(parents=True, exist_ok=True)

        self._prime_split_cache(data)
        aggregate_split = self._cached_aggregate_split
        self._write_jsonl_split_files(
            split_root,
            aggregate_split.train,
            aggregate_split.val,
            aggregate_split.test,
        )
        self.stats.split_counts["aggregate"] = {
            "train": len(aggregate_split.train),
            "val": len(aggregate_split.val),
            "test": len(aggregate_split.test),
        }

        stage_names = sorted(
            set(self.config.stage_distribution) | set(self._cached_stage_split_map)
        )
        for stage in stage_names:
            stage_dir = split_root / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_split = self._cached_stage_split_map.get(
                stage,
                self.split_records_with_preferences([]),
            )
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

        logger.info("🧪 Wrote aggregate and per-stage split artifacts to %s", split_root)

    @staticmethod
    def split_records_with_preferences(records: list[dict[str, Any]]) -> Any:
        """Expose split preferences behind the service for pipeline compatibility."""
        return split_records_with_preferences(records)

    def _prime_split_cache(self, data: list[dict[str, Any]]) -> None:
        source_id = id(data)
        if self._cached_split_source_id == source_id:
            return

        self._cached_split_source_id = source_id
        self._cached_aggregate_split = self.split_records_with_preferences(data)

        by_stage: dict[str, list[dict[str, Any]]] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            by_stage.setdefault(stage, []).append(item)
        self._cached_stage_split_map = {
            stage: self.split_records_with_preferences(records)
            for stage, records in by_stage.items()
        }

    def _write_jsonl_split_files(
        self,
        output_dir: Path,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
    ) -> None:
        split_map = {
            "train.jsonl": train_data,
            "val.jsonl": val_data,
            "test.jsonl": test_data,
        }
        for filename, records in split_map.items():
            self._write_jsonl(output_dir / filename, records)

    @staticmethod
    def _write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")


__all__ = ["DatasetOutputService"]
