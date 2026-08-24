"""
PIX-34: Dataset Freezer — create versioned training dataset snapshots.

Reads sliced stage-specific JSONL files, computes comprehensive statistics
(source counts, quality scores, rejection reasons, license distribution),
and writes a versioned snapshot with manifest for training pipeline consumption.

Output layout:
  <output_dir>/
  ├── v1_training_slice/
  │   ├── slice_manifest.json          # Full statistics and metadata
  │   ├── stage1_foundation.jsonl      # Copied from sliced input
  │   ├── stage2_therapeutic_expertise.jsonl
  │   ├── stage3_edge_stress_test.jsonl
  │   ├── stage4_voice_persona.jsonl
  │   └── supplementary.jsonl
  └── latest -> v1_training_slice/     # Symlink to current version
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SourceStats:
    """Statistics for a single data source."""

    count: int = 0
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0
    licenses: dict[str, int] = field(default_factory=dict)
    topic_tags: dict[str, int] = field(default_factory=dict)
    therapeutic_modalities: dict[str, int] = field(default_factory=dict)

    def update(self, record: dict[str, Any]) -> None:
        self.count += 1
        metadata = record.get("metadata", {}) or {}
        quality = metadata.get("quality_score", 0.0) or 0.0

        # Update quality stats
        self.avg_quality_score = (self.avg_quality_score * (self.count - 1) + quality) / self.count
        self.min_quality_score = min(self.min_quality_score, quality)
        self.max_quality_score = max(self.max_quality_score, quality)

        # Track licenses
        lic = record.get("license", "unknown") or "unknown"
        self.licenses[lic] = self.licenses.get(lic, 0) + 1

        # Track topic tags
        for tag in metadata.get("topic_tags") or []:
            self.topic_tags[tag] = self.topic_tags.get(tag, 0) + 1

        # Track therapeutic modalities
        modality = metadata.get("therapeutic_modality", "N/A") or "N/A"
        self.therapeutic_modalities[modality] = self.therapeutic_modalities.get(modality, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "min_quality_score": round(self.min_quality_score, 4),
            "max_quality_score": round(self.max_quality_score, 4),
            "licenses": dict(sorted(self.licenses.items(), key=lambda x: -x[1])),
            "top_topic_tags": dict(sorted(self.topic_tags.items(), key=lambda x: -x[1])[:20]),
            "therapeutic_modalities": dict(sorted(self.therapeutic_modalities.items(), key=lambda x: -x[1])),
        }


@dataclass
class StageStats:
    """Statistics for a single training stage."""

    total_records: int = 0
    sources: dict[str, SourceStats] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    file_size_bytes: int = 0
    file_path: str = ""

    def update(self, record: dict[str, Any]) -> None:
        self.total_records += 1
        source = record.get("source", "unknown") or "unknown"
        if source not in self.sources:
            self.sources[source] = SourceStats()
        self.sources[source].update(record)

        metadata = record.get("metadata", {}) or {}
        quality = metadata.get("quality_score", 0.0) or 0.0
        self.avg_quality_score = (self.avg_quality_score * (self.total_records - 1) + quality) / self.total_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "file_size_bytes": self.file_size_bytes,
            "file_path": self.file_path,
            "sources": {src: stats.to_dict() for src, stats in sorted(self.sources.items(), key=lambda x: -x[1].count)},
        }


@dataclass
class FreezeManifest:
    """Complete manifest for a frozen dataset version."""

    version: str
    created_at: str
    created_by: str
    slice_id: str
    input_dir: str
    output_dir: str
    total_records: int
    stage_stats: dict[str, dict[str, Any]]
    global_stats: dict[str, Any]
    quality_distribution: dict[str, int]
    license_distribution: dict[str, int]
    rejection_summary: dict[str, Any]
    processing_time_seconds: float
    pix_tickets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "slice_id": self.slice_id,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "total_records": self.total_records,
            "stage_stats": self.stage_stats,
            "global_stats": self.global_stats,
            "quality_distribution": self.quality_distribution,
            "license_distribution": self.license_distribution,
            "rejection_summary": self.rejection_summary,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "pix_tickets": self.pix_tickets,
        }


@dataclass
class FreezeResult:
    """Result of a freeze operation."""

    manifest: FreezeManifest
    output_dir: Path
    version_dir: Path

    def summary(self) -> str:
        """Human-readable summary."""
        m = self.manifest
        lines = [
            "PIX-34 Dataset Freeze Result",
            "=" * 40,
            f"  Version:              {m.version}",
            f"  Slice ID:             {m.slice_id}",
            f"  Output directory:     {self.version_dir}",
            f"  Total records:        {m.total_records}",
            f"  Processing time:      {m.processing_time_seconds:.2f}s",
            "",
            "Stage Breakdown:",
        ]
        for stage_name, stats in sorted(m.stage_stats.items()):
            records = stats.get("total_records", 0)
            quality = stats.get("avg_quality_score", 0)
            sources = len(stats.get("sources", {}))
            lines.append(f"    {stage_name}: {records} records, avg quality {quality:.2f}, {sources} sources")

        lines.append("")
        lines.append("License Distribution:")
        for lic, count in sorted(m.license_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"    {lic}: {count}")

        lines.append("")
        lines.append("Quality Distribution:")
        for bucket, count in sorted(m.quality_distribution.items()):
            lines.append(f"    {bucket}: {count}")

        if m.rejection_summary:
            lines.append("")
            lines.append("Rejection Summary:")
            for key, val in m.rejection_summary.items():
                lines.append(f"    {key}: {val}")

        return "\n".join(lines)


class DatasetFreezer:
    """
    Creates versioned snapshots of sliced training datasets.

    Pipeline:
      1. Read sliced stage JSONL files
      2. Compute per-source and per-stage statistics
      3. Compute global quality and license distributions
      4. Copy stage files to versioned output directory
      5. Write freeze manifest
      6. Update 'latest' symlink
    """

    STAGES = [
        "stage1_foundation",
        "stage2_therapeutic_expertise",
        "stage3_edge_stress_test",
        "stage4_voice_persona",
        "supplementary",
    ]

    def __init__(
        self,
        version: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            version: Version string (e.g., "v1"). Auto-generated if None.
            output_dir: Output directory for frozen snapshots.
        """
        self.version = version or f"v{datetime.now(UTC).strftime('%Y%m%d')}"
        self.output_dir = Path(output_dir) if output_dir else Path("frozen_datasets")

    def freeze(
        self,
        slice_dir: str | Path,
        slice_id: str | None = None,
        rejection_report: str | Path | None = None,
    ) -> FreezeResult:
        """
        Execute the full freeze pipeline.

        Args:
            slice_dir: Directory containing sliced stage JSONL files.
            slice_id: Source slice identifier.
            rejection_report: Path to rejection report JSON from PIX-32.

        Returns:
            FreezeResult with manifest and output paths.
        """
        start_time = time.monotonic()
        slice_dir = Path(slice_dir)

        if not slice_dir.is_dir():
            raise FileNotFoundError(f"Slice directory not found: {slice_dir}")

        # Read slice manifest if available
        slice_manifest_path = slice_dir / "slice_manifest.json"
        slice_manifest_data: dict[str, Any] = {}
        if slice_manifest_path.exists():
            with slice_manifest_path.open("r", encoding="utf-8") as f:
                slice_manifest_data = json.load(f)

        # Create versioned output directory
        version_dir = self.output_dir / self.version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Process each stage
        stage_stats: dict[str, dict[str, Any]] = {}
        global_quality_scores: list[float] = []
        global_licenses: dict[str, int] = defaultdict(int)
        total_records = 0

        for stage_name in self.STAGES:
            stage_file = slice_dir / f"{stage_name}.jsonl"
            if not stage_file.exists():
                logger.warning("Stage file not found: %s", stage_file)
                stage_stats[stage_name] = StageStats().to_dict()
                continue

            stats = StageStats()
            with stage_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        stats.update(record)

                        metadata = record.get("metadata", {}) or {}
                        quality = metadata.get("quality_score", 0.0) or 0.0
                        global_quality_scores.append(quality)

                        lic = record.get("license", "unknown") or "unknown"
                        global_licenses[lic] += 1
                    except json.JSONDecodeError:
                        continue

            # Copy stage file to versioned output
            dest_file = version_dir / f"{stage_name}.jsonl"
            shutil.copy2(stage_file, dest_file)

            stats.file_size_bytes = dest_file.stat().st_size
            stats.file_path = str(dest_file)
            total_records += stats.total_records
            stage_stats[stage_name] = stats.to_dict()

            logger.info(
                "Stage %s: %d records, avg quality %.2f, %d sources",
                stage_name,
                stats.total_records,
                stats.avg_quality_score,
                len(stats.sources),
            )

        # Compute quality distribution buckets
        quality_distribution = {
            "excellent (0.9-1.0)": 0,
            "good (0.7-0.9)": 0,
            "acceptable (0.5-0.7)": 0,
            "low (0.3-0.5)": 0,
            "very_low (0.0-0.3)": 0,
        }
        for score in global_quality_scores:
            if score >= 0.9:
                quality_distribution["excellent (0.9-1.0)"] += 1
            elif score >= 0.7:
                quality_distribution["good (0.7-0.9)"] += 1
            elif score >= 0.5:
                quality_distribution["acceptable (0.5-0.7)"] += 1
            elif score >= 0.3:
                quality_distribution["low (0.3-0.5)"] += 1
            else:
                quality_distribution["very_low (0.0-0.3)"] += 1

        # Compute rejection summary
        rejection_summary: dict[str, Any] = {}
        if rejection_report and Path(rejection_report).exists():
            with Path(rejection_report).open("r", encoding="utf-8") as f:
                try:
                    rejection_data = json.load(f)
                    rejection_summary = rejection_data.get("rejection_reasons", rejection_data)
                except json.JSONDecodeError:
                    rejection_summary = {"error": "Failed to parse rejection report"}

        # Global stats
        avg_global_quality = sum(global_quality_scores) / len(global_quality_scores) if global_quality_scores else 0.0
        global_stats = {
            "total_records": total_records,
            "total_sources": len(global_licenses),
            "avg_quality_score": round(avg_global_quality, 4),
            "stages_processed": len([s for s in self.STAGES if stage_stats.get(s, {}).get("total_records", 0) > 0]),
        }

        processing_time = time.monotonic() - start_time

        # Build manifest
        manifest = FreezeManifest(
            version=self.version,
            created_at=datetime.now(UTC).isoformat(),
            created_by=os.environ.get("USER", "pipeline"),
            slice_id=slice_id or slice_manifest_data.get("slice_id", "unknown"),
            input_dir=str(slice_dir),
            output_dir=str(version_dir),
            total_records=total_records,
            stage_stats=stage_stats,
            global_stats=global_stats,
            quality_distribution=quality_distribution,
            license_distribution=dict(sorted(global_licenses.items(), key=lambda x: -x[1])),
            rejection_summary=rejection_summary,
            processing_time_seconds=processing_time,
            pix_tickets=[
                "PIX-27",
                "PIX-28",
                "PIX-30",
                "PIX-31",
                "PIX-32",
                "PIX-35",
                "PIX-34",
            ],
        )

        # Write manifest
        manifest_path = version_dir / "slice_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        # Update 'latest' symlink
        latest_link = self.output_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.name, target_is_directory=True)

        logger.info("Freeze complete: %s (%d records)", self.version, total_records)
        return FreezeResult(
            manifest=manifest,
            output_dir=self.output_dir,
            version_dir=version_dir,
        )


__all__ = [
    "DatasetFreezer",
    "FreezeManifest",
    "FreezeResult",
    "SourceStats",
    "StageStats",
]
