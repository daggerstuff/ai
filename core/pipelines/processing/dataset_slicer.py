"""
PIX-35: Dataset Slicer — splits normalized data into stage-specific slices.

Reads normalized/deduped JSONL files, classifies each record into a training
stage, and writes stage-specific output files with manifest tracking.

Output layout:
  <output_dir>/
  ├── stage1_foundation.jsonl
  ├── stage2_therapeutic_expertise.jsonl
  ├── stage3_edge_stress_test.jsonl
  ├── stage4_voice_persona.jsonl
  ├── supplementary.jsonl
  └── slice_manifest.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .stage_classifier import (
    Stage,
    StageClassifier,
)

logger = logging.getLogger(__name__)


@dataclass
class SliceManifest:
    """Manifest tracking for a dataset slice operation."""

    slice_id: str
    created_at: str
    input_files: list[str]
    stage_targets: dict[str, int]
    stage_counts: dict[str, int]
    total_records: int
    classified_records: int
    supplementary_records: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    classification_confidence_avg: float = 0.0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "created_at": self.created_at,
            "input_files": self.input_files,
            "stage_targets": self.stage_targets,
            "stage_counts": self.stage_counts,
            "total_records": self.total_records,
            "classified_records": self.classified_records,
            "supplementary_records": self.supplementary_records,
            "rejection_reasons": self.rejection_reasons,
            "classification_confidence_avg": self.classification_confidence_avg,
            "processing_time_seconds": self.processing_time_seconds,
        }


@dataclass
class SliceResult:
    """Result of a complete slice operation."""

    manifest: SliceManifest
    output_dir: Path
    stage_files: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        m = self.manifest
        lines = [
            "PIX-35 Dataset Slice Result",
            "=" * 40,
            f"  Slice ID:             {m.slice_id}",
            f"  Output directory:     {self.output_dir}",
            f"  Input files:          {len(m.input_files)}",
            f"  Total records:        {m.total_records}",
            f"  Classified:           {m.classified_records}",
            f"  Supplementary:        {m.supplementary_records}",
            f"  Processing time:      {m.processing_time_seconds:.2f}s",
            "",
            "Stage Distribution:",
        ]
        for stage_name, count in sorted(m.stage_counts.items()):
            target = m.stage_targets.get(stage_name, 0)
            target_str = f" (target: {target})" if target else ""
            lines.append(f"    {stage_name}: {count}{target_str}")

        if m.classification_confidence_avg > 0:
            conf = f"{m.classification_confidence_avg:.2%}"
            lines.append(f"\n  Avg classification confidence: {conf}")

        if m.rejection_reasons:
            lines.append("\n  Rejection reasons:")
            for reason, count in sorted(m.rejection_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")

        lines.append("\nStage Files:")
        for stage_name, file_path in sorted(self.stage_files.items()):
            lines.append(f"    {stage_name}: {file_path}")

        return "\n".join(lines)


class DatasetSlicer:
    """
    Slices normalized JSONL data into stage-specific datasets.

    Pipeline:
      1. Read normalized JSONL records
      2. Classify each record into a stage
      3. Write stage-specific output files
      4. Generate slice manifest
    """

    def __init__(
        self,
        stage_targets: dict[str, int] | None = None,
        enforce_targets: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            stage_targets: Target sample counts per stage.
            enforce_targets: If True, cap each stage at its target.
            output_dir: Output directory for sliced files.
        """
        self.classifier = StageClassifier(
            stage_targets=stage_targets,
            enforce_targets=enforce_targets,
        )
        self.stage_targets = stage_targets or {}
        self.enforce_targets = enforce_targets
        self.output_dir = Path(output_dir) if output_dir else Path("sliced_output")

    def slice(
        self,
        input_paths: list[str | Path],
        slice_id: str | None = None,
    ) -> SliceResult:
        """
        Execute the full slicing pipeline.

        Args:
            input_paths: List of normalized JSONL file paths.
            slice_id: Unique identifier for this slice (auto-generated if None).

        Returns:
            SliceResult with manifest and output file paths.
        """
        start_time = time.monotonic()

        if slice_id is None:
            slice_id = f"slice-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read all records
        all_records: list[dict[str, Any]] = []
        input_files: list[str] = []
        rejection_reasons: dict[str, int] = {}

        for path in input_paths:
            p = Path(path)
            if p.is_file() and p.suffix == ".jsonl":
                input_files.append(str(p))
                records = self._read_jsonl(p, rejection_reasons)
                all_records.extend(records)
            elif p.is_dir():
                for jsonl_file in sorted(p.rglob("*.jsonl")):
                    input_files.append(str(jsonl_file))
                    records = self._read_jsonl(jsonl_file, rejection_reasons)
                    all_records.extend(records)
            else:
                logger.warning("Input path not found or not JSONL: %s", p)

        # Classify records
        classified, counts = self.classifier.classify_batch(all_records)

        # Write stage-specific files
        stage_files: dict[str, Path] = {}
        stage_file_handles: dict[str, Any] = {}
        confidence_sum = 0.0
        confidence_count = 0

        for stage in Stage:
            stage_file = self.output_dir / f"{stage.value}.jsonl"
            stage_files[stage.value] = stage_file
            stage_file_handles[stage.value] = stage_file.open("w", encoding="utf-8")

        try:
            for record, result in classified:
                stage_key = result.stage.value
                handle = stage_file_handles[stage_key]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                confidence_sum += result.confidence
                confidence_count += 1
        finally:
            for handle in stage_file_handles.values():
                handle.close()

        # Compute average confidence
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0

        processing_time = time.monotonic() - start_time

        manifest = SliceManifest(
            slice_id=slice_id,
            created_at=datetime.now(UTC).isoformat(),
            input_files=input_files,
            stage_targets=self.stage_targets,
            stage_counts=counts.to_dict(),
            total_records=len(all_records),
            classified_records=len(classified),
            supplementary_records=counts.supplementary,
            rejection_reasons=rejection_reasons,
            classification_confidence_avg=avg_confidence,
            processing_time_seconds=processing_time,
        )

        # Write manifest
        manifest_path = self.output_dir / "slice_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Slice complete: %s", manifest.slice_id)
        return SliceResult(
            manifest=manifest,
            output_dir=self.output_dir,
            stage_files=stage_files,
        )

    def _read_jsonl(
        self,
        path: Path,
        rejection_reasons: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Read and validate JSONL records from a file."""
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as exc:
                    reason = "json_parse_error"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    logger.warning("Failed to parse %s line %d: %s", path.name, line_num, exc)
        logger.info("Read %d records from %s", len(records), path.name)
        return records


__all__ = [
    "DatasetSlicer",
    "SliceManifest",
    "SliceResult",
]
