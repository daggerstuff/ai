#!/usr/bin/env python3
"""
DACT-06: Stage-Based Dataset Slicing

Splits validated corpus into stage-specific slices for training.
Each slice is small enough to inspect and version cleanly.

Usage:
    python -m ai.core.pipelines.processing.stage_slicer --output-dir ai/data/staged_datasets
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StageMapping:
    """Maps dataset sources to training stages."""

    stage_id: str
    stage_name: str
    target_share: float
    source_files: list[str] = field(default_factory=list)
    output_file: str = ""


# Stage taxonomy mapping local datasets to stages
# Paths are relative to the ai/ directory (where this script runs from)
STAGE_TAXONOMY = {
    "stage1_foundation": StageMapping(
        stage_id="stage1_foundation",
        stage_name="Stage 1 – Foundation & Rapport",
        target_share=0.40,
        source_files=[
            "data/normalized/mental_health_counseling_normalized.jsonl",
            "training/sliced/stage1_foundation/additional_specialized_filtered.json",
            "training/sliced/stage1_foundation/Psychology-6K.json",
        ],
        output_file="stage1_foundation.jsonl",
    ),
    "stage2_therapeutic_expertise": StageMapping(
        stage_id="stage2_therapeutic_expertise",
        stage_name="Stage 2 – Therapeutic Expertise & Reasoning",
        target_share=0.25,
        source_files=[
            "data/normalized/cot_reasoning_normalized.jsonl",
            "training/sliced/stage2_therapeutic_expertise/clinical_diagnosis_mental_health.json",
            "training/sliced/stage2_therapeutic_expertise/cultural_nuances.json",
        ],
        output_file="stage2_therapeutic_expertise.jsonl",
    ),
    "stage3_edge_stress_test": StageMapping(
        stage_id="stage3_edge_stress_test",
        stage_name="Stage 3 – Edge Stress Test & Scenario Bank",
        target_share=0.20,
        source_files=[
            "pipelines/edge_case/output/edge_cases_training_format.jsonl",
            "data/cleaned_v3/nightmare_fuel_cleaned.jsonl",
            "training/ready_packages/datasets/synthetic/nightmare_fuel.jsonl",
        ],
        output_file="stage3_edge_stress_test.jsonl",
    ),
    "stage4_voice_persona": StageMapping(
        stage_id="stage4_voice_persona",
        stage_name="Stage 4 – Voice, Persona & Delivery",
        target_share=0.15,
        source_files=[
            "data/tim_fletcher_voice/exports/tim_fletcher_conversations.jsonl",
            "data/heidi_priebe_voice/exports/heidi_priebe_conversations.jsonl",
            "data/doctorramani_voice/exports/doctorramani_conversations.jsonl",
            "data/therapy_in_a_nutshell_voice/exports/therapy_in_a_nutshell_conversations.jsonl",
            "data/patrick_teahan_voice/exports/patrick_teahan__conversations.jsonl",
            "data/crappy_childhood_fairy_voice/exports/crappy_childhood_fairy_conversations.jsonl",
            "data/doc_snipes_voice/exports/doc_snipes_conversations.jsonl",
        ],
        output_file="stage4_voice_persona.jsonl",
    ),
}


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_jsonl_file(file_path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Load records from a JSON file (array of objects or single object)."""
    if not file_path.exists():
        return []

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Wrap single object
            return [data]
        return []


def normalize_record(record: dict[str, Any], source: str, stage_id: str) -> dict[str, Any]:
    """Normalize a record to standard format with metadata."""
    # Extract or infer conversation/text fields
    text = (
        record.get("text", "")
        or record.get("content", "")
        or record.get("message", "")
        or record.get("conversation", "")
        or record.get("input", "")
        or record.get("output", "")
        or json.dumps(record, ensure_ascii=False)
    )

    # Create normalized record
    normalized = {
        "id": compute_sha256(f"{source}:{json.dumps(record, sort_keys=True)}")[:16],
        "text": str(text),
        "stage": stage_id,
        "source": source,
        "metadata": {
            "original_keys": list(record.keys()) if isinstance(record, dict) else [],
        },
    }

    # Preserve additional fields in metadata
    if isinstance(record, dict):
        for key in record:
            if key not in [
                "text",
                "content",
                "message",
                "conversation",
                "input",
                "output",
            ]:
                normalized["metadata"][key] = record[key]

    return normalized


def slice_datasets(output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    """
    Slice validated datasets into training stages.

    Args:
        output_dir: Directory to write staged datasets
        dry_run: If True, only report what would be done

    Returns:
        Report dictionary with slicing results
    """
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stages": {},
        "total_records": 0,
        "total_size_bytes": 0,
        "errors": [],
        "warnings": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_id, mapping in STAGE_TAXONOMY.items():
        stage_report = {
            "stage_id": stage_id,
            "stage_name": mapping.stage_name,
            "target_share": mapping.target_share,
            "source_files": [],
            "records_processed": 0,
            "records_written": 0,
            "output_file": str(output_dir / mapping.output_file),
            "size_bytes": 0,
        }

        all_records = []

        for source_path in mapping.source_files:
            source = Path(source_path)
            source_name = source.stem

            # Check if source exists
            if not source.exists():
                # Try relative to project root
                source = Path.cwd() / source_path
                if not source.exists():
                    report["warnings"].append(f"Source not found: {source_path}")
                    stage_report["source_files"].append(
                        {
                            "path": source_path,
                            "status": "not_found",
                            "records": 0,
                        }
                    )
                    continue

            # Load based on file type
            if source.suffix == ".jsonl":
                records = load_jsonl_file(source)
            elif source.suffix == ".json":
                records = load_json_file(source)
            elif source.is_dir():
                # Handle directory sources
                records = []
                for f in source.glob("*.json*"):
                    if f.suffix in {".jsonl", ".json"}:
                        records.extend(load_jsonl_file(f))
            else:
                report["warnings"].append(f"Unknown file type: {source_path}")
                continue

            # Normalize and collect records
            for record in records:
                normalized = normalize_record(record, source_name, stage_id)
                all_records.append(normalized)

            stage_report["source_files"].append(
                {
                    "path": source_path,
                    "status": "processed",
                    "records": len(records),
                }
            )

        # Write staged output
        output_file = output_dir / mapping.output_file
        if all_records and not dry_run:
            with open(output_file, "w", encoding="utf-8") as f:
                for record in all_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        stage_report["records_processed"] = len(all_records)
        stage_report["records_written"] = len(all_records) if not dry_run else 0

        if output_file.exists():
            stage_report["size_bytes"] = output_file.stat().st_size

        report["stages"][stage_id] = stage_report
        report["total_records"] += stage_report["records_processed"]
        report["total_size_bytes"] += stage_report["size_bytes"]

    return report


def generate_metadata(report: dict[str, Any], output_dir: Path) -> None:
    """Generate metadata file for the sliced datasets."""
    metadata = {
        "version": "1.0.0",
        "created_at": report["timestamp"],
        "dact": "DACT-06",
        "description": "Stage-based dataset slices for curriculum training",
        "stages": list(report["stages"].keys()),
        "total_records": report["total_records"],
        "total_size_bytes": report["total_size_bytes"],
        "stage_details": report["stages"],
    }

    metadata_file = output_dir / "dact06_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="DACT-06: Stage-Based Dataset Slicing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ai/data/staged_datasets",
        help="Output directory for staged datasets",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be done")
    parser.add_argument(
        "--generate-metadata",
        action="store_true",
        default=True,
        help="Generate metadata file",
    )

    args = parser.parse_args()

    # Execute slicing
    report = slice_datasets(Path(args.output_dir), dry_run=args.dry_run)

    # Print report

    for _stage_id, stage_report in report["stages"].items():
        for _src in stage_report["source_files"]:
            pass

    if report["warnings"]:
        for _warning in report["warnings"]:
            pass

    if report["errors"]:
        for _error in report["errors"]:
            pass

    # Generate metadata
    if args.generate_metadata and not args.dry_run:
        generate_metadata(report, Path(args.output_dir))

    return report


if __name__ == "__main__":
    main()
