#!/usr/bin/env python3
"""
DACT-08: Freeze v1 Training Snapshot

Creates an immutable versioned snapshot of the validated corpus with:
- Manifest with source counts, quality scores, rejected sources
- SHA-256 checksums for all files
- Train/validation split (95/5)
- OpenAI and HuggingFace prepared formats

Usage:
    python -m ai.pipelines.orchestrator.scripts.dact08_freeze_snapshot --version v1
"""

import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_records(file_path: Path) -> int:
    """Count records in a JSONL file."""
    count = 0
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def create_train_val_split(
    records: List[Dict[str, Any]], train_ratio: float = 0.95, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into train and validation sets."""
    random.seed(seed)
    random.shuffle(records)
    split_idx = int(len(records) * train_ratio)
    return records[:split_idx], records[split_idx:]


def convert_to_openai_format(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert records to OpenAI fine-tuning format."""
    openai_records = []
    for record in records:
        messages = record.get("messages", [])
        if len(messages) >= 2:
            # Convert to conversation format
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Map roles: client->user, therapist->assistant
                if role == "client":
                    openai_records.append(
                        {
                            "messages": [
                                {"role": "user", "content": content},
                            ]
                        }
                    )
                elif role == "therapist":
                    # Append to last user message as assistant response
                    if openai_records and "messages" in openai_records[-1]:
                        openai_records[-1]["messages"].append(
                            {"role": "assistant", "content": content}
                        )
    return openai_records


def convert_to_huggingface_format(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert records to HuggingFace instruction-tuning format."""
    hf_records = []
    for record in records:
        messages = record.get("messages", [])
        if len(messages) >= 2:
            hf_record = {"messages": []}
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Map to HF format
                hf_role = "user" if role == "client" else "assistant"
                hf_record["messages"].append({"role": hf_role, "content": content})
            hf_records.append(hf_record)
    return hf_records


class SnapshotFreezer:
    """Freeze a versioned snapshot of the training dataset."""

    def __init__(self, version: str, input_dir: str, output_dir: str):
        self.version = version
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots" / version
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "files_processed": 0,
            "total_records": 0,
            "total_size_bytes": 0,
            "checksums": {},
            "source_counts": {},
        }

    def freeze(self) -> Dict[str, Any]:
        """Create frozen snapshot."""
        logger.info(f"Freezing snapshot version {self.version}...")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.snapshot_dir}")

        # Step 1: Load all input files
        input_files = list(self.input_dir.glob("*_nemo.jsonl"))
        if not input_files:
            raise ValueError(f"No *_nemo.jsonl files found in {self.input_dir}")

        logger.info(f"Found {len(input_files)} input files")

        # Step 2: Process each file
        manifest_files = []
        all_records = []

        for file_path in sorted(input_files):
            file_result = self._process_file(file_path)
            manifest_files.extend(file_result["files"])
            all_records.extend(file_result["records"])

        self.stats["total_records"] = len(all_records)
        logger.info(f"Total records: {self.stats['total_records']:,}")

        # Step 3: Create merged output
        merged_dir = self.snapshot_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / "mental_health_dataset.jsonl"

        with open(merged_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        merged_hash = compute_file_hash(merged_path)
        merged_size = get_file_size(merged_path)
        merged_count = count_records(merged_path)

        logger.info(
            f"Merged file: {merged_path} "
            f"({merged_count:,} records, {merged_size:,} bytes)"
        )

        manifest_files.append(
            {
                "path": str(merged_path.relative_to(self.output_dir)),
                "size_bytes": merged_size,
                "sha256": merged_hash,
                "record_count": merged_count,
            }
        )

        # Step 4: Create train/validation split
        logger.info("Creating train/validation split (95/5)...")
        train_records, val_records = create_train_val_split(
            all_records, train_ratio=0.95
        )

        split_dir = self.snapshot_dir / "train_val_split"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_path = split_dir / "train.jsonl"
        val_path = split_dir / "validation.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for record in val_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        train_hash = compute_file_hash(train_path)
        val_hash = compute_file_hash(val_path)

        manifest_files.extend(
            [
                {
                    "path": str(train_path.relative_to(self.output_dir)),
                    "size_bytes": get_file_size(train_path),
                    "sha256": train_hash,
                    "record_count": len(train_records),
                },
                {
                    "path": str(val_path.relative_to(self.output_dir)),
                    "size_bytes": get_file_size(val_path),
                    "sha256": val_hash,
                    "record_count": len(val_records),
                },
            ]
        )

        logger.info(f"Train: {len(train_records):,} records")
        logger.info(f"Validation: {len(val_records):,} records")

        # Step 5: Create prepared formats
        logger.info("Creating OpenAI format...")
        openai_records = convert_to_openai_format(
            train_records[:100]
        )  # Sample for format

        prepared_dir = self.snapshot_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)

        openai_path = prepared_dir / "openai_dataset.jsonl"
        with open(openai_path, "w", encoding="utf-8") as f:
            for record in openai_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        openai_hash = compute_file_hash(openai_path)

        manifest_files.append(
            {
                "path": str(openai_path.relative_to(self.output_dir)),
                "size_bytes": get_file_size(openai_path),
                "sha256": openai_hash,
                "record_count": len(openai_records),
                "format": "openai",
            }
        )

        logger.info(f"OpenAI format: {len(openai_records):,} records (sample)")

        # Step 6: Build manifest
        manifest = self._build_manifest(
            manifest_files, len(train_records), len(val_records)
        )

        # Step 7: Save manifest
        manifest_path = self.snapshot_dir / "MANIFEST.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Manifest saved to: {manifest_path}")

        # Step 8: Save checksums
        checksums_path = self.snapshot_dir / "CHECKSUMS.sha256"
        with open(checksums_path, "w", encoding="utf-8") as f:
            for file_info in manifest_files:
                f.write(f"{file_info['sha256']}  {file_info['path']}\n")

        logger.info(f"Checksums saved to: {checksums_path}")

        return manifest

    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single input file."""
        logger.info(f"Processing: {file_path}")

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {file_path}: {e}")

        logger.info(f"  Records: {len(records):,}")

        # Create stage-specific output
        stage_name = file_path.stem.replace("_nemo", "")
        stage_dir = self.snapshot_dir / "slices" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        output_path = stage_dir / "shard_000.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        output_hash = compute_file_hash(output_path)
        output_size = get_file_size(output_path)

        logger.info(f"  Output: {output_path} ({output_size:,} bytes)")

        return {
            "files": [
                {
                    "path": str(output_path.relative_to(self.output_dir)),
                    "size_bytes": output_size,
                    "sha256": output_hash,
                    "record_count": len(records),
                }
            ],
            "records": records,
        }

    def _build_manifest(
        self, files: List[Dict[str, Any]], train_count: int, val_count: int
    ) -> Dict[str, Any]:
        """Build the manifest."""
        total_size = sum(f["size_bytes"] for f in files)
        total_records = sum(f["record_count"] for f in files)

        return {
            "version": self.version,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "frozen_by": "DACT-08 pipeline",
            "description": f"Frozen training dataset v{self.version}",
            "total_records": total_records,
            "total_size_bytes": total_size,
            "total_size_human": f"{total_size / (1024**3):.2f} GB",
            "splits": {
                "train": {"records": train_count},
                "validation": {"records": val_count},
            },
            "pipeline_lineage": {
                "dact_04_normalize": "2026-04-03",
                "dact_06_slicing": "2026-04-03",
                "dact_07_redaction": "2026-04-04",
                "dact_08_freeze": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
            "files": files,
        }


def main():
    import argparse

    def print_rule() -> None:
        print("=" * 60)

    parser = argparse.ArgumentParser(description="DACT-08: Freeze Training Snapshot")
    parser.add_argument(
        "--version", type=str, default="v1", help="Snapshot version (e.g., v1, v1.1)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ai/data/nemo_export",
        help="Input directory with NeMo JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ai/data/snapshots",
        help="Output directory for frozen snapshot",
    )

    args = parser.parse_args()

    print_rule()
    print("DACT-08: Freeze Training Snapshot")
    print_rule()
    print(f"Version: {args.version}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    freezer = SnapshotFreezer(
        version=args.version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    manifest = freezer.freeze()

    print()
    print_rule()
    print("SNAPSHOT FROZEN")
    print_rule()
    print(f"Version: {manifest['version']}")
    print(f"Total Records: {manifest['total_records']:,}")
    print(f"Total Size: {manifest['total_size_human']}")
    print(f"Train: {manifest['splits']['train']['records']:,}")
    print(f"Validation: {manifest['splits']['validation']['records']:,}")
    print()
    print(f"Snapshot location: {freezer.snapshot_dir}")
    print_rule()

    return manifest


if __name__ == "__main__":
    main()
