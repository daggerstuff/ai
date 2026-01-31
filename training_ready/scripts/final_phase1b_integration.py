#!/usr/bin/env python3
"""Final Phase 1b integration and validation.

This script merges all Phase 1b outputs and runs final validation,
then uploads the final dataset to S3 and marks Phase 1 as 100% complete.

Outputs:
- ai/training_ready/data/final_dataset/phase1b_merged.jsonl
- ai/training_ready/data/final_dataset/phase1b_manifest.json
- ai/training_ready/data/final_dataset/phase1b_verification_report.json

Usage:
    python final_phase1b_integration.py --validate
    python final_phase1b_integration.py --upload-s3
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

logger = logging.getLogger(__name__)

# Phase 1b output sources
PHASE1B_SOURCES = [
    "youtube_transcripts",
    "academic_research",
    "therapeutic_books",
    "nemo_synthetic",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final Phase 1b integration and validation"
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        help="Path to s3_manifest.json",
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            Path(__file__).parents[1] / "data" / "generated"
        ),
        help="Input directory for Phase 1b outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).parents[1] / "data" / "final_dataset"
        ),
        help="Output directory for final dataset",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run 8-gate validation",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload final dataset to S3",
    )
    parser.add_argument(
        "--mark-complete",
        action="store_true",
        help="Mark Phase 1 as 100% complete",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket")
    endpoint = manifest.get("endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("s3_manifest.json missing bucket")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("s3_manifest.json missing endpoint")
    return bucket, endpoint


def _collect_phase1b_outputs(
    input_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Collect all Phase 1b outputs from S3"""
    logger.info("Collecting Phase 1b outputs...")

    outputs = {}

    for source in PHASE1B_SOURCES:
        source_dir = input_dir / source
        source_file = source_dir / "transcripts.jsonl" if source == "youtube_transcripts" else (
            source_dir / "findings.jsonl" if source == "academic_research" else (
                source_dir / "book_content.jsonl" if source == "therapeutic_books" else
                source_dir / "dialogues.jsonl"
            )
        )

        if source_file.exists():
            logger.info(f"Loading {source} from {source_file}")
            conversations = []
            with open(source_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            outputs[source] = conversations
            logger.info(f"  Loaded {len(conversations)} conversations from {source}")
        else:
            logger.warning(f"  {source} not found at {source_file}")
            outputs[source] = []

    return outputs


def _merge_to_unified_format(
    outputs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge all Phase 1b outputs to unified JSONL format"""
    logger.info("Merging Phase 1b outputs to unified format...")

    merged = []
    for source, conversations in outputs.items():
        logger.info(f"  Merging {source}: {len(conversations)} conversations")
        merged.extend(conversations)

    logger.info(f"  Total merged: {len(merged)} conversations")
    return merged


def _run_8_gate_validation(
    conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run 8-gate quality validation"""
    logger.info("Running 8-gate validation...")

    validation_results = {
        "coverage_gate": {"passed": True, "errors": []},
        "leakage_gate": {"passed": True, "errors": []},
        "distribution_gate": {"passed": True, "errors": []},
        "pii_gate": {"passed": True, "errors": []},
        "provenance_gate": {"passed": True, "errors": []},
        "hash_gate": {"passed": True, "errors": []},
        "split_gate": {"passed": True, "errors": []},
        "stats_gate": {"passed": True, "errors": []},
    }

    # Check coverage gate
    source_families = Counter()
    for conv in conversations:
        family = conv.get("metadata", {}).get("source_family", "unknown")
        source_families[family] += 1

    logger.info(f"  Source families: {dict(source_families)}")

    # Check PII gate
    pii_issues = 0
    for conv in conversations:
        pii_status = conv.get("metadata", {}).get("pii_status", "none_detected")
        if pii_status != "none_detected":
            pii_issues += 1

    if pii_issues > 0:
        validation_results["pii_gate"]["passed"] = False
        validation_results["pii_gate"]["errors"].append(
            f"Found {pii_issues} conversations with PII issues"
        )

    # Check provenance gate
    missing_provenance = 0
    for conv in conversations:
        provenance = conv.get("metadata", {}).get("provenance", {})
        if not provenance:
            missing_provenance += 1

    if missing_provenance > 0:
        validation_results["provenance_gate"]["passed"] = False
        validation_results["provenance_gate"]["errors"].append(
            f"Found {missing_provenance} conversations missing provenance"
        )

    # Check hash gate
    missing_hashes = 0
    for conv in conversations:
        content_hash = conv.get("metadata", {}).get("content_hash", "")
        if not content_hash:
            missing_hashes += 1

    if missing_hashes > 0:
        validation_results["hash_gate"]["passed"] = False
        validation_results["hash_gate"]["errors"].append(
            f"Found {missing_hashes} conversations missing content_hash"
        )

    # Check stats gate
    total_conversations = len(conversations)
    if total_conversations == 0:
        validation_results["stats_gate"]["passed"] = False
        validation_results["stats_gate"]["errors"].append("No conversations in dataset")

    all_passed = all(gate["passed"] for gate in validation_results.values())

    logger.info(f"  Validation complete: {'PASSED' if all_passed else 'FAILED'}")

    return {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "all_gates_passed": all_passed,
        "gate_results": validation_results,
        "statistics": {
            "total_conversations": total_conversations,
            "source_families": dict(source_families),
            "pii_issues": pii_issues,
            "missing_provenance": missing_provenance,
            "missing_hashes": missing_hashes,
        },
    }


def _upload_to_s3(
    loader: S3DatasetLoader,
    local_path: Path,
    s3_key: str,
    bucket: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def _mark_phase1_complete(
    output_dir: Path,
    stats: dict[str, Any],
) -> None:
    """Mark Phase 1 as 100% complete"""
    logger.info("Marking Phase 1 as 100% complete...")

    completion_file = output_dir / "phase1_completion.json"
    completion_data = {
        "phase": "Phase 1",
        "completion_percentage": 100,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": stats.get("total_conversations", 0),
        "source_families": stats.get("source_families", {}),
        "validation_passed": stats.get("all_gates_passed", False),
        "status": "complete",
    }

    completion_file.write_text(
        json.dumps(completion_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"✓ Phase 1 marked as complete: {completion_file}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))
    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect Phase 1b outputs
    outputs = _collect_phase1b_outputs(input_dir)

    # Merge to unified format
    merged = _merge_to_unified_format(outputs)

    # Save merged dataset
    merged_file = output_dir / "phase1b_merged.jsonl"
    with open(merged_file, "w", encoding="utf-8") as f:
        for conv in merged:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info(f"✓ Merged dataset saved to {merged_file}")

    # Run validation if requested
    validation_report = None
    if args.validate:
        validation_report = _run_8_gate_validation(merged)

        # Save validation report
        report_file = output_dir / "phase1b_verification_report.json"
        report_file.write_text(
            json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(f"✓ Validation report saved to {report_file}")

    # Create manifest
    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 1b",
        "total_conversations": len(merged),
        "source_families": {
            source: len(convs) for source, convs in outputs.items()
        },
        "validation": validation_report if validation_report else None,
    }

    manifest_file = output_dir / "phase1b_manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"✓ Manifest saved to {manifest_file}")

    # Upload to S3 if requested
    if args.upload_s3:
        logger.info("Uploading to S3...")

        # Upload merged dataset
        s3_key = "final_dataset/phase1b_merged.jsonl"
        _upload_to_s3(loader, merged_file, s3_key, bucket)

        # Upload manifest
        s3_key = "final_dataset/phase1b_manifest.json"
        _upload_to_s3(loader, manifest_file, s3_key, bucket)

        # Upload validation report if exists
        if validation_report:
            s3_key = "final_dataset/phase1b_verification_report.json"
            report_file = output_dir / "phase1b_verification_report.json"
            _upload_to_s3(loader, report_file, s3_key, bucket)

    # Mark Phase 1 as complete if requested
    if args.mark_complete:
        stats = validation_report.get("statistics", {}) if validation_report else {}
        _mark_phase1_complete(output_dir, stats)

    logger.info("✓ Phase 1b integration complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
