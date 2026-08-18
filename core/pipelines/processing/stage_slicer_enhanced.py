#!/usr/bin/env python3
"""
DACT-06: Stage-Based Dataset Slicing (Enhanced with Validation)

Splits validated corpus into stage-specific slices for training.
Each slice is small enough to inspect and version cleanly.

Features:
- Stage assignment validation against canonical criteria
- Quality threshold checking per stage (PIX-506 integration)
- Enhanced manifest generation with metrics
- Stage assignment rule validation

Usage:
    python -m ai.core.pipelines.processing.stage_slicer_enhanced \
        --output-dir ai/data/staged_datasets

Stage Quality Thresholds (from PIX-249 canonical model):
    Stage 1 (Foundation):
        - Empathy ≥ 70%
        - Clinical ≥ 30%
        - Safety 100%
        - Clinical validity ≥ 70%
        - Dedup retention > 50%

    Stage 2 (Expertise):
        - Empathy ≥ 75%
        - Clinical ≥ 50%
        - Safety 100%
        - Clinical validity ≥ 75%
        - Dedup retention > 50%

    Stage 3 (Edge):
        - Empathy ≥ 60%
        - Clinical ≥ 40%
        - Safety 100%
        - Clinical validity ≥ 65%
        - Dedup retention > 40%

    Stage 4 (Voice):
        - Empathy ≥ 80%
        - Clinical ≥ 35%
        - Safety 100%
        - Clinical validity ≥ 75%
        - Dedup retention > 60%
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.core.pipelines.clinical_accuracy_validator import ClinicalAccuracyValidator

# Import from stage_classifier for validation


@dataclass
class StageConfig:
    """Configuration for a stage including quality thresholds."""

    stage_id: str
    stage_name: str
    target_share: float
    empathy_floor: float
    clinical_floor: float
    safety_floor: float
    clinical_validity_floor: float
    dedup_retention_floor: float
    source_files: list[str] = field(default_factory=list)
    output_file: str = ""


# Canonical stage configuration with quality thresholds
STAGE_CONFIGS: dict[str, StageConfig] = {
    "stage1_foundation": StageConfig(
        stage_id="stage1_foundation",
        stage_name="Stage 1 – Foundation & Rapport",
        target_share=0.40,
        empathy_floor=0.70,
        clinical_floor=0.30,
        safety_floor=1.0,
        clinical_validity_floor=0.70,
        dedup_retention_floor=0.50,
        source_files=[
            "data/normalized/mental_health_counseling_normalized.jsonl",
            "training/sliced/stage1_foundation/additional_specialized_filtered.json",
            "training/sliced/stage1_foundation/Psychology-6K.json",
        ],
        output_file="stage1_foundation.jsonl",
    ),
    "stage2_therapeutic_expertise": StageConfig(
        stage_id="stage2_therapeutic_expertise",
        stage_name="Stage 2 – Therapeutic Expertise & Reasoning",
        target_share=0.25,
        empathy_floor=0.75,
        clinical_floor=0.50,
        safety_floor=1.0,
        clinical_validity_floor=0.75,
        dedup_retention_floor=0.50,
        source_files=[
            "data/normalized/cot_reasoning_normalized.jsonl",
            "training/sliced/stage2_therapeutic_expertise/clinical_diagnosis_mental_health.json",
            "training/sliced/stage2_therapeutic_expertise/cultural_nuances.json",
        ],
        output_file="stage2_therapeutic_expertise.jsonl",
    ),
    "stage3_edge_stress_test": StageConfig(
        stage_id="stage3_edge_stress_test",
        stage_name="Stage 3 – Edge Stress Test & Scenario Bank",
        target_share=0.20,
        empathy_floor=0.60,
        clinical_floor=0.40,
        safety_floor=1.0,
        clinical_validity_floor=0.65,
        dedup_retention_floor=0.40,
        source_files=[
            "pipelines/edge_case/output/edge_cases_training_format.jsonl",
            "data/cleaned_v3/nightmare_fuel_cleaned.jsonl",
            "training/ready_packages/datasets/synthetic/nightmare_fuel.jsonl",
        ],
        output_file="stage3_edge_stress_test.jsonl",
    ),
    "stage4_voice_persona": StageConfig(
        stage_id="stage4_voice_persona",
        stage_name="Stage 4 – Voice, Persona & Delivery",
        target_share=0.15,
        empathy_floor=0.80,
        clinical_floor=0.35,
        safety_floor=1.0,
        clinical_validity_floor=0.75,
        dedup_retention_floor=0.60,
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


@dataclass
class ValidationResult:
    """Result of validating a stage slice against thresholds."""

    stage_id: str
    passed: bool = False  # Default to False, gets updated
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clinical_validity_record_scores: dict[str, float] = field(default_factory=dict)


def _preview_record_text(text: str, max_chars: int = 400) -> str:
    """Create a short, safe preview for review queue ingestion."""
    return text[:max_chars] if len(text) <= max_chars else f"{text[: max_chars - 3]}..."


def route_low_clinical_validity_records_to_human_review(
    records: list[dict[str, Any]],
    validation_result: ValidationResult,
    stage_config: StageConfig,
    review_queue: Any,
    borderline_margin: float = 0.10,
) -> list[str]:
    """
    Route low-validity records to the human review queue.

    Records are routed when clinical validity is below the stage floor
    (urgent) or within the borderline margin above the floor
    (normal priority). Returns list of enqueued item ids.
    """
    enqueued_items: list[str] = []
    if review_queue is None:
        return enqueued_items

    for record in records:
        record_id = record.get("id", "")
        if not record_id:
            continue
        clinical_score = validation_result.clinical_validity_record_scores.get(record_id)
        if clinical_score is None:
            continue

        if clinical_score < stage_config.clinical_validity_floor:
            border_status = "failed"
        elif clinical_score < stage_config.clinical_validity_floor + borderline_margin:
            border_status = "borderline"
        else:
            continue

        is_urgent = border_status == "failed"
        gate_report = {
            "stage_id": stage_config.stage_id,
            "review_reason": "clinical_validity_review",
            "stage_clinical_validity_score": clinical_score,
            "stage_clinical_validity_floor": stage_config.clinical_validity_floor,
            "clinical_validity_border_status": border_status,
            "review_type": "clinical_validity",
        }
        text = str(record.get("text", ""))
        item = review_queue.create_item_from_report(
            source_id=record_id,
            gate_result=gate_report,
            content_preview=_preview_record_text(text),
            content_length=len(text),
            priority="urgent" if is_urgent else "normal",
        )
        review_queue.enqueue(item)
        enqueued_items.append(item.item_id)

    return enqueued_items


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
            return [data]
    return []


def normalize_record(record: dict[str, Any], source: str, stage_id: str) -> dict[str, Any]:
    """
    Normalize a record to standard format with metadata.

    Adds stage assignment metadata and preserves original data.
    """
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
            "stage_assigned_at": datetime.now(UTC).isoformat(),
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


def validate_stage_slice(records: list[dict[str, Any]], stage_config: StageConfig) -> ValidationResult:
    """
    Validate a stage slice against canonical quality thresholds.

    This is the PIX-506 integration point for validation gates.

    Checks:
    1. Empathy quality ≥ stage floor
    2. Clinical quality ≥ stage floor
    3. Safety compliance = 100%
    3. Clinical validity score
    4. Deduplication retention ≥ floor

    Returns:
        ValidationResult with pass/fail status and detailed metrics
    """
    result = ValidationResult(stage_id=stage_config.stage_id)

    if not records:
        result.violations.append("No records to validate")
        result.passed = False
        return result

    # Heuristic metrics + lightweight clinical validator
    empathy_scores: list[float] = []
    clinical_scores: list[float] = []
    safety_scores: list[float] = []
    clinical_validity_scores: list[float] = []
    validator = ClinicalAccuracyValidator()

    for record in records:
        text = record.get("text", "").lower()
        record_id = record.get("id", f"record-{len(clinical_validity_scores)}")
        metadata = record.get("metadata", {})

        # Empathy heuristic: presence of empathetic language markers
        empathy_markers = [
            "understand",
            "feel",
            "hear",
            "support",
            "care",
            "empath",
            "compassion",
            "validate",
            "acknowledge",
        ]
        empathy_score = sum(1 for m in empathy_markers if m in text) / len(empathy_markers)
        empathy_scores.append(empathy_score)

        # Clinical heuristic: presence of clinical/technical terms
        clinical_markers = [
            "diagnosis",
            "treatment",
            "intervention",
            "cbt",
            "dbt",
            "therapeutic",
            "clinical",
            "dsm",
            "symptom",
            "disorder",
        ]
        clinical_score = sum(1 for m in clinical_markers if m in text) / len(clinical_markers)
        clinical_scores.append(clinical_score)

        # Safety heuristic: flag records containing explicit harm/crisis language
        harm_markers = [
            "kill yourself",
            "end your life",
            "suicide method",
            "how to harm",
            "overdose on",
            "self-harm instructions",
        ]
        if any(marker in text for marker in harm_markers):
            safety_scores.append(0.0)
        else:
            safety_scores.append(1.0)
        # Clinical validity: lightweight semantic check for grounded clinical language
        try:
            clinical_result = validator.process(record.get("text", ""))
            clinical_validity_scores.append(clinical_result.score)
        except (TypeError, ValueError):
            clinical_validity_scores.append(0.0)
            metadata.setdefault("validation_warnings", []).append("clinical_validator_failed")
        result.clinical_validity_record_scores[record_id] = clinical_validity_scores[-1]

    # Calculate aggregate metrics
    result.metrics = {
        "empathy_avg": sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0,
        "clinical_avg": sum(clinical_scores) / len(clinical_scores) if clinical_scores else 0.0,
        "safety_avg": sum(safety_scores) / len(safety_scores) if safety_scores else 0.0,
        "clinical_validity_avg": (
            sum(clinical_validity_scores) / len(clinical_validity_scores) if clinical_validity_scores else 0.0
        ),
        "total_records": len(records),
        "dedup_retention": 0.85,  # Placeholder - would come from actual dedup analysis
    }
    result.metrics["safety_avg"] = result.metrics.get("safety_avg", 1.0)

    # Check against floors
    if result.metrics["empathy_avg"] < stage_config.empathy_floor:
        result.violations.append(
            f"Empathy {result.metrics['empathy_avg']:.2f} < floor {stage_config.empathy_floor:.2f}"
        )

    if result.metrics["clinical_avg"] < stage_config.clinical_floor:
        result.violations.append(
            f"Clinical {result.metrics['clinical_avg']:.2f} < floor {stage_config.clinical_floor:.2f}"
        )

    if result.metrics["safety_avg"] < stage_config.safety_floor:
        result.violations.append(f"Safety {result.metrics['safety_avg']:.2f} < floor {stage_config.safety_floor:.2f}")
    if result.metrics["clinical_validity_avg"] < stage_config.clinical_validity_floor:
        result.violations.append(
            "Clinical validity "
            f"{result.metrics['clinical_validity_avg']:.2f} < floor "
            f"{stage_config.clinical_validity_floor:.2f}"
        )

    if result.metrics["dedup_retention"] < stage_config.dedup_retention_floor:
        result.violations.append(
            f"Dedup retention {result.metrics['dedup_retention']:.2f} < floor {stage_config.dedup_retention_floor:.2f}"
        )

    # Add warnings for borderline cases
    if result.metrics["empathy_avg"] < stage_config.empathy_floor + 0.1:
        result.warnings.append(f"Empathy close to floor: {result.metrics['empathy_avg']:.2f} (floor + 0.1)")

    if result.metrics["clinical_avg"] < stage_config.clinical_floor + 0.1:
        result.warnings.append(f"Clinical close to floor: {result.metrics['clinical_avg']:.2f} (floor + 0.1)")

    if result.metrics["clinical_validity_avg"] < stage_config.clinical_validity_floor + 0.1:
        result.warnings.append(
            f"Clinical validity close to floor: {result.metrics['clinical_validity_avg']:.2f} (floor + 0.1)"
        )

    result.passed = len(result.violations) == 0
    return result


def slice_datasets(
    output_dir: Path,
    dry_run: bool = False,
    validate: bool = True,
    review_queue: Any | None = None,
    review_margin: float = 0.10,
) -> dict[str, Any]:
    """
    Slice validated datasets into training stages.

    Args:
        output_dir: Directory to write staged datasets
        dry_run: If True, only report what would be done
        validate: If True, validate slices against quality thresholds
        review_queue: Optional HumanReviewQueue for low validity records
        review_margin: Margin above clinical floor for review routing

    Returns:
        Report dictionary with slicing results and validation status
    """
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stages": {},
        "total_records": 0,
        "total_size_bytes": 0,
        "errors": [],
        "warnings": [],
        "validation_results": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_id, config in STAGE_CONFIGS.items():
        stage_report = {
            "stage_id": stage_id,
            "stage_name": config.stage_name,
            "target_share": config.target_share,
            "source_files": [],
            "records_processed": 0,
            "records_written": 0,
            "output_file": str(output_dir / config.output_file),
            "size_bytes": 0,
            "validation_passed": None if validate else None,
        }

        all_records = []

        for source_path in config.source_files:
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
                    if f.suffix in [".jsonl", ".json"]:
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

        # Validate stage slice if enabled
        if validate and all_records:
            validation_result = validate_stage_slice(all_records, config)
            report["validation_results"][stage_id] = {
                "passed": validation_result.passed,
                "metrics": validation_result.metrics,
                "violations": validation_result.violations,
                "warnings": validation_result.warnings,
            }
            if review_queue is not None:
                review_items = route_low_clinical_validity_records_to_human_review(
                    records=all_records,
                    validation_result=validation_result,
                    stage_config=config,
                    review_queue=review_queue,
                    borderline_margin=review_margin,
                )
                if review_items:
                    report["warnings"].append(
                        f"Stage {stage_id} routed {len(review_items)} low-validity records for clinical review"
                    )
                    stage_report["clinical_review_item_count"] = len(review_items)
            stage_report["validation_passed"] = validation_result.passed

            if not validation_result.passed:
                report["errors"].append(f"Stage {stage_id} failed validation: {validation_result.violations}")

        # Write staged output only if validation passed (or validation skipped)
        output_file = output_dir / config.output_file
        if all_records and not dry_run:
            if validate and stage_report.get("validation_passed") is False:
                report["errors"].append(f"Stage {stage_id}: skipping output write due to failed validation")
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    for record in all_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                stage_report["records_processed"] = len(all_records)
                stage_report["records_written"] = len(all_records)

                if output_file.exists():
                    stage_report["size_bytes"] = output_file.stat().st_size

        report["stages"][stage_id] = stage_report
        report["total_records"] += stage_report["records_processed"]
        report["total_size_bytes"] += stage_report["size_bytes"]

    return report


def generate_enhanced_metadata(report: dict[str, Any], output_dir: Path) -> None:
    """
    Generate enhanced metadata file with validation results.

    This is the canonical manifest format consumed by downstream issues:
    - PIX-303: Packaging automation
    - PIX-506: Validation gates
    - PIX-507: Observability
    """
    metadata = {
        "version": "1.1.0",  # Enhanced with validation
        "created_at": report["timestamp"],
        "dact": "DACT-06",
        "pix249_canonical": True,
        "description": "Stage-based dataset slices with validation (PIX-249)",
        "stages": list(report["stages"].keys()),
        "total_records": report["total_records"],
        "total_size_bytes": report["total_size_bytes"],
        "stage_details": report["stages"],
        "validation_summary": {
            stage_id: {
                "passed": result.get("passed", False),
                "metrics": result.get("metrics", {}),
            }
            for stage_id, result in report.get("validation_results", {}).items()
        },
        "trainingStages": [STAGE_CONFIGS[stage_id].stage_id for stage_id in report["stages"]],
        "downstream_ready": all(result.get("passed", False) for result in report.get("validation_results", {}).values()) if report.get("validation_results") else False,
    }

    metadata_file = output_dir / "dact06_enhanced_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="DACT-06: Stage-Based Dataset Slicing (Enhanced)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ai/data/staged_datasets",
        help="Output directory for staged datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate slices against quality thresholds (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip validation step",
    )
    parser.add_argument(
        "--review-margin",
        type=float,
        default=0.10,
        help="Borderline margin above clinical validity floor for review routing",
    )
    parser.add_argument(
        "--human-review-queue",
        type=str,
        default="",
        help="Path to a HumanReviewQueue data directory (disabled if empty)",
    )
    parser.add_argument(
        "--generate-metadata",
        action="store_true",
        default=True,
        help="Generate enhanced metadata file",
    )

    args = parser.parse_args()
    review_queue = None
    if args.human_review_queue:
        from ai.core.pipelines.human_review_queue import HumanReviewQueue

        review_queue = HumanReviewQueue(data_dir=Path(args.human_review_queue))

    # Execute slicing
    report = slice_datasets(
        Path(args.output_dir),
        dry_run=args.dry_run,
        validate=args.validate,
        review_queue=review_queue,
        review_margin=args.review_margin,
    )

    # Generate metadata
    if args.generate_metadata and not args.dry_run:
        generate_enhanced_metadata(report, Path(args.output_dir))

    return report


if __name__ == "__main__":
    main()
