"""
Dataset Validation Script for Phase 2 Baseline Validation.

Walks all dataset directories, validates file structure, counts
conversations, checks role consistency, and outputs a manifest.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DatasetValidator")

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = [
    REPO_ROOT / "ai/data/acquired_datasets",
    REPO_ROOT / "ai/data/transcripts/ingested",
    REPO_ROOT / "ai/training/ready_packages/data/generated",
]

VALID_ROLES_THERAPEUTIC = {"client", "therapist"}
VALID_ROLES_CHATML = {"system", "user", "assistant"}


def validate_json_array_file(file_path: Path) -> dict[str, Any]:
    """Validate a JSON file containing an array of conversations."""
    errors: list[str] = []
    conversation_count = 0
    empty_messages = 0
    role_set: set[str] = set()

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "file": str(file_path.relative_to(REPO_ROOT)),
            "format": "json_array",
            "valid": False,
            "errors": [f"Invalid JSON: {e}"],
            "conversations": 0,
        }

    if not isinstance(data, list):
        return {
            "file": str(file_path.relative_to(REPO_ROOT)),
            "format": "json_array",
            "valid": False,
            "errors": ["Root element is not a list"],
            "conversations": 0,
        }

    for i, record in enumerate(data):
        conv = record.get("conversation", record.get("messages", []))
        if not conv:
            errors.append(f"Record {i}: no 'conversation' or 'messages' field")
            continue

        conversation_count += 1
        for msg in conv:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_set.add(role)
            if not content or not content.strip():
                empty_messages += 1

    return {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "format": "json_array",
        "size_bytes": file_path.stat().st_size,
        "valid": not errors,
        "conversations": conversation_count,
        "roles_found": sorted(role_set),
        "empty_messages": empty_messages,
        "errors": errors[:10],
    }


def validate_jsonl_file(file_path: Path) -> dict[str, Any]:
    """Validate a JSONL file (one JSON record per line)."""
    errors: list[str] = []
    conversation_count = 0
    empty_messages = 0
    role_set: set[str] = set()
    line_count = 0

    try:
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: invalid JSON")
                    continue

                conv = record.get("messages", record.get("conversation", []))
                if not conv:
                    errors.append(f"Line {line_num}: no 'messages' or 'conversation'")
                    continue

                conversation_count += 1
                for msg in conv:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    role_set.add(role)
                    if not content or not content.strip():
                        empty_messages += 1

    except Exception as e:
        return {
            "file": str(file_path.relative_to(REPO_ROOT)),
            "format": "jsonl",
            "valid": False,
            "errors": [f"Read error: {e}"],
            "conversations": 0,
        }

    return {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "format": "jsonl",
        "size_bytes": file_path.stat().st_size,
        "valid": not errors,
        "conversations": conversation_count,
        "total_lines": line_count,
        "roles_found": sorted(role_set),
        "empty_messages": empty_messages,
        "errors": errors[:10],
    }


def validate_transcript_md(file_path: Path) -> dict[str, Any]:
    """Validate an ingested transcript Markdown file."""
    errors: list[str] = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {
            "file": str(file_path.relative_to(REPO_ROOT)),
            "format": "transcript_md",
            "valid": False,
            "errors": [f"Read error: {e}"],
        }

    has_title = content.startswith("# ")
    has_transcript_section = "## Transcript" in content
    word_count = len(content.split())

    if not has_title:
        errors.append("Missing H1 title")
    if not has_transcript_section:
        errors.append("Missing '## Transcript' section")
    if word_count < 50:
        errors.append(f"Very short transcript ({word_count} words)")

    return {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "format": "transcript_md",
        "size_bytes": file_path.stat().st_size,
        "valid": not errors,
        "word_count": word_count,
        "has_title": has_title,
        "has_transcript_section": has_transcript_section,
        "errors": errors,
    }


def run_validation() -> dict[str, Any]:
    """Run full validation across all dataset directories."""
    results: list[dict[str, Any]] = []
    total_errors = 0
    total_conversations = 0
    total_transcripts = 0

    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            logger.warning(f"Directory not found, skipping: {scan_dir}")
            continue

        logger.info(f"Scanning: {scan_dir.relative_to(REPO_ROOT)}")

        for file_path in sorted(scan_dir.iterdir()):
            if file_path.is_dir():
                continue

            if file_path.suffix == ".json" and file_path.name not in ("acquisition_summary.json",):
                # Skip stats/report files, only validate actual datasets
                if "_stats" in file_path.name or "_report" in file_path.name:
                    continue
                result = validate_json_array_file(file_path)
                results.append(result)
                total_conversations += result.get("conversations", 0)

            elif file_path.suffix == ".jsonl":
                result = validate_jsonl_file(file_path)
                results.append(result)
                total_conversations += result.get("conversations", 0)

            elif file_path.suffix == ".md" and file_path.name != "TRAINING_DATASET_GUIDE.md":
                result = validate_transcript_md(file_path)
                results.append(result)
                total_transcripts += 1

            if results and not results[-1].get("valid", True):
                total_errors += 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_files_scanned": len(results),
            "total_conversations": total_conversations,
            "total_transcripts": total_transcripts,
            "files_with_errors": total_errors,
            "scan_directories": [str(d.relative_to(REPO_ROOT)) for d in SCAN_DIRS if d.exists()],
        },
        "files": results,
    }


def main():
    logger.info("Starting Phase 2 Dataset Validation...")
    manifest = run_validation()

    output_path = REPO_ROOT / "ai/data/dataset_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = manifest["summary"]
    logger.info(
        f"Validation complete: "
        f"{summary['total_files_scanned']} files, "
        f"{summary['total_conversations']} conversations, "
        f"{summary['total_transcripts']} transcripts, "
        f"{summary['files_with_errors']} errors"
    )
    logger.info(f"Manifest written to: {output_path}")

    if summary["files_with_errors"] > 0:
        logger.warning("Some files have validation errors. Check manifest for details.")
        for file_result in manifest["files"]:
            if not file_result.get("valid", True):
                logger.warning(f"  ❌ {file_result['file']}: {file_result['errors'][:3]}")

    return 0 if summary["files_with_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
