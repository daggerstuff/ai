#!/usr/bin/env python3
"""
Dataset validation script that computes checksums, validates schemas,
and updates the dataset registry with validation results.
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from s3_client_helper import get_s3_client

S3_URI_PREFIX = "s3://"
S3_PREFIX_LENGTH = len(S3_URI_PREFIX)
JSON_TYPE_MARKERS = (
    "jsonl",
    "json",
    "conversation",
    "dpo",
    "forum",
    "reddit",
    "prompt",
    "rlhf",
    "emotion",
    "empathy",
    "deception",
    "reasoning",
    "knowledge",
    "perspective",
    "transcript",
    "synthetic",
    "profession",
    "curated",
    "edge",
    "seed",
    "embedding",
)
CSV_TYPE_MARKERS = ("csv", "table", "spreadsheet")
DEFAULT_SCHEMA_EXTENSIONS = {".jsonl", ".json", ".csv"}
S3_PATH_PARTS_WITH_CATEGORY = 3
S3_PATH_PARTS_WITH_SECTION = 2
S3_PATH_PARTS_MIN_LEN = 2
MIN_LOCAL_CSV_ROWS = 2

OTHER_REGISTRY_SECTIONS = [
    "rlhf_alignment",
    "emotion_recognition",
    "advanced_reasoning",
    "embeddings",
    "edge_case_sources",
    "voice_persona",
    "supplementary",
]
SIZE_WARNING_RATIO = 0.1
SIZE_FAILURE_RATIO = 0.5


class DatasetValidator:
    """Validates datasets and updates registry with results."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.s3_client = get_s3_client()

    @staticmethod
    def _strip_s3_prefix(dataset_path: str) -> tuple[str, str] | None:
        if not dataset_path.startswith(S3_URI_PREFIX):
            return None
        parts = dataset_path[S3_PREFIX_LENGTH:].split("/", 1)
        if len(parts) != S3_PATH_PARTS_MIN_LEN or not parts[1]:
            return None
        return parts[0], parts[1]

    @staticmethod
    def _normalize_dataset_size(raw_size: Any) -> float | None:
        if isinstance(raw_size, (int, float)) and raw_size > 0:
            return float(raw_size)
        if isinstance(raw_size, str):
            try:
                parsed_size = float(raw_size.strip())
            except ValueError:
                return None
            if parsed_size > 0:
                return parsed_size
        return None

    @staticmethod
    def _infer_expected_extensions(dataset_type: str) -> set[str]:
        normalized_type = dataset_type.lower()
        if any(marker in normalized_type for marker in JSON_TYPE_MARKERS):
            expected_exts = {".jsonl", ".json"}
        else:
            expected_exts = set(DEFAULT_SCHEMA_EXTENSIONS)
        if any(marker in normalized_type for marker in CSV_TYPE_MARKERS):
            expected_exts = {".csv"}
        return expected_exts

    @staticmethod
    def _validate_jsonl_schema(local_path: Path, errors: list[str]) -> bool:
        try:
            non_empty_lines = 0
            with open(local_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    non_empty_lines += 1
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSONL on data line {non_empty_lines}")
                        return False
            if non_empty_lines == 0:
                errors.append("JSONL dataset is empty")
                return False
        except Exception:
            errors.append("Failed to read JSONL file")
            return False
        return True

    @staticmethod
    def _validate_json_schema(local_path: Path, errors: list[str]) -> bool:
        try:
            with open(local_path, encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list) and not payload:
                errors.append("JSON dataset is empty")
                return False
        except json.JSONDecodeError:
            errors.append("Invalid JSON file")
            return False
        except Exception:
            errors.append("Failed to read JSON file")
            return False
        return True

    @staticmethod
    def _validate_csv_schema(local_path: Path, errors: list[str]) -> bool:
        try:
            with open(local_path, encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                errors.append("CSV dataset is empty")
                return False
            if not rows[0]:
                errors.append("CSV dataset has no header row")
                return False
        except Exception:
            errors.append("Failed to read CSV file")
            return False
        return True

    def _run_local_integrity_checks(
        self, dataset_path: str, file_size_bytes: int | None, warnings: list[str], errors: list[str]
    ) -> bool:
        integrity_ok = True
        extension = Path(dataset_path).suffix.lower()
        if extension == ".jsonl":
            try:
                with open(dataset_path, encoding="utf-8") as f:
                    for raw_line in f:
                        if raw_line.strip():
                            break
                    else:
                        errors.append("JSONL dataset has no non-empty rows")
                        integrity_ok = False
            except Exception:
                warnings.append("Failed to run JSONL completeness check")
                integrity_ok = False
        elif extension == ".csv":
            try:
                with open(dataset_path, encoding="utf-8", newline="") as f:
                    rows = list(csv.reader(f))
                if len(rows) < MIN_LOCAL_CSV_ROWS:
                    warnings.append(f"CSV dataset has fewer than {MIN_LOCAL_CSV_ROWS} rows (header only or empty)")
            except Exception:
                warnings.append("Failed to run CSV completeness check")
                integrity_ok = False

        if file_size_bytes == 0:
            errors.append("Dataset file is empty")
            integrity_ok = False

        return integrity_ok

    def _resolve_dataset_size_bytes(self, dataset_path: str, errors: list[str]) -> int | None:
        if dataset_path.startswith(S3_URI_PREFIX):
            s3_parts = self._strip_s3_prefix(dataset_path)
            if s3_parts is None:
                errors.append("Invalid S3 path")
                return None
            bucket, key = s3_parts
            try:
                head = self.s3_client.head_object(Bucket=bucket, Key=key)
                return int(head.get("ContentLength", 0))
            except ClientError as e:
                errors.append(f"Failed to access S3 object metadata: {e}")
                return None

        local_path = Path(dataset_path)
        if not local_path.exists():
            errors.append("Local dataset path does not exist")
            return None
        return local_path.stat().st_size

    @staticmethod
    def _evaluate_size_tolerance(
        observed_size_mb: float,
        expected_size_mb: float,
        errors: list[str],
        warnings: list[str],
    ) -> bool:
        size_delta = abs(observed_size_mb - expected_size_mb)
        if size_delta > (expected_size_mb * SIZE_FAILURE_RATIO):
            errors.append(
                f"Observed size {observed_size_mb:.2f}MB differs from expected "
                f"{expected_size_mb:.2f}MB by more than 50%"
            )
            return False
        if size_delta > (expected_size_mb * SIZE_WARNING_RATIO):
            warnings.append(
                f"Observed size {observed_size_mb:.2f}MB differs from expected "
                f"{expected_size_mb:.2f}MB by more than 10%"
            )
        return True

    def load_registry(self) -> dict[str, Any]:
        """Load the dataset registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Save the updated registry."""
        registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def calculate_checksum_s3(self, s3_path: str) -> str | None:
        """Calculate SHA256 checksum of S3 object."""
        try:
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                return None

            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            # Download and compute hash
            sha256_hash = hashlib.sha256()

            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            for chunk in response["Body"].iter_chunks(chunk_size=8192):
                sha256_hash.update(chunk)

            return sha256_hash.hexdigest()
        except ClientError:
            return None

    def calculate_checksum_local(self, file_path: Path) -> str | None:
        """Calculate SHA256 checksum of local file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return None

    def validate_schema(self, dataset_path: str, dataset_type: str) -> dict[str, Any]:
        """Validate dataset schema based on type."""
        errors: list[str] = []
        warnings: list[str] = []
        schema_valid = True

        dataset_path = dataset_path.strip()

        if not dataset_path:
            return {
                "schema_valid": False,
                "errors": ["Missing dataset path"],
                "warnings": [],
            }

        extension = Path(dataset_path.removeprefix("s3://").split("/", 1)[-1]).suffix.lower()
        if not extension:
            warnings.append("Dataset path has no file extension; schema checks are limited")

        expected_exts = self._infer_expected_extensions(dataset_type)
        if extension and extension not in expected_exts:
            schema_valid = False
            errors.append(f"Unexpected file extension '{extension}' for dataset type '{dataset_type}'")

        if dataset_path.startswith(S3_URI_PREFIX):
            if not extension:
                schema_valid = False
            return {"schema_valid": schema_valid, "errors": errors, "warnings": warnings}

        local_path = Path(dataset_path)
        if not local_path.exists():
            return {
                "schema_valid": False,
                "errors": ["Local dataset path does not exist"],
                "warnings": warnings,
            }

        if extension == ".jsonl":
            schema_valid = self._validate_jsonl_schema(local_path, errors) and schema_valid
        elif extension == ".json":
            schema_valid = self._validate_json_schema(local_path, errors) and schema_valid
        elif extension == ".csv":
            schema_valid = self._validate_csv_schema(local_path, errors) and schema_valid
        elif extension:
            warnings.append(f"No schema validation rule for extension '{extension}'")

        if not errors:
            warnings.append("No schema errors found")

        return {"schema_valid": schema_valid, "errors": errors, "warnings": warnings}

    def check_integrity(self, dataset_path: str, size_mb: Any) -> dict[str, Any]:
        """Check dataset integrity."""
        errors: list[str] = []
        warnings: list[str] = []
        integrity_check = True
        file_size_bytes = None

        dataset_path = dataset_path.strip()
        expected_size_mb = self._normalize_dataset_size(size_mb)

        if not dataset_path:
            return {
                "integrity_check": False,
                "file_size_bytes": None,
                "errors": ["Missing dataset path"],
                "warnings": [],
            }

        file_size_bytes = self._resolve_dataset_size_bytes(dataset_path, errors)

        if file_size_bytes is None:
            integrity_check = False
            return {
                "integrity_check": integrity_check,
                "file_size_bytes": file_size_bytes,
                "errors": errors,
                "warnings": warnings,
            }

        if expected_size_mb is not None:
            observed_size_mb = file_size_bytes / (1024 * 1024)
            integrity_check = self._evaluate_size_tolerance(observed_size_mb, expected_size_mb, errors, warnings)

        # Integrity completeness checks on local datasets based on lightweight parsing.
        if not dataset_path.startswith(S3_URI_PREFIX):
            integrity_check = (
                self._run_local_integrity_checks(dataset_path, file_size_bytes, warnings, errors) and integrity_check
            )

        if integrity_check and not errors and not warnings:
            warnings.append("No integrity issues detected")

        if file_size_bytes == 0:
            integrity_check = False
            if "Dataset file is empty" not in errors:
                errors.append("Dataset file is empty")

        return {
            "integrity_check": integrity_check,
            "file_size_bytes": file_size_bytes,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_dataset(self, _dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a single dataset and return validation results.

        Args:
            dataset_name: Name of the dataset
            dataset_entry: Dataset entry from registry

        Returns:
            Validation results dictionary
        """

        validation = dataset_entry.get("validation", {})
        validation.setdefault("validation_errors", [])
        validation.setdefault("validation_warnings", [])

        # Calculate checksum
        dataset_path = dataset_entry.get("path", "")
        if dataset_path.startswith("s3://"):
            checksum = self.calculate_checksum_s3(dataset_path)
        else:
            checksum = self.calculate_checksum_local(Path(dataset_path))

        if checksum:
            validation["checksum_sha256"] = checksum
            validation["integrity_check"] = True
        else:
            validation["integrity_check"] = False
            validation["validation_errors"].append("Failed to calculate checksum")

        expected_size_mb = dataset_entry.get("size_mb")
        integrity_result = self.check_integrity(dataset_path, expected_size_mb)
        validation["integrity_check"] = bool(validation.get("integrity_check") and integrity_result["integrity_check"])
        validation["file_size_bytes"] = integrity_result["file_size_bytes"]
        validation["validation_warnings"].extend(integrity_result["warnings"])
        validation["validation_errors"].extend(integrity_result["errors"])

        # Validate schema
        dataset_type = dataset_entry.get("type", "unknown")
        schema_result = self.validate_schema(dataset_path, dataset_type)
        validation["schema_valid"] = schema_result["schema_valid"]
        if schema_result["errors"]:
            validation["validation_errors"].extend(schema_result["errors"])

        # Update validation metadata
        validation["last_validated"] = datetime.now(UTC).isoformat() + "Z"
        validation["validation_status"] = (
            "validated" if validation.get("integrity_check") and validation.get("schema_valid") else "failed"
        )
        validation["requires_revalidation"] = False

        return validation

    @staticmethod
    def _collect_dataset_entries(
        section_name: str, section_data: Any, datasets_to_validate: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not isinstance(section_data, dict):
            return
        for dataset_name, dataset_entry in section_data.items():
            if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                datasets_to_validate.append((f"{section_name}.{dataset_name}", dataset_entry))

    @staticmethod
    def _apply_updated_validation(
        registry: dict[str, Any], dataset_path_key: str, updated_validation: dict[str, Any]
    ) -> None:
        parts = dataset_path_key.split(".")
        updates = {
            S3_PATH_PARTS_WITH_CATEGORY: lambda: registry["datasets"][parts[1]][parts[2]].update(
                {"validation": updated_validation}
            ),
            S3_PATH_PARTS_WITH_SECTION: lambda: registry[parts[0]][parts[1]].update({"validation": updated_validation}),
        }
        if len(parts) in updates:
            updates[len(parts)]()

    def validate_all_datasets(self, limit: int | None = None) -> dict[str, Any]:
        """
        Validate all datasets in the registry.

        Args:
            limit: Maximum number of datasets to validate (None = all)

        Returns:
            Statistics about validation
        """
        registry = self.load_registry()
        stats = {"total_validated": 0, "successful": 0, "failed": 0, "skipped": 0}

        datasets_to_validate: list[tuple[str, dict[str, Any]]] = []

        datasets_section = registry.get("datasets", {})
        if isinstance(datasets_section, dict):
            for category_name, category_data in datasets_section.items():
                if isinstance(category_data, dict):
                    self._collect_dataset_entries(f"datasets.{category_name}", category_data, datasets_to_validate)
        for section_name in OTHER_REGISTRY_SECTIONS:
            self._collect_dataset_entries(section_name, registry.get(section_name), datasets_to_validate)

        if limit:
            datasets_to_validate = datasets_to_validate[:limit]

        for dataset_path_key, dataset_entry in datasets_to_validate:
            if not dataset_entry.get("validation", {}).get("requires_revalidation", True):
                stats["skipped"] += 1
                continue

            try:
                updated_validation = self.validate_dataset(dataset_path_key, dataset_entry)
                self._apply_updated_validation(registry, dataset_path_key, updated_validation)

                stats["total_validated"] += 1
                if updated_validation["validation_status"] == "validated":
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
            except Exception:
                stats["failed"] += 1

        if "registry_statistics" in registry:
            registry["registry_statistics"]["validation_summary"] = {
                "validated": stats["successful"],
                "pending_validation": stats["skipped"],
                "validation_failed": stats["failed"],
            }

        self.save_registry(registry)

        return stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Validate datasets and update registry")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to validate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without updating registry",
    )

    args = parser.parse_args()

    validator = DatasetValidator(args.registry)
    validator.validate_all_datasets(limit=args.limit)


if __name__ == "__main__":
    main()
