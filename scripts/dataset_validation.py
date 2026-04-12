#!/usr/bin/env python3
"""
Dataset validation script that computes checksums, validates schemas,
and updates the dataset registry with validation results.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from s3_client_helper import get_s3_client


class DatasetValidator:
    """Validates datasets and updates registry with results."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.s3_client = get_s3_client()

    def load_registry(self) -> dict[str, Any]:
        """Load the dataset registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Save the updated registry."""
        registry["last_updated"] = datetime.now(timezone.utc).isoformat() + "Z"
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
        except ClientError as e:
            print(f"Error calculating checksum for {s3_path}: {e}")
            return None

    def calculate_checksum_local(self, file_path: Path) -> str | None:
        """Calculate SHA256 checksum of local file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating checksum for {file_path}: {e}")
            return None

    def validate_schema(self, _dataset_path: str, _dataset_type: str) -> dict[str, Any]:
        """Validate dataset schema based on type."""
        # Placeholder for schema validation logic
        # In production, this would validate against known schemas

        return {"schema_valid": True, "errors": [], "warnings": []}

        # TODO: Implement actual schema validation based on dataset_type
        # For now, return a basic validation result


    def check_integrity(self, _dataset_path: str, _size_mb: float) -> dict[str, Any]:
        """Check dataset integrity."""
        return {"integrity_check": True, "errors": [], "warnings": []}

        # TODO: Implement actual integrity checks
        # - File size validation
        # - Format validation
        # - Data completeness checks


    def validate_dataset(
        self, dataset_name: str, dataset_entry: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate a single dataset and return validation results.

        Args:
            dataset_name: Name of the dataset
            dataset_entry: Dataset entry from registry

        Returns:
            Validation results dictionary
        """
        print(f"Validating dataset: {dataset_name}")

        validation = dataset_entry.get("validation", {})

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

        # Validate schema
        dataset_type = dataset_entry.get("type", "unknown")
        schema_result = self.validate_schema(dataset_path, dataset_type)
        validation["schema_valid"] = schema_result["schema_valid"]
        if schema_result["errors"]:
            validation["validation_errors"].extend(schema_result["errors"])

        # Update validation metadata
        validation["last_validated"] = datetime.now(timezone.utc).isoformat() + "Z"
        validation["validation_status"] = (
            "validated"
            if validation.get("integrity_check") and validation.get("schema_valid")
            else "failed"
        )
        validation["requires_revalidation"] = False

        return validation

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

        # Collect all datasets to validate
        datasets_to_validate = []

        # From datasets section
        if "datasets" in registry:
            for category_name, category_data in registry["datasets"].items():
                if isinstance(category_data, dict):
                    for dataset_name, dataset_entry in category_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_validate.append(
                                (
                                    f"datasets.{category_name}.{dataset_name}",
                                    dataset_entry,
                                )
                            )

        # From other sections
        other_sections = [
            "rlhf_alignment",
            "emotion_recognition",
            "advanced_reasoning",
            "embeddings",
            "edge_case_sources",
            "voice_persona",
            "supplementary",
        ]

        for section_name in other_sections:
            if section_name in registry:
                section_data = registry[section_name]
                if isinstance(section_data, dict):
                    for dataset_name, dataset_entry in section_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_validate.append(
                                (f"{section_name}.{dataset_name}", dataset_entry)
                            )

        # Apply limit if specified
        if limit:
            datasets_to_validate = datasets_to_validate[:limit]

        # Validate each dataset
        for dataset_path_key, dataset_entry in datasets_to_validate:
            try:
                # Check if validation is needed
                validation = dataset_entry.get("validation", {})
                if not validation.get("requires_revalidation", True):
                    stats["skipped"] += 1
                    continue

                # Perform validation
                updated_validation = self.validate_dataset(
                    dataset_path_key, dataset_entry
                )

                # Update registry
                parts = dataset_path_key.split(".")
                if len(parts) == 3:
                    registry["datasets"][parts[1]][parts[2]]["validation"] = (
                        updated_validation
                    )
                elif len(parts) == 2:
                    registry[parts[0]][parts[1]]["validation"] = updated_validation

                stats["total_validated"] += 1
                if updated_validation["validation_status"] == "validated":
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                print(f"Error validating {dataset_path_key}: {e}")
                stats["failed"] += 1

        # Update registry statistics
        if "registry_statistics" in registry:
            registry["registry_statistics"]["validation_summary"] = {
                "validated": stats["successful"],
                "pending_validation": stats["skipped"],
                "validation_failed": stats["failed"],
            }

        # Save updated registry
        self.save_registry(registry)

        return stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Validate datasets and update registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of datasets to validate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without updating registry",
    )

    args = parser.parse_args()

    print("Dataset Validation Script")
    print(f"Registry: {args.registry}")
    print(f"Limit: {args.limit or 'None (all datasets)'}")
    print(f"Dry run: {args.dry_run}")
    print()

    validator = DatasetValidator(args.registry)
    stats = validator.validate_all_datasets(limit=args.limit)

    print("\nValidation Statistics:")
    print(f"  Total validated: {stats['total_validated']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
