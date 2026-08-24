#!/usr/bin/env python3
"""
Dataset sync verification script that checks consistency between
source (Google Drive) and canonical (S3) storage.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from s3_client_helper import get_s3_client


class DatasetSyncVerifier:
    """Verifies sync status between GDrive and S3."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.s3_client = get_s3_client()

    def load_registry(self) -> dict[str, Any]:
        """Load the dataset registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Save the updated registry."""
        registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def check_s3_exists(self, s3_path: str) -> bool:
        """Check if object exists in S3."""
        try:
            if not s3_path.startswith("s3://"):
                return False

            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def check_gdrive_exists(self, gdrive_path: str) -> bool:
        """Check if file exists in Google Drive mount."""
        if not gdrive_path:
            return False

        # Handle both direct paths and gdrive mount paths
        if gdrive_path.startswith("/mnt/gdrive/"):
            return Path(gdrive_path).exists()

        # Try prepending mount point
        full_path = Path("/mnt/gdrive/datasets") / gdrive_path.lstrip("/")
        return full_path.exists()

    def get_s3_metadata(self, s3_path: str) -> dict[str, Any] | None:
        """Get metadata for S3 object."""
        try:
            if not s3_path.startswith("s3://"):
                return None

            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            response = self.s3_client.head_object(Bucket=bucket, Key=key)

            return {
                "size": response.get("ContentLength"),
                "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
                "etag": response.get("ETag", "").strip('"'),
                "content_type": response.get("ContentType"),
            }
        except ClientError:
            return None

    def get_gdrive_metadata(self, gdrive_path: str) -> dict[str, Any] | None:
        """Get metadata for Google Drive file."""
        try:
            if not gdrive_path:
                return None

            # Resolve path
            if gdrive_path.startswith("/mnt/gdrive/"):
                path = Path(gdrive_path)
            else:
                path = Path("/mnt/gdrive/datasets") / gdrive_path.lstrip("/")

            if not path.exists():
                return None

            stat = path.stat()

            return {
                "size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(path),
            }
        except Exception:
            return None

    def compare_sources(self, s3_path: str, gdrive_path: str) -> dict[str, Any]:
        """Compare S3 and GDrive versions of dataset."""
        comparison = {
            "s3_exists": False,
            "gdrive_exists": False,
            "sizes_match": None,
            "s3_metadata": None,
            "gdrive_metadata": None,
            "discrepancies": [],
        }

        # Check S3
        comparison["s3_exists"] = self.check_s3_exists(s3_path)
        if comparison["s3_exists"]:
            comparison["s3_metadata"] = self.get_s3_metadata(s3_path)

        # Check GDrive
        comparison["gdrive_exists"] = self.check_gdrive_exists(gdrive_path)
        if comparison["gdrive_exists"]:
            comparison["gdrive_metadata"] = self.get_gdrive_metadata(gdrive_path)

        # Compare sizes if both exist
        if comparison["s3_exists"] and comparison["gdrive_exists"]:
            s3_size = comparison["s3_metadata"].get("size") if comparison["s3_metadata"] else None
            gdrive_size = comparison["gdrive_metadata"].get("size") if comparison["gdrive_metadata"] else None

            if s3_size and gdrive_size:
                comparison["sizes_match"] = s3_size == gdrive_size

                if not comparison["sizes_match"]:
                    comparison["discrepancies"].append(
                        {
                            "type": "size_mismatch",
                            "s3_size": s3_size,
                            "gdrive_size": gdrive_size,
                            "difference": abs(s3_size - gdrive_size),
                        }
                    )

        # Check for missing sources
        if not comparison["s3_exists"]:
            comparison["discrepancies"].append(
                {
                    "type": "missing_s3",
                    "message": "Dataset not found in S3 canonical storage",
                }
            )

        if not comparison["gdrive_exists"]:
            comparison["discrepancies"].append(
                {
                    "type": "missing_gdrive",
                    "message": "Dataset not found in Google Drive source",
                }
            )

        return comparison

    def verify_dataset_sync(self, dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Verify sync status for a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_entry: Dataset entry from registry

        Returns:
            Sync status dictionary
        """

        s3_path = dataset_entry.get("path", "")
        fallback_paths = dataset_entry.get("fallback_paths", {})
        gdrive_path = fallback_paths.get("gdrive", fallback_paths.get("gdrive_dir", ""))

        comparison = self.compare_sources(s3_path, gdrive_path)

        return {
            "gdrive_synced": comparison["gdrive_exists"],
            "s3_synced": comparison["s3_exists"],
            "last_sync_timestamp": datetime.now(UTC).isoformat() + "Z",
            "sync_discrepancies": comparison["discrepancies"],
            "sync_verified": comparison["s3_exists"]
            and (not comparison["gdrive_exists"] or comparison.get("sizes_match", False)),
        }

    def verify_all_datasets(self, limit: int | None = None) -> dict[str, Any]:
        """
        Verify sync status for all datasets.

        Args:
            limit: Maximum number of datasets to verify

        Returns:
            Statistics about sync verification
        """
        registry = self.load_registry()
        stats = {
            "total_verified": 0,
            "in_sync": 0,
            "out_of_sync": 0,
            "missing_s3": 0,
            "missing_gdrive": 0,
        }

        # Collect all datasets
        datasets_to_verify = []

        # From datasets section
        if "datasets" in registry:
            for category_name, category_data in registry["datasets"].items():
                if isinstance(category_data, dict):
                    for dataset_name, dataset_entry in category_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_verify.append(
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
                            datasets_to_verify.append((f"{section_name}.{dataset_name}", dataset_entry))

        # Apply limit
        if limit:
            datasets_to_verify = datasets_to_verify[:limit]

        # Verify each dataset
        for dataset_path_key, dataset_entry in datasets_to_verify:
            try:
                sync_status = self.verify_dataset_sync(dataset_path_key, dataset_entry)

                # Update registry
                parts = dataset_path_key.split(".")
                if len(parts) == 3:
                    if "sync_status" not in registry["datasets"][parts[1]][parts[2]]:
                        registry["datasets"][parts[1]][parts[2]]["sync_status"] = {}
                    registry["datasets"][parts[1]][parts[2]]["sync_status"] = sync_status
                elif len(parts) == 2:
                    if "sync_status" not in registry[parts[0]][parts[1]]:
                        registry[parts[0]][parts[1]]["sync_status"] = {}
                    registry[parts[0]][parts[1]]["sync_status"] = sync_status

                stats["total_verified"] += 1

                if sync_status["sync_verified"]:
                    stats["in_sync"] += 1
                else:
                    stats["out_of_sync"] += 1

                if not sync_status["s3_synced"]:
                    stats["missing_s3"] += 1

                if not sync_status["gdrive_synced"]:
                    stats["missing_gdrive"] += 1

            except Exception:
                stats["out_of_sync"] += 1

        # Update registry statistics
        if "registry_statistics" in registry:
            registry["registry_statistics"]["sync_summary"] = {
                "in_sync": stats["in_sync"],
                "out_of_sync": stats["out_of_sync"],
                "sync_unknown": 0,
            }

        # Update source_staging sync info
        if "source_staging" in registry:
            registry["source_staging"]["sync_configuration"]["last_sync_check"] = datetime.now(UTC).isoformat() + "Z"
            registry["source_staging"]["sync_configuration"]["automated_verification"] = True

        # Save updated registry
        self.save_registry(registry)

        return stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Verify dataset sync status")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/configs/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to verify")

    args = parser.parse_args()

    verifier = DatasetSyncVerifier(args.registry)
    verifier.verify_all_datasets(limit=args.limit)


if __name__ == "__main__":
    main()
