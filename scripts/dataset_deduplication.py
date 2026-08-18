#!/usr/bin/env python3
"""
Dataset deduplication script that identifies and removes duplicate entries
within and across datasets.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from s3_client_helper import get_s3_client

S3_PATH_MIN_PARTS = 2
DEFAULT_REGISTRY_PATH = "/home/vivi/pixelated/ai/config/dataset_registry.json"
DEDUP_SUFFIX = "_deduped"
JSONL_EXTENSION = ".jsonl"
JSON_EXTENSION = ".json"
S3_SCHEME_PREFIX = "s3://"
MIN_DATASET_NAME_PARTS = 2
DATASETS_SECTION_PARTS = 3
DATASET_SECTION_NAMES = [
    "rlhf_alignment",
    "emotion_recognition",
    "advanced_reasoning",
    "embeddings",
    "edge_case_sources",
    "voice_persona",
    "supplementary",
]


class DatasetDeduplicator:
    """Identifies and removes duplicate entries in datasets."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.s3_client = get_s3_client()
        self.dedup_stats = {
            "datasets_checked": 0,
            "total_records": 0,
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "duplicate_groups": 0,
            "bytes_saved": 0,
        }

    def load_registry(self) -> dict[str, Any]:
        """Load the dataset registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Save the updated registry."""
        registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def compute_content_hash(self, content: str) -> str:
        """Compute hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def compute_record_hash(self, record: dict[str, Any], key_fields: list[str] | None = None) -> str:
        """
        Compute hash of a record for deduplication.

        Args:
            record: Dataset record
            key_fields: Fields to use for hashing (None = all fields)

        Returns:
            Hash string
        """
        if key_fields:
            hash_content = json.dumps({k: record.get(k) for k in key_fields if k in record}, sort_keys=True)
        else:
            # Exclude metadata fields
            exclude_fields = {"_id", "_hash", "_timestamp", "_source"}
            hash_content = json.dumps(
                {k: v for k, v in record.items() if k not in exclude_fields},
                sort_keys=True,
            )

        return self.compute_content_hash(hash_content)

    def load_dataset_from_s3(self, s3_path: str) -> list[dict[str, Any]]:
        """Load dataset from S3."""
        try:
            if not s3_path.startswith("s3://"):
                return []

            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")

            # Detect format
            if s3_path.endswith(".jsonl"):
                records = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
            elif s3_path.endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # Could be wrapped in a key
                    for key in ["data", "records", "samples", "items"]:
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break
                    else:
                        records = [data]
                else:
                    records = []
            else:
                records = []

            return records
        except Exception:
            return []

    def find_duplicates_in_dataset(
        self, records: list[dict[str, Any]], key_fields: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
        """
        Find duplicate records within a dataset.

        Args:
            records: List of records to deduplicate
            key_fields: Fields to use for deduplication

        Returns:
            Tuple of (deduplicated records, duplicate groups)
        """
        seen_hashes: dict[str, int] = {}
        duplicate_groups: dict[str, list[int]] = defaultdict(list)
        deduplicated = []

        for idx, record in enumerate(records):
            record_hash = self.compute_record_hash(record, key_fields)

            if record_hash in seen_hashes:
                # Duplicate found
                duplicate_groups[record_hash].append(idx)
            else:
                seen_hashes[record_hash] = len(deduplicated)
                deduplicated.append(record)

        return deduplicated, dict(duplicate_groups)

    def _write_deduplicated_to_s3(
        self, source_s3_path: str, deduplicated_records: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Write deduplicated records to a sibling S3 object."""
        if not source_s3_path.startswith(S3_SCHEME_PREFIX):
            raise ValueError("Source path is not an S3 path")

        parts = source_s3_path[len(S3_SCHEME_PREFIX) :].split("/", 1)
        if len(parts) < S3_PATH_MIN_PARTS:
            raise ValueError("Invalid S3 path format")

        bucket = parts[0]
        source_key = parts[1]
        source_key_lower = source_key.lower()
        output_key = (
            source_key[: -len(JSONL_EXTENSION)] + f"{DEDUP_SUFFIX}{JSONL_EXTENSION}"
            if source_key_lower.endswith(JSONL_EXTENSION)
            else source_key[: -len(JSON_EXTENSION)] + f"{DEDUP_SUFFIX}{JSON_EXTENSION}"
            if source_key_lower.endswith(JSON_EXTENSION)
            else source_key + f"{DEDUP_SUFFIX}{JSON_EXTENSION}"
        )

        if source_key_lower.endswith(JSONL_EXTENSION):
            content = "\n".join(json.dumps(record, ensure_ascii=False) for record in deduplicated_records)
            if content:
                content += "\n"
            content_type = "application/x-ndjson"
            body = content.encode("utf-8")
        else:
            body = json.dumps(deduplicated_records, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"

        self.s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=body,
            ContentType=content_type,
        )

        return f"s3://{bucket}/{output_key}", len(body)

    def find_duplicates_across_datasets(
        self, datasets: dict[str, list[dict[str, Any]]], key_fields: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Find duplicate records across multiple datasets.

        Args:
            datasets: Dict mapping dataset names to their records
            key_fields: Fields to use for deduplication

        Returns:
            Dict mapping hash to list of dataset:record_index
        """
        hash_to_locations: dict[str, list[str]] = defaultdict(list)

        for dataset_name, records in datasets.items():
            for idx, record in enumerate(records):
                record_hash = self.compute_record_hash(record, key_fields)
                hash_to_locations[record_hash].append(f"{dataset_name}:{idx}")

        # Filter to only hashes that appear in multiple datasets
        return {
            h: locations
            for h, locations in hash_to_locations.items()
            if len({loc.split(":")[0] for loc in locations}) > 1
        }

    def deduplicate_dataset(
        self,
        _dataset_name: str,
        dataset_entry: dict[str, Any],
        key_fields: list[str] | None = None,
        write_output: bool = False,
    ) -> dict[str, Any]:
        """
        Deduplicate a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_entry: Dataset entry from registry
            key_fields: Fields to use for deduplication
            write_output: Whether to write deduplicated dataset back

        Returns:
            Deduplication statistics
        """

        s3_path = dataset_entry.get("path", "")
        if not s3_path:
            return {"error": "No S3 path found"}

        # Load dataset
        records = self.load_dataset_from_s3(s3_path)
        if not records:
            return {"error": "Failed to load dataset or dataset is empty"}

        original_count = len(records)

        # Find duplicates
        deduplicated, duplicate_groups = self.find_duplicates_in_dataset(records, key_fields)

        duplicate_count = original_count - len(deduplicated)

        stats = {
            "original_count": original_count,
            "deduplicated_count": len(deduplicated),
            "duplicates_found": duplicate_count,
            "duplicate_groups": len(duplicate_groups),
            "deduplication_ratio": round(duplicate_count / original_count * 100, 2) if original_count > 0 else 0,
        }

        # Write deduplicated dataset if requested
        if write_output and duplicate_count > 0:
            try:
                output_path, output_bytes = self._write_deduplicated_to_s3(s3_path, deduplicated)
                stats["output_path"] = output_path
                stats["output_bytes"] = output_bytes
            except Exception as exc:
                stats["output_error"] = str(exc)

        self.dedup_stats["datasets_checked"] += 1
        self.dedup_stats["total_records"] += original_count
        self.dedup_stats["duplicates_found"] += duplicate_count

        if duplicate_count > 0:
            self.dedup_stats["duplicate_groups"] += len(duplicate_groups)

        return stats

    def _collect_nested_datasets(self, section_data: dict[str, Any], prefix: str) -> list[tuple[str, dict[str, Any]]]:
        if not isinstance(section_data, dict):
            return []
        return [
            (f"{prefix}.{dataset_name}", dataset_entry)
            for dataset_name, dataset_entry in section_data.items()
            if isinstance(dataset_entry, dict) and "path" in dataset_entry
        ]

    def _collect_deduplication_datasets(self, registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        datasets: list[tuple[str, dict[str, Any]]] = []

        category_data = registry.get("datasets")
        if isinstance(category_data, dict):
            for category_name, category_entries in category_data.items():
                if isinstance(category_entries, dict):
                    datasets.extend(
                        self._collect_nested_datasets(
                            category_entries,
                            f"datasets.{category_name}",
                        )
                    )

        for section_name in DATASET_SECTION_NAMES:
            datasets.extend(self._collect_nested_datasets(registry.get(section_name, {}), section_name))

        return datasets

    def _lookup_dataset_entry(self, registry: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
        parts = dataset_name.split(".")
        if len(parts) < MIN_DATASET_NAME_PARTS:
            return None

        if parts[0] == "datasets" and len(parts) == DATASETS_SECTION_PARTS:
            return registry.get("datasets", {}).get(parts[1], {}).get(parts[2], {})

        return registry.get(parts[0], {}).get(parts[1], {})

    def deduplicate_all_datasets(
        self,
        limit: int | None = None,
        key_fields: list[str] | None = None,
        write_output: bool = False,
    ) -> dict[str, Any]:
        """
        Deduplicate all datasets in the registry.

        Args:
            limit: Maximum number of datasets to process
            key_fields: Fields to use for deduplication
            write_output: Whether to write deduplicated datasets

        Returns:
            Overall deduplication statistics
        """
        registry = self.load_registry()

        datasets_to_process = self._collect_deduplication_datasets(registry)
        if limit:
            datasets_to_process = datasets_to_process[:limit]

        # Process each dataset
        results = {}
        for dataset_path_key, dataset_entry in datasets_to_process:
            try:
                dedup_result = self.deduplicate_dataset(
                    dataset_path_key,
                    dataset_entry,
                    key_fields=key_fields,
                    write_output=write_output,
                )
                results[dataset_path_key] = dedup_result

                # Update registry with dedup metadata
                if "quality_metrics" in dataset_entry:
                    dataset_entry["quality_metrics"]["duplicate_count"] = dedup_result.get("duplicates_found", 0)
                    dataset_entry["quality_metrics"]["deduplication_ratio"] = dedup_result.get("deduplication_ratio", 0)

            except Exception as e:
                results[dataset_path_key] = {"error": str(e)}

        # Save updated registry
        self.save_registry(registry)

        return {"statistics": self.dedup_stats, "dataset_results": results}

    def find_cross_dataset_duplicates(
        self,
        dataset_names: list[str] | None = None,
        key_fields: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Find duplicates across multiple datasets.

        Args:
            dataset_names: Specific datasets to check (None = all)
            key_fields: Fields to use for deduplication
            limit: Maximum number of datasets to compare

        Returns:
            Cross-dataset duplicate analysis
        """
        registry = self.load_registry()

        # Collect datasets to compare
        if dataset_names:
            datasets_to_load = {}
            for name in dataset_names:
                entry = self._lookup_dataset_entry(registry, name)
                if entry and "path" in entry:
                    datasets_to_load[name] = entry["path"]
        else:
            all_datasets = self._collect_deduplication_datasets(registry)
            if limit:
                all_datasets = all_datasets[:limit]
            datasets_to_load = {name: entry.get("path") for name, entry in all_datasets}

        # Load all datasets
        loaded_datasets = {}
        for name, path in {k: v for k, v in datasets_to_load.items() if v}.items():
            records = self.load_dataset_from_s3(path)
            if records:
                loaded_datasets[name] = records

        # Find cross-dataset duplicates
        cross_duplicates = self.find_duplicates_across_datasets(loaded_datasets, key_fields)

        # Analyze results
        duplicate_stats = {
            "total_unique_hashes": len(cross_duplicates),
            "datasets_analyzed": len(loaded_datasets),
            "total_records_analyzed": sum(len(r) for r in loaded_datasets.values()),
            "cross_dataset_duplicate_count": sum(len(locs) for locs in cross_duplicates.values()),
        }

        # Group by which datasets share duplicates
        dataset_overlap = defaultdict(int)
        for locations in cross_duplicates.values():
            datasets_involved = tuple(sorted({loc.split(":")[0] for loc in locations}))
            dataset_overlap[datasets_involved] += 1

        duplicate_stats["dataset_overlaps"] = {
            "->".join(k): v for k, v in sorted(dataset_overlap.items(), key=lambda x: x[1], reverse=True)[:20]
        }

        return {
            "statistics": duplicate_stats,
            "sample_duplicates": dict(list(cross_duplicates.items())[:10]),  # Show first 10
        }


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Deduplicate datasets")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(DEFAULT_REGISTRY_PATH),
        help="Path to dataset registry",
    )
    parser.add_argument(
        "--action",
        choices=["dedupe", "cross-dataset", "both"],
        default="dedupe",
        help="Deduplication action to perform",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to process")
    parser.add_argument(
        "--key-fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to use for deduplication",
    )
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="Write deduplicated datasets back to storage",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of specific datasets to analyze (for cross-dataset)",
    )

    args = parser.parse_args()

    # Parse key fields
    key_fields = None
    if args.key_fields:
        key_fields = [f.strip() for f in args.key_fields.split(",")]

    deduplicator = DatasetDeduplicator(args.registry)

    if args.action in ["dedupe", "both"]:
        results = deduplicator.deduplicate_all_datasets(
            limit=args.limit, key_fields=key_fields, write_output=args.write_output
        )

        stats = results["statistics"]

        for _dataset_name, result in results["dataset_results"].items():
            if "error" in result:
                pass
            else:
                pass

    if args.action in ["cross-dataset", "both"]:
        dataset_names = None
        if args.datasets:
            dataset_names = [d.strip() for d in args.datasets.split(",")]

        results = deduplicator.find_cross_dataset_duplicates(
            dataset_names=dataset_names, key_fields=key_fields, limit=args.limit
        )

        stats = results["statistics"]

        if stats.get("dataset_overlaps"):
            for _overlap, _count in list(stats["dataset_overlaps"].items())[:10]:
                pass


if __name__ == "__main__":
    main()
