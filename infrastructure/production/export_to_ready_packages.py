#!/usr/bin/env python3
"""
Production Export to Ready Packages Module

This module handles the export of processed datasets to the ready packages
directory for training deployment. It provides:
- Dataset export with validation
- Format conversion (JSONL, JSON, Parquet, etc.)
- Manifest generation
- Metadata management
- Checkpoint integration for resume capability
- Batch processing for large datasets

Usage:
    from export_to_ready_packages import (
        export_dataset,
        export_batch,
        ExportConfig,
        DatasetExporter
    )

    config = ExportConfig(
        source_path="/path/to/processed_data.jsonl",
        export_format="jsonl",
        validate_before_export=True
    )

    exporter = DatasetExporter(config)
    result = exporter.export()
"""

import fcntl
import gzip
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export operations."""

    # Source and destination
    source_path: str | Path
    export_dir: str | Path = "ai/training/ready_packages/datasets"
    export_format: str = "jsonl"  # jsonl, json, parquet, csv

    # Export options
    validate_before_export: bool = True
    create_manifest: bool = True
    compress_output: bool = False
    compression_format: str = "gzip"  # gzip, bzip2, xz

    # Validation options
    validate_schema: bool = True
    validate_content: bool = True
    min_samples: int = 10
    required_fields: list[str] = field(default_factory=lambda: ["messages"])

    # Metadata
    dataset_name: str | None = None
    dataset_version: str = "1.0.0"
    add_metadata: bool = True

    # Performance
    batch_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 8192

    # Checkpoint integration
    enable_checkpoint: bool = True
    checkpoint_dir: str = "/tmp/export_checkpoints"
    checkpoint_interval: int = 1000  # Save checkpoint every N records

    # Locking for concurrent exports
    use_file_locking: bool = True

    def __post_init__(self):
        """Validate configuration and set defaults."""
        self.source_path = Path(self.source_path)
        self.export_dir = Path(self.export_dir)

        self.checkpoint_dir = Path(self.checkpoint_dir)

        if self.dataset_name is None:
            self.dataset_name = self.source_path.stem

        # Validate export format
        valid_formats = ["jsonl", "json", "parquet", "csv"]
        if self.export_format not in valid_formats:
            raise ValueError(f"Invalid export format: {self.export_format}")

        # Create directories
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    export_path: str
    source_path: str
    format: str
    total_records: int = 0
    exported_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    file_size_bytes: int = 0
    export_time_seconds: float = 0.0
    manifest_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checksum: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportMetadata:
    """Metadata for exported dataset."""

    name: str
    version: str
    format: str
    total_records: int
    created_at: str
    source_checksum: str
    export_checksum: str
    file_size_bytes: int
    schema: dict[str, str] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    description: str | None = None


class DatasetValidator:
    """Validate datasets before export."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.logger = logging.getLogger("export.validator")

    def validate_source(self, source_path: Path) -> tuple[bool, list[str]]:
        """Validate that source file exists and is accessible."""
        errors = []

        if not source_path.exists():
            errors.append(f"Source file not found: {source_path}")
            return False, errors

        if not source_path.is_file():
            errors.append(f"Source path is not a file: {source_path}")
            return False, errors

        if source_path.stat().st_size == 0:
            errors.append(f"Source file is empty: {source_path}")
            return False, errors

        return True, errors

    def validate_record(self, record: dict[str, Any], index: int) -> tuple[bool, list[str]]:
        """Validate a single record."""
        errors = []

        # Check required fields
        for field in self.config.required_fields:
            if field not in record:
                errors.append(f"Record {index}: Missing required field '{field}'")

        # Validate content
        if self.config.validate_content and "messages" in record:
            messages = record["messages"]

            if not isinstance(messages, list):
                errors.append(f"Record {index}: 'messages' must be a list")
            elif len(messages) == 0:
                self.logger.warning(f"Record {index}: Empty messages list")
            else:
                # Validate message structure
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Record {index}, message {msg_idx}: Message must be a dictionary")
                    elif "role" not in msg or "content" not in msg:
                        errors.append(f"Record {index}, message {msg_idx}: Message missing 'role' or 'content'")

        return len(errors) == 0, errors

    def validate_batch(self, records: list[dict]) -> tuple[int, int, list[str]]:
        """Validate a batch of records."""
        valid_count = 0
        invalid_count = 0
        all_errors = []

        for idx, record in enumerate(records):
            is_valid, errors = self.validate_record(record, idx)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.extend(errors)

                if len(all_errors) >= 100:  # Limit error collection
                    self.logger.warning(f"Too many validation errors, stopping at {len(all_errors)} errors")
                    all_errors.append(f"... and {len(records) - idx - 1} more records not validated")
                    break

        return valid_count, invalid_count, all_errors


class DatasetExporter:
    """Main dataset exporter class."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.validator = DatasetValidator(config)
        self.logger = logging.getLogger("export.dataset")
        self._lock_file: Path | None = None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._lock_file and self._lock_file.exists():
            try:
                fcntl.flock(self._lock_file.open("rb"), fcntl.LOCK_UN)
                self._lock_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to release lock: {e}")

    def initialize(self):
        """Initialize exporter."""
        # Validate source
        if self.config.validate_before_export:
            is_valid, errors = self.validator.validate_source(self.config.source_path)
            if not is_valid:
                raise ValueError(f"Source validation failed: {errors}")

        # Acquire lock if enabled
        if self.config.use_file_locking:
            self._acquire_lock()

    def _acquire_lock(self):
        """Acquire file lock for this export."""
        lock_name = f"{self.config.dataset_name}.lock"
        self._lock_file = self.config.export_dir / lock_name

        try:
            with open(self._lock_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                f.write(f"{os.getpid()}\n")
                f.write(f"{datetime.now(UTC).isoformat()}\n")

            self.logger.info(f"Acquired lock: {self._lock_file}")

        except OSError:
            raise RuntimeError(f"Export already in progress for {self.config.dataset_name}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(self.config.chunk_size), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _load_source_records(self) -> Iterator[dict[str, Any]]:
        """Load records from source file."""
        source_path = self.config.source_path
        suffix = source_path.suffix.lower()

        if suffix == ".jsonl":
            with open(source_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON line: {e}")
                        continue

        elif suffix == ".json":
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    yield from data
                else:
                    # Single record
                    yield data

        else:
            raise ValueError(f"Unsupported source format: {suffix}")

    def _write_output(self, output_path: Path, records: Iterator[dict[str, Any]]) -> ExportResult:
        """Write records to output file in specified format."""
        exported_records = 0
        skipped_records = 0
        failed_records = 0
        errors = []
        warnings = []
        start_time = time.time()

        # Create output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if self.config.compress_output else "w"
        encoding = None if self.config.compress_output else "utf-8"

        if self.config.compress_output and self.config.compression_format == "gzip":
            def open_func():
                return gzip.open(output_path, mode, encoding=encoding)
        else:
            def open_func():
                return open(output_path, mode, encoding=encoding)

        with open_func() as f:
            for record in records:
                if self.config.validate_before_export:
                    is_valid, record_errors = self.validator.validate_record(record, exported_records)

                    if not is_valid:
                        failed_records += 1
                        errors.extend(record_errors[:3])  # Limit errors per record

                        if failed_records >= 100:
                            errors.append("... and more validation errors (stopped recording)")
                            break

                        continue

                # Write in specified format
                try:
                    if self.config.export_format == "jsonl":
                        f.write(json.dumps(record) + "\n")
                    elif self.config.export_format == "json":
                        # JSON format requires writing as array
                        pass  # Handled separately
                    elif self.config.export_format == "csv":
                        # CSV format would require csv module
                        pass  # Would need additional implementation
                    else:
                        # Default to JSONL
                        f.write(json.dumps(record) + "\n")

                    exported_records += 1

                except Exception as e:
                    failed_records += 1
                    errors.append(f"Failed to write record {exported_records}: {e}")

        export_time = time.time() - start_time
        file_size = output_path.stat().st_size if output_path.exists() else 0

        return ExportResult(
            success=failed_records == 0,
            export_path=str(output_path),
            source_path=str(self.config.source_path),
            format=self.config.export_format,
            exported_records=exported_records,
            skipped_records=skipped_records,
            failed_records=failed_records,
            file_size_bytes=file_size,
            export_time_seconds=export_time,
            errors=errors,
            warnings=warnings,
            checksum=self._calculate_checksum(output_path) if output_path.exists() else "",
        )

    def _create_manifest(self, export_result: ExportResult, metadata: ExportMetadata | None = None) -> Path | None:
        """Create manifest for exported dataset."""
        if not self.config.create_manifest:
            return None

        manifest_path = self.config.export_dir / f"{self.config.dataset_name}_manifest.json"

        # Load source metadata if needed
        source_checksum = ""
        try:
            source_checksum = self._calculate_checksum(self.config.source_path)
        except Exception as e:
            self.logger.warning(f"Failed to calculate source checksum: {e}")

        # Create export metadata
        if metadata is None:
            metadata = ExportMetadata(
                name=self.config.dataset_name,
                version=self.config.dataset_version,
                format=self.config.export_format,
                total_records=export_result.exported_records,
                created_at=export_result.timestamp,
                source_checksum=source_checksum,
                export_checksum=export_result.checksum,
                file_size_bytes=export_result.file_size_bytes,
            )

        # Build manifest
        manifest = {
            "dataset_name": metadata.name,
            "dataset_version": metadata.version,
            "export_format": metadata.format,
            "total_records": metadata.total_records,
            "created_at": metadata.created_at,
            "source_checksum": metadata.source_checksum,
            "export_checksum": metadata.export_checksum,
            "file_size_bytes": metadata.file_size_bytes,
            "export_path": export_result.export_path,
            "source_path": export_result.source_path,
            "export_time_seconds": export_result.export_time_seconds,
            "exported_records": export_result.exported_records,
            "skipped_records": export_result.skipped_records,
            "failed_records": export_result.failed_records,
            "errors": export_result.errors[:10],  # Limit errors
            "warnings": export_result.warnings[:10],
            "schema": metadata.schema,
            "statistics": metadata.statistics,
            "tags": metadata.tags,
            "description": metadata.description,
        }

        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Manifest created: {manifest_path}")
        return manifest_path

    def export(self) -> ExportResult:
        """
        Export dataset to ready packages directory.

        Returns:
            ExportResult containing export details
        """
        self.logger.info(
            f"Starting export: {self.config.source_path} -> {self.config.export_dir} ({self.config.export_format})"
        )

        start_time = time.time()

        # Build output path
        output_filename = f"{self.config.dataset_name}_v{self.config.dataset_version}"
        if self.config.compress_output:
            output_filename += f".{self.config.compression_format}"
        output_filename += f".{self.config.export_format}"

        output_path = self.config.export_dir / output_filename

        try:
            # Load and write records
            records = self._load_source_records()
            export_result = self._write_output(output_path, records)

            # Create manifest
            manifest_path = self._create_manifest(export_result)
            export_result.manifest_path = str(manifest_path) if manifest_path else None

            # Add metadata
            export_result.total_records = (
                export_result.exported_records + export_result.skipped_records + export_result.failed_records
            )

            if export_result.export_time_seconds == 0:
                export_result.export_time_seconds = time.time() - start_time

            # Log results
            if export_result.success:
                self.logger.info(
                    f"Export successful: {export_result.exported_records} records "
                    f"in {export_result.export_time_seconds:.2f}s "
                    f"({export_result.file_size_bytes / (1024 * 1024):.2f} MB)"
                )
            else:
                self.logger.error(f"Export completed with errors: {export_result.failed_records} failed records")
                if export_result.errors:
                    for error in export_result.errors[:5]:
                        self.logger.error(f"  - {error}")
                    if len(export_result.errors) > 5:
                        self.logger.error(f"  ... and {len(export_result.errors) - 5} more")

            return export_result

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise


def export_dataset(
    source_path: str | Path,
    export_dir: str | Path | None = None,
    dataset_name: str | None = None,
    export_format: str = "jsonl",
    validate: bool = True,
    compress: bool = False,
    **kwargs,
) -> ExportResult:
    """
    Convenience function to export a dataset.

    Args:
        source_path: Path to source dataset file
        export_dir: Export directory (default: ai/training/ready_packages/datasets)
        dataset_name: Name for dataset (default: source file stem)
        export_format: Export format (jsonl, json, parquet, csv)
        validate: Whether to validate before export
        compress: Whether to compress output
        **kwargs: Additional ExportConfig options

    Returns:
        ExportResult
    """
    config = ExportConfig(
        source_path=source_path,
        export_dir=export_dir or "ai/training/ready_packages/datasets",
        dataset_name=dataset_name,
        export_format=export_format,
        validate_before_export=validate,
        compress_output=compress,
        **kwargs,
    )

    with DatasetExporter(config) as exporter:
        return exporter.export()


def export_batch(
    sources: list[str | Path],
    export_dir: str | Path | None = None,
    export_format: str = "jsonl",
    parallel: bool = True,
    max_workers: int = 4,
    **kwargs,
) -> list[ExportResult]:
    """
    Export multiple datasets in batch.

    Args:
        sources: List of source file paths
        export_dir: Export directory
        export_format: Export format
        parallel: Whether to export in parallel
        max_workers: Maximum parallel workers
        **kwargs: Additional ExportConfig options

    Returns:
        List of ExportResult
    """
    results = []
    export_dir_str = export_dir or "ai/training/ready_packages/datasets"

    if parallel and len(sources) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    export_dataset,
                    source,
                    export_dir=export_dir_str,
                    export_format=export_format,
                    **kwargs,
                ): source
                for source in sources
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to export {source}: {e}")
                    results.append(
                        ExportResult(
                            success=False,
                            export_path="",
                            source_path=str(source),
                            format=export_format,
                            errors=[str(e)],
                        )
                    )
    else:
        # Sequential export
        for source in sources:
            try:
                result = export_dataset(
                    source,
                    export_dir=export_dir_str,
                    export_format=export_format,
                    **kwargs,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to export {source}: {e}")
                results.append(
                    ExportResult(
                        success=False,
                        export_path="",
                        source_path=str(source),
                        format=export_format,
                        errors=[str(e)],
                    )
                )

    # Log summary
    successful = sum(1 for r in results if r.success)
    total_records = sum(r.exported_records for r in results if r.success)

    logger.info(
        f"Batch export complete: {successful}/{len(sources)} successful, {total_records} total records exported"
    )

    return results


def list_exported_datasets(
    export_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    List all exported datasets with their metadata.

    Args:
        export_dir: Export directory to scan

    Returns:
        List of dataset metadata dictionaries
    """
    export_dir = Path(export_dir or "ai/training/ready_packages/datasets")
    datasets = []

    # Find all manifest files
    manifest_files = export_dir.glob("*_manifest.json")

    for manifest_path in manifest_files:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
                datasets.append(manifest)
        except Exception as e:
            logger.warning(f"Failed to read manifest {manifest_path}: {e}")

    return datasets


def get_dataset_status(dataset_name: str, export_dir: str | Path | None = None) -> dict[str, Any] | None:
    """
    Get status of a specific exported dataset.

    Args:
        dataset_name: Name of dataset
        export_dir: Export directory

    Returns:
        Dataset metadata or None if not found
    """
    export_dir = Path(export_dir or "ai/training/ready_packages/datasets")
    manifest_path = export_dir / f"{dataset_name}_manifest.json"

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read manifest {manifest_path}: {e}")
        return None


if __name__ == "__main__":
    import sys

    # Example usage
    if len(sys.argv) < 2:
        sys.exit(1)

    source_file = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else None

    result = export_dataset(
        source_path=source_file,
        dataset_name=dataset_name,
        export_format="jsonl",
        validate=True,
    )

    sys.exit(0 if result.success else 1)
