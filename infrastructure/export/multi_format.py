#!/usr/bin/env python3
"""
Multi-Format Export Module for Pixelated Empathy AI

Provides comprehensive dataset export capabilities in multiple formats:
- JSONL (JSON Lines): Streaming-friendly, one record per line
- JSON: Standard JSON array format
- CSV: Tabular format with proper escaping
- Parquet: Columnar storage for analytics

This module supports the COULD priority requirement from the PRD for
multi-format export capabilities, enabling integration with various
data science and analytics workflows.

Usage:
    from ai.infrastructure.export.multi_format import MultiFormatExporter, ExportFormat

    exporter = MultiFormatExporter()

    # Export to JSONL
    exporter.export_dataset(
        dataset_path="data/input.jsonl",
        output_path="data/output.jsonl",
        format=ExportFormat.JSONL
    )

    # Export to Parquet for analytics
    exporter.export_dataset(
        dataset_path="data/input.jsonl",
        output_path="data/output.parquet",
        format=ExportFormat.PARQUET
    )
"""

import csv
import json
import logging
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class ExportConfig:
    """Configuration for multi-format export operations."""

    def __init__(
        self,
        format: ExportFormat = ExportFormat.JSONL,
        compression: str | None = None,
        batch_size: int = 1000,
        include_metadata: bool = True,
        flatten_nested: bool = False,
        csv_delimiter: str = ",",
        csv_quotechar: str = '"',
        parquet_compression: str = "snappy",
        validate_schema: bool = True,
    ):
        """
        Initialize export configuration.

        Args:
            format: Target export format
            compression: Compression type (gzip, bz2, xz, or None)
            batch_size: Number of records to process per batch
            include_metadata: Whether to include export metadata
            flatten_nested: Whether to flatten nested JSON structures for CSV
            csv_delimiter: CSV field delimiter
            csv_quotechar: CSV quote character
            parquet_compression: Parquet compression codec
            validate_schema: Whether to validate schema before export
        """
        self.format = format
        self.compression = compression
        self.batch_size = batch_size
        self.include_metadata = include_metadata
        self.flatten_nested = flatten_nested
        self.csv_delimiter = csv_delimiter
        self.csv_quotechar = csv_quotechar
        self.parquet_compression = parquet_compression
        self.validate_schema = validate_schema


class ExportResult:
    """Result of an export operation."""

    def __init__(
        self,
        success: bool,
        output_path: Path,
        format: ExportFormat,
        records_exported: int,
        bytes_written: int,
        errors: list[str],
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize export result.

        Args:
            success: Whether export completed successfully
            output_path: Path to exported file
            format: Export format used
            records_exported: Number of records exported
            bytes_written: Total bytes written
            errors: List of error messages (if any)
            metadata: Optional export metadata
        """
        self.success = success
        self.output_path = output_path
        self.format = format
        self.records_exported = records_exported
        self.bytes_written = bytes_written
        self.errors = errors
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path),
            "format": self.format.value,
            "records_exported": self.records_exported,
            "bytes_written": self.bytes_written,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class MultiFormatExporter:
    """
    Multi-format dataset exporter supporting JSONL, JSON, CSV, and Parquet.

    Provides efficient streaming export for large datasets with proper
    error handling and progress tracking.
    """

    SUPPORTED_FORMATS = {
        ExportFormat.JSONL: ".jsonl",
        ExportFormat.JSON: ".json",
        ExportFormat.CSV: ".csv",
        ExportFormat.PARQUET: ".parquet",
    }

    def __init__(self, config: ExportConfig | None = None):
        """
        Initialize the multi-format exporter.

        Args:
            config: Export configuration (uses defaults if None)
        """
        self.config = config or ExportConfig()
        self.logger = logging.getLogger("multi_format_exporter")

    def export_dataset(
        self,
        dataset_path: str | Path,
        output_path: str | Path,
        format: ExportFormat | None = None,
        config: ExportConfig | None = None,
    ) -> ExportResult:
        """
        Export a dataset to the specified format.

        Args:
            dataset_path: Path to input dataset (JSONL)
            output_path: Path for output file
            format: Target format (uses config default if None)
            config: Optional override configuration

        Returns:
            ExportResult with operation details
        """
        config = config or self.config
        format = format or config.format

        dataset_path = Path(dataset_path)
        output_path = Path(output_path)

        # Validate input
        if not dataset_path.exists():
            return ExportResult(
                success=False,
                output_path=output_path,
                format=format,
                records_exported=0,
                bytes_written=0,
                errors=[f"Input file not found: {dataset_path}"],
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Route to appropriate exporter
        try:
            if format == ExportFormat.JSONL:
                return self._export_jsonl(dataset_path, output_path, config)
            if format == ExportFormat.JSON:
                return self._export_json(dataset_path, output_path, config)
            if format == ExportFormat.CSV:
                return self._export_csv(dataset_path, output_path, config)
            if format == ExportFormat.PARQUET:
                return self._export_parquet(dataset_path, output_path, config)
            return ExportResult(
                success=False,
                output_path=output_path,
                format=format,
                records_exported=0,
                bytes_written=0,
                errors=[f"Unsupported format: {format}"],
            )
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                output_path=output_path,
                format=format,
                records_exported=0,
                bytes_written=0,
                errors=[str(e)],
            )

    def _load_dataset(self, dataset_path: Path) -> Iterator[dict[str, Any]]:
        """
        Load dataset as a streaming iterator.

        Args:
            dataset_path: Path to JSONL file

        Yields:
            Dictionary records
        """
        with open(dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Line {line_num}: JSON parse error - {e}")

    def _export_jsonl(self, dataset_path: Path, output_path: Path, _config: ExportConfig) -> ExportResult:
        """Export to JSONL format."""
        records_exported = 0
        errors = []

        with open(output_path, "w", encoding="utf-8") as f:
            for record in self._load_dataset(dataset_path):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_exported += 1

        bytes_written = output_path.stat().st_size

        return ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.JSONL,
            records_exported=records_exported,
            bytes_written=bytes_written,
            errors=errors,
            metadata={"format": "JSON Lines"},
        )

    def _export_json(self, dataset_path: Path, output_path: Path, _config: ExportConfig) -> ExportResult:
        """Export to JSON array format."""
        records = list(self._load_dataset(dataset_path))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        bytes_written = output_path.stat().st_size

        return ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.JSON,
            records_exported=len(records),
            bytes_written=bytes_written,
            errors=[],
            metadata={"format": "JSON Array"},
        )

    def _flatten_record(self, record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """
        Flatten a nested dictionary for CSV export.

        Args:
            record: Nested dictionary
            prefix: Key prefix for nested fields

        Returns:
            Flattened dictionary
        """
        items = []
        for key, value in record.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(self._flatten_record(value, f"{new_key}_").items())
            elif isinstance(value, list):
                items.append((new_key, json.dumps(value)))
            else:
                items.append((new_key, value))
        return dict(items)

    def _export_csv(self, dataset_path: Path, output_path: Path, config: ExportConfig) -> ExportResult:
        """Export to CSV format."""
        records = list(self._load_dataset(dataset_path))

        if not records:
            return ExportResult(
                success=False,
                output_path=output_path,
                format=ExportFormat.CSV,
                records_exported=0,
                bytes_written=0,
                errors=["No records to export"],
            )

        # Flatten records if configured
        if config.flatten_nested:
            records = [self._flatten_record(r) for r in records]

        # Get all fieldnames from all records
        fieldnames = set()
        for record in records:
            fieldnames.update(record.keys())
        fieldnames = sorted(fieldnames)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=config.csv_delimiter,
                quotechar=config.csv_quotechar,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(records)

        bytes_written = output_path.stat().st_size

        return ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.CSV,
            records_exported=len(records),
            bytes_written=bytes_written,
            errors=[],
            metadata={
                "format": "CSV",
                "fieldnames": fieldnames,
                "flattened": config.flatten_nested,
            },
        )

    def _export_parquet(self, dataset_path: Path, output_path: Path, config: ExportConfig) -> ExportResult:
        """Export to Parquet format."""
        try:
            pass
        except ImportError:
            return ExportResult(
                success=False,
                output_path=output_path,
                format=ExportFormat.PARQUET,
                records_exported=0,
                bytes_written=0,
                errors=["pandas required for Parquet export: uv add pandas"],
            )

        try:
            pass
        except ImportError:
            return ExportResult(
                success=False,
                output_path=output_path,
                format=ExportFormat.PARQUET,
                records_exported=0,
                bytes_written=0,
                errors=["pyarrow required for Parquet export: uv add pyarrow"],
            )

        records = list(self._load_dataset(dataset_path))

        if not records:
            return ExportResult(
                success=False,
                output_path=output_path,
                format=ExportFormat.PARQUET,
                records_exported=0,
                bytes_written=0,
                errors=["No records to export"],
            )

        # Convert to DataFrame and export
        df = pd.DataFrame(records)
        df.to_parquet(
            output_path,
            compression=config.parquet_compression,
            index=False,
        )

        bytes_written = output_path.stat().st_size

        return ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.PARQUET,
            records_exported=len(records),
            bytes_written=bytes_written,
            errors=[],
            metadata={
                "format": "Parquet",
                "compression": config.parquet_compression,
                "columns": list(df.columns),
            },
        )

    def convert_format(
        self,
        input_path: str | Path,
        output_path: str | Path,
        input_format: ExportFormat,
        output_format: ExportFormat,
    ) -> ExportResult:
        """
        Convert between formats (assumes input is valid format).

        Args:
            input_path: Path to input file
            output_path: Path for output file
            input_format: Format of input (must be JSONL currently)
            output_format: Desired output format

        Returns:
            ExportResult with operation details
        """
        if input_format != ExportFormat.JSONL:
            return ExportResult(
                success=False,
                output_path=Path(output_path),
                format=output_format,
                records_exported=0,
                bytes_written=0,
                errors=["Currently only JSONL input is supported"],
            )

        return self.export_dataset(input_path, output_path, output_format)

    def batch_export(
        self,
        datasets: list[dict[str, Any]],
        output_dir: str | Path,
        base_name: str = "dataset",
    ) -> list[ExportResult]:
        """
        Export a dataset to all supported formats.

        Args:
            datasets: List of dataset configurations with 'path' and optional 'config'
            output_dir: Directory for output files
            base_name: Base filename for outputs

        Returns:
            List of ExportResult for each format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for fmt in ExportFormat:
            output_path = output_dir / f"{base_name}{self.SUPPORTED_FORMATS[fmt]}"

            for dataset_config in datasets:
                dataset_path = dataset_config["path"]
                config = dataset_config.get("config", self.config)

                result = self.export_dataset(
                    dataset_path=dataset_path,
                    output_path=output_path,
                    format=fmt,
                    config=config,
                )
                results.append(result)

                if result.success:
                    self.logger.info(f"Exported {result.records_exported} records to {fmt.value}: {output_path}")
                else:
                    self.logger.error(f"Failed to export to {fmt.value}: {result.errors}")

        return results


# Convenience functions for common use cases


def export_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> ExportResult:
    """Convenience function to export dataset to JSONL."""
    exporter = MultiFormatExporter(config=ExportConfig(format=ExportFormat.JSONL))
    return exporter.export_dataset(input_path, output_path, ExportFormat.JSONL)


def export_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    flatten_nested: bool = True,
) -> ExportResult:
    """Convenience function to export dataset to CSV."""
    config = ExportConfig(format=ExportFormat.CSV, flatten_nested=flatten_nested)
    exporter = MultiFormatExporter(config=config)
    return exporter.export_dataset(input_path, output_path, ExportFormat.CSV)


def export_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    compression: str = "snappy",
) -> ExportResult:
    """Convenience function to export dataset to Parquet."""
    config = ExportConfig(format=ExportFormat.PARQUET, parquet_compression=compression)
    exporter = MultiFormatExporter(config=config)
    return exporter.export_dataset(input_path, output_path, ExportFormat.PARQUET)


if __name__ == "__main__":
    # Example usage

    # Example: Create a sample dataset and export it

    # Create sample data
    sample_data = [
        {"id": 1, "text": "Hello world", "label": "greeting", "confidence": 0.95},
        {"id": 2, "text": "Sample message", "label": "generic", "confidence": 0.87},
        {"id": 3, "text": "Test data", "label": "test", "confidence": 0.99},
    ]

    # Write sample JSONL
    sample_path = Path("/tmp/sample_dataset.jsonl")
    with open(sample_path, "w") as f:
        for record in sample_data:
            f.write(json.dumps(record) + "\n")

    # Export to all formats
    exporter = MultiFormatExporter()

    for fmt in ExportFormat:
        output_path = Path(f"/tmp/sample_dataset{exporter.SUPPORTED_FORMATS[fmt]}")
        result = exporter.export_dataset(sample_path, output_path, fmt)

        if result.success:
            pass
        else:
            pass
