"""
Format Converter for Dataset Standardization

Converts between different dataset formats (JSON, JSONL, CSV, Parquet)
with schema validation and data integrity checks.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


class DataFormat(Enum):
    """Supported data formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TSV = "tsv"


@dataclass
class ConversionResult:
    """Result of format conversion."""

    success: bool
    input_format: DataFormat
    output_format: DataFormat
    input_file: str
    output_file: str
    records_processed: int
    errors: list[str]
    conversion_time: float
    timestamp: datetime


class FormatConverter:
    """Converts between different dataset formats."""

    def __init__(self):
        self.logger = get_logger(__name__)
        logger.info("FormatConverter initialized")

    def convert_file(
        self,
        input_path: str,
        output_path: str,
        input_format: DataFormat | None = None,
        output_format: DataFormat | None = None,
    ) -> ConversionResult:
        """Convert file from one format to another."""
        start_time = datetime.now()

        # Auto-detect formats if not provided
        if input_format is None:
            input_format = self._detect_format(input_path)
        if output_format is None:
            output_format = self._detect_format(output_path)

        logger.info(
            f"Converting {input_path} from {input_format.value} to {output_format.value}"
        )

        errors = []
        records_processed = 0

        try:
            # Load data
            data = self._load_data(input_path, input_format)
            records_processed = len(data) if isinstance(data, list) else 1

            # Save data
            self._save_data(data, output_path, output_format)

            conversion_time = (datetime.now() - start_time).total_seconds()

            result = ConversionResult(
                success=True,
                input_format=input_format,
                output_format=output_format,
                input_file=input_path,
                output_file=output_path,
                records_processed=records_processed,
                errors=errors,
                conversion_time=conversion_time,
                timestamp=start_time,
            )

            logger.info(
                f"Conversion completed: {records_processed} records in {conversion_time:.2f}s"
            )
            return result

        except Exception as e:
            errors.append(str(e))
            conversion_time = (datetime.now() - start_time).total_seconds()

            result = ConversionResult(
                success=False,
                input_format=input_format,
                output_format=output_format,
                input_file=input_path,
                output_file=output_path,
                records_processed=records_processed,
                errors=errors,
                conversion_time=conversion_time,
                timestamp=start_time,
            )

            logger.error(f"Conversion failed: {e}")
            return result

    def _detect_format(self, file_path: str) -> DataFormat:
        """Auto-detect file format from extension."""
        suffix = Path(file_path).suffix.lower()

        format_map = {
            ".json": DataFormat.JSON,
            ".jsonl": DataFormat.JSONL,
            ".csv": DataFormat.CSV,
            ".parquet": DataFormat.PARQUET,
            ".tsv": DataFormat.TSV,
        }

        return format_map.get(suffix, DataFormat.JSON)

    def _load_data(
        self, file_path: str, format_type: DataFormat
    ) -> list[dict] | dict:
        """Load data from file based on format."""
        if format_type == DataFormat.JSON:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)

        elif format_type == DataFormat.JSONL:
            data = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data

        elif format_type == DataFormat.CSV:
            df = pd.read_csv(file_path)
            return df.to_dict("records")

        elif format_type == DataFormat.TSV:
            df = pd.read_csv(file_path, sep="\t")
            return df.to_dict("records")

        elif format_type == DataFormat.PARQUET:
            df = pd.read_parquet(file_path)
            return df.to_dict("records")

        else:
            raise ValueError(f"Unsupported input format: {format_type}")

    def _save_data(
        self, data: list[dict] | dict, file_path: str, format_type: DataFormat
    ):
        """Save data to file based on format."""
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        if format_type == DataFormat.JSON:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format_type == DataFormat.JSONL:
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]

            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        elif format_type == DataFormat.CSV:
            # Convert to DataFrame
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

            df.to_csv(file_path, index=False, encoding="utf-8")

        elif format_type == DataFormat.TSV:
            # Convert to DataFrame
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

            df.to_csv(file_path, sep="\t", index=False, encoding="utf-8")

        elif format_type == DataFormat.PARQUET:
            # Convert to DataFrame
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

            df.to_parquet(file_path, index=False)

        else:
            raise ValueError(f"Unsupported output format: {format_type}")

    def batch_convert(
        self, conversions: list[dict[str, Any]]
    ) -> list[ConversionResult]:
        """Perform batch conversions."""
        results = []

        for conversion in conversions:
            result = self.convert_file(
                input_path=conversion["input_path"],
                output_path=conversion["output_path"],
                input_format=conversion.get("input_format"),
                output_format=conversion.get("output_format"),
            )
            results.append(result)

        return results

    def validate_conversion(
        self, original_path: str, converted_path: str
    ) -> dict[str, Any]:
        """Validate that conversion preserved data integrity."""
        try:
            # Load both files
            original_format = self._detect_format(original_path)
            converted_format = self._detect_format(converted_path)

            original_data = self._load_data(original_path, original_format)
            converted_data = self._load_data(converted_path, converted_format)

            # Normalize to lists for comparison
            if not isinstance(original_data, list):
                original_data = [original_data]
            if not isinstance(converted_data, list):
                converted_data = [converted_data]

            # Compare record counts
            record_count_match = len(original_data) == len(converted_data)

            # Sample comparison (first few records)
            sample_size = min(5, len(original_data), len(converted_data))
            sample_match = True

            for i in range(sample_size):
                if original_data[i] != converted_data[i]:
                    sample_match = False
                    break

            return {
                "valid": record_count_match and sample_match,
                "original_records": len(original_data),
                "converted_records": len(converted_data),
                "record_count_match": record_count_match,
                "sample_match": sample_match,
                "validation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat(),
            }

    def get_format_info(self, file_path: str) -> dict[str, Any]:
        """Get information about a file's format and structure."""
        try:
            format_type = self._detect_format(file_path)
            file_size = Path(file_path).stat().st_size

            # Load and analyze data
            data = self._load_data(file_path, format_type)

            if isinstance(data, list):
                record_count = len(data)
                sample_record = data[0] if data else {}
            else:
                record_count = 1
                sample_record = data

            # Analyze structure
            fields = (
                list(sample_record.keys()) if isinstance(sample_record, dict) else []
            )

            return {
                "file_path": file_path,
                "format": format_type.value,
                "file_size_bytes": file_size,
                "record_count": record_count,
                "fields": fields,
                "sample_record": sample_record,
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }


# Example usage
if __name__ == "__main__":
    converter = FormatConverter()

    # Create test data
    test_data = [
        {"id": 1, "content": "Hello world", "role": "user"},
        {"id": 2, "content": "Hi there!", "role": "assistant"},
    ]

    # Save as JSON
    with open("test_input.json", "w") as f:
        json.dump(test_data, f)

    try:
        # Convert JSON to JSONL
        result = converter.convert_file("test_input.json", "test_output.jsonl")

        # Validate conversion
        validation = converter.validate_conversion(
            "test_input.json", "test_output.jsonl"
        )

        # Get format info
        info = converter.get_format_info("test_input.json")

    finally:
        # Clean up
        import os

        for file in ["test_input.json", "test_output.jsonl"]:
            if os.path.exists(file):
                os.remove(file)
