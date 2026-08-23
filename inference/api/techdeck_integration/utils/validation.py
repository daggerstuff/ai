"""
Validation utilities for TechDeck Flask service.

This module provides comprehensive input validation, data sanitization,
and security measures for HIPAA++ compliance.
"""

import json
import mimetypes
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
from werkzeug.utils import secure_filename

MAX_PIPELINE_TIMEOUT_SECONDS = 3600
MAX_VALIDATION_STRING_ARGS = 2
VALIDATION_PATTERNS = MappingProxyType(
    {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "alphanumeric": r"^[a-zA-Z0-9_-]+$",
        "numeric": r"^\d+$",
        "float": r"^\d*\.?\d+$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "datetime_iso": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$",
    }
)
SENSITIVE_PATTERNS = (
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
    r"\b\d+\.\d+\.\d+\.\d+\b",  # IP address
)
COMPILED_SENSITIVE_PATTERNS = tuple(re.compile(pattern) for pattern in SENSITIVE_PATTERNS)
RE_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class ValidationError(Exception):
    """Custom validation error with detailed information."""

    def __init__(self, message: str, field: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code or "VALIDATION_ERROR"


class InputValidator:
    """
    Comprehensive input validator with security measures.

    Provides validation for various input types with HIPAA++
    compliance and security considerations.
    """

    # Common patterns for validation
    _COMPILED_SENSITIVE_PATTERNS = COMPILED_SENSITIVE_PATTERNS
    _RE_CONTROL_CHARS = RE_CONTROL_CHARS
    SENSITIVE_FIELDS = ("ssn", "password", "secret", "key")

    # ⚡ Bolt Optimization: Precompile regex patterns globally to avoid the overhead of implicit compilation

    def __init__(self, max_string_length: int = 1000, max_file_size_mb: int = 100):
        """
        Initialize validator with configuration.

        Args:
            max_string_length: Maximum allowed string length
            max_file_size_mb: Maximum allowed file size in MB
        """
        self.max_string_length = max_string_length
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_string(
        self,
        value: str,
        field_name: str,
        min_length: int = 1,
        max_length: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Validate string input with comprehensive checks.

        Args:
            value: String value to validate
            field_name: Field name for error reporting
            min_length: Minimum string length
            max_length: Maximum string length (overrides default)
            pattern: Regex pattern to match
            allow_empty: Whether to allow empty strings

        Returns:
            Validated and sanitized string

        Raises:
            ValidationError: If validation fails
        """
        max_length = max_length or self.max_string_length

        allow_empty = kwargs.pop("allow_empty", False)
        pattern = kwargs.pop("pattern", None)
        if args:
            if len(args) > MAX_VALIDATION_STRING_ARGS:
                raise TypeError(
                    f"validate_string accepts at most {MAX_VALIDATION_STRING_ARGS} positional validation options"
                )
            if len(args) >= 1:
                pattern = args[0]
            if len(args) == MAX_VALIDATION_STRING_ARGS:
                allow_empty = args[1]
        if kwargs:
            unexpected_keys = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected argument(s): {unexpected_keys}")

        # Check if empty is allowed
        if allow_empty and not value:
            return value

        # Check type
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name)

        # Check length
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters long",
                field_name,
            )

        if len(value) > max_length:
            raise ValidationError(f"{field_name} must not exceed {max_length} characters", field_name)

        # Check pattern if provided
        if pattern and not re.match(pattern, value):
            raise ValidationError(f"{field_name} format is invalid", field_name)

        return self._sanitize_string(value)

    def validate_email(self, email: str, field_name: str = "email") -> str:
        """
        Validate email address format.

        Args:
            email: Email address to validate
            field_name: Field name for error reporting

        Returns:
            Validated email address

        Raises:
            ValidationError: If validation fails
        """
        return self.validate_string(email, field_name, pattern=VALIDATION_PATTERNS["email"])

    def validate_uuid(self, uuid_str: str, field_name: str = "id") -> str:
        """
        Validate UUID format.

        Args:
            uuid_str: UUID string to validate
            field_name: Field name for error reporting

        Returns:
            Validated UUID string

        Raises:
            ValidationError: If validation fails
        """
        return self.validate_string(uuid_str, field_name, pattern=VALIDATION_PATTERNS["uuid"])

    def validate_integer(
        self,
        value: int | str,
        field_name: str,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        """
        Validate integer value.

        Args:
            value: Value to validate
            field_name: Field name for error reporting
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"{field_name} must be a valid integer", field_name) from e

        return self._validate_numeric_bounds(min_value, int_value, field_name, max_value)

    def validate_float(
        self,
        value: float | str,
        field_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        """
        Validate float value.

        Args:
            value: Value to validate
            field_name: Field name for error reporting
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated float

        Raises:
            ValidationError: If validation fails
        """
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"{field_name} must be a valid number", field_name) from e

        return self._validate_numeric_bounds(min_value, float_value, field_name, max_value)

    def _validate_numeric_bounds(self, min_value, arg1, field_name, max_value):
        if min_value is not None and arg1 < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name)
        if max_value is not None and arg1 > max_value:
            raise ValidationError(f"{field_name} must not exceed {max_value}", field_name)
        return arg1

    def validate_file_upload(
        self,
        file,
        allowed_extensions: list[str] | None = None,
        allowed_mime_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate file upload with security checks.

        Args:
            file: File object to validate
            allowed_extensions: List of allowed file extensions
            allowed_mime_types: List of allowed MIME types

        Returns:
            Dictionary with validated file information

        Raises:
            ValidationError: If validation fails
        """
        if not file:
            raise ValidationError("No file provided", "file")

        # Get file information
        filename = getattr(file, "filename", None)
        if not filename:
            raise ValidationError("File has no filename", "file")

        # Secure filename
        secure_name = secure_filename(filename)
        if not secure_name:
            raise ValidationError("Invalid filename", "file")

        # Check file extension
        file_extension = Path(secure_name).suffix.lower().lstrip(".")

        if allowed_extensions and file_extension not in allowed_extensions:
            allowed_str = ", ".join(allowed_extensions)
            raise ValidationError(
                f"File type '{file_extension}' not allowed. Allowed: {allowed_str}",
                "file",
            )

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer

        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            msg = f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)"
            raise ValidationError(msg, "file")

        # Check MIME type if provided
        if allowed_mime_types:
            mime_type, _ = mimetypes.guess_type(secure_name)
            if mime_type not in allowed_mime_types:
                raise ValidationError(f"File MIME type '{mime_type}' not allowed", "file")

        return {
            "filename": secure_name,
            "extension": file_extension,
            "size_bytes": file_size,
            "mime_type": mime_type if allowed_mime_types else None,
        }

    def validate_dataset_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Validate dataset metadata.

        Args:
            metadata: Dataset metadata dictionary

        Returns:
            Validated metadata

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ["name", "description", "format"]

        # Check required fields
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                raise ValidationError(f"Missing required field: {field}", field)

        # Validate name
        name = self.validate_string(metadata["name"], "name", min_length=3, max_length=100)

        # Validate description
        description = self.validate_string(metadata["description"], "description", min_length=10, max_length=500)

        # Validate format
        allowed_formats = ["csv", "json", "jsonl", "parquet"]
        format_type = metadata["format"].lower()
        if format_type not in allowed_formats:
            allowed_str = ", ".join(allowed_formats)
            raise ValidationError(
                f"Invalid format '{format_type}'. Allowed: {allowed_str}",
                "format",
            )

        # Validate optional fields
        validated_metadata = {
            "name": name,
            "description": description,
            "format": format_type,
        }

        # Validate tags if present
        if "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list):
                raise ValidationError("Tags must be a list", "tags")

            validated_tags = []
            for tag in tags:
                validated_tag = self.validate_string(tag, "tag", min_length=1, max_length=50)
                validated_tags.append(validated_tag)

            validated_metadata["tags"] = validated_tags

        # Validate privacy level if present
        if "privacy_level" in metadata:
            privacy_level = self.validate_string(metadata["privacy_level"], "privacy_level")
            allowed_levels = ["public", "internal", "confidential", "restricted"]
            if privacy_level not in allowed_levels:
                allowed_str = ", ".join(allowed_levels)
                raise ValidationError(
                    f"Invalid privacy level '{privacy_level}'. Allowed: {allowed_str}",
                    "privacy_level",
                )
            validated_metadata["privacy_level"] = privacy_level

        return validated_metadata

    def sanitize_for_output(self, data: Any) -> Any:
        """
        Sanitize data for safe output (remove sensitive information).

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return {
                key: (
                    "[REDACTED]"
                    if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS)
                    else self.sanitize_for_output(value)
                )
                for key, value in data.items()
            }
        if isinstance(data, list):
            return [self.sanitize_for_output(item) for item in data]
        if isinstance(data, str):
            # Remove sensitive patterns
            sanitized = data
            for pattern in self._COMPILED_SENSITIVE_PATTERNS:
                sanitized = pattern.sub("[REDACTED]", sanitized)
            return sanitized
        return data

    def _sanitize_string(self, value: str) -> str:
        """
        Basic string sanitization.

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        # Remove null bytes and control characters
        sanitized = value.replace("\x00", "")
        sanitized = self._RE_CONTROL_CHARS.sub("", sanitized)

        # Trim whitespace
        return sanitized.strip()


class DatasetValidator:
    """Validator for dataset files and content."""

    def __init__(self, max_rows: int = 1000000, max_columns: int = 1000):
        """
        Initialize dataset validator.

        Args:
            max_rows: Maximum allowed rows
            max_columns: Maximum allowed columns
        """
        self.max_rows = max_rows
        self.max_columns = max_columns

    def validate_csv_file(self, file_path: str) -> dict[str, Any]:
        """
        Validate CSV file format and content.

        Args:
            file_path: Path to CSV file

        Returns:
            Validation results

        Raises:
            ValidationError: If validation fails
        """
        try:
            return self._validate_csv_structure(file_path)
        except pd.errors.EmptyDataError as e:
            raise ValidationError("CSV file is empty", "file") from e
        except pd.errors.ParserError as e:
            raise ValidationError(f"CSV parsing error: {e!s}", "file") from e
        except Exception as e:
            raise ValidationError(f"CSV validation error: {e!s}", "file") from e

    def _validate_csv_structure(self, file_path):
        # Read first few rows to validate structure
        df = pd.read_csv(file_path, nrows=100)

        # Check column count
        if len(df.columns) > self.max_columns:
            raise ValidationError(
                f"Too many columns ({len(df.columns)}). Maximum allowed: {self.max_columns}",
                "columns",
            )

        # Check for required columns (if any)
        # This can be extended based on specific requirements

        # Validate column names
        for col in df.columns:
            if not isinstance(col, str) or not col.strip():
                raise ValidationError(f"Invalid column name: {col}", "column_names")

            # Check for potentially sensitive column names
            sensitive_keywords = ["ssn", "password", "secret", "key"]
            if any(keyword in col.lower() for keyword in sensitive_keywords):
                raise ValidationError(f"Potentially sensitive column name detected: {col}", "column_names")

        # Get full row count
        with open(file_path, encoding="utf-8") as csv_file:
            row_count = sum(1 for _ in csv_file) - 1  # Subtract header

        if row_count > self.max_rows:
            raise ValidationError(f"Too many rows ({row_count}). Maximum allowed: {self.max_rows}", "rows")

        return {
            "columns": list(df.columns),
            "sample_rows": len(df),
            "total_rows": row_count,
            "validation_status": "valid",
        }

    def validate_json_file(self, file_path: str) -> dict[str, Any]:
        """
        Validate JSON file format and content.

        Args:
            file_path: Path to JSON file

        Returns:
            Validation results

        Raises:
            ValidationError: If validation fails
        """
        try:
            with open(file_path, encoding="utf-8") as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON format: {exc!s}", "file") from exc
        except Exception as exc:
            raise ValidationError(f"JSON validation error: {exc!s}", "file") from exc

        # Basic structure validation
        if isinstance(data, list):
            if len(data) > self.max_rows:
                raise ValidationError(
                    f"Too many records ({len(data)}). Maximum allowed: {self.max_rows}",
                    "records",
                )

            if data and isinstance(data[0], dict) and len(data[0]) > self.max_columns:
                raise ValidationError(
                    f"Too many fields ({len(data[0])}). Maximum allowed: {self.max_columns}",
                    "fields",
                )

        return {
            "data_type": type(data).__name__,
            "record_count": len(data) if isinstance(data, list) else 1,
            "validation_status": "valid",
        }


# Convenience validation functions
def validate_dataset_name(name: str) -> str:
    """Validate dataset name."""
    validator = InputValidator()
    return validator.validate_string(name, "dataset_name", min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


def validate_dataset_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate dataset metadata."""
    validator = InputValidator()
    return validator.validate_dataset_metadata(metadata)


def validate_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate pipeline configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Validated configuration

    Raises:
        ValidationError: If validation fails
    """
    required_fields = ["stages", "timeout"]

    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}", field)

    # Validate stages
    stages = config["stages"]
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValidationError("Stages must be a non-empty list", "stages")

    allowed_stages = [
        "data_ingestion",
        "preprocessing",
        "feature_engineering",
        "model_training",
        "validation",
        "bias_detection",
        "output_generation",
    ]

    for stage in stages:
        if stage not in allowed_stages:
            raise ValidationError(
                f"Invalid stage '{stage}'. Allowed: {', '.join(allowed_stages)}",
                "stages",
            )

    # Validate timeout
    timeout = config["timeout"]
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValidationError("Timeout must be a positive integer", "timeout")

    if timeout > MAX_PIPELINE_TIMEOUT_SECONDS:  # 1 hour max
        raise ValidationError(f"Timeout cannot exceed {MAX_PIPELINE_TIMEOUT_SECONDS} seconds (1 hour)", "timeout")

    return {"stages": stages, "timeout": timeout}


def sanitize_user_input(data: Any) -> Any:
    """
    Sanitize user input for safe processing.

    Args:
        data: Data to sanitize

    Returns:
        Sanitized data
    """
    validator = InputValidator()
    return validator.sanitize_for_output(data)


# Backwards-compatible helper expected by pipeline orchestrator
def validate_pipeline_input(config: dict[str, Any], dataset_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate pipeline configuration and dataset info.

    This is a thin wrapper around validate_pipeline_config and dataset validation
    used by older modules.
    """
    validated_config = validate_pipeline_config(config)

    if dataset_info:
        validator = InputValidator()
        # If dataset metadata present, validate it
        if isinstance(dataset_info, dict) and (metadata := dataset_info.get("metadata")) is not None:
            validator.validate_dataset_metadata(metadata)

    return {"config": validated_config, "dataset_info": dataset_info or {}}


def sanitize_input(data: Any) -> Any:
    """Compatibility wrapper for input sanitization used across the codebase."""
    return sanitize_user_input(data)


def validate_state_data(state: Any) -> dict[str, Any]:
    """
    Backwards-compatible state validation expected by the pipeline orchestrator.

    Returns a dictionary with keys:
      - is_valid: bool
      - errors: list

    The implementation is intentionally permissive: it ensures the value is a
    mapping and checks for a few common required fields. More strict checks
    are possible but kept minimal to preserve compatibility across callers.
    """
    if not isinstance(state, dict):
        return {"is_valid": False, "errors": ["state must be a dict"]}

    required = ["execution_id", "user_id", "status", "current_stage", "start_time"]
    errors: list[str] = [f"missing required field: {key}" for key in required if key not in state]

    if errors:
        return {"is_valid": False, "errors": errors}

    # If start_time is a string, try to accept it (further parsing occurs elsewhere)
    return {"is_valid": True, "errors": []}


def validate_file_upload(
    file,
    config: Any | None = None,
    allowed_extensions: list[str] | None = None,
    allowed_mime_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate file upload (top-level helper).

    Args:
        file: File object to validate
        config: Configuration object or dict containing max size settings
        allowed_extensions: List of allowed file extensions
        allowed_mime_types: List of allowed MIME types

    Returns:
        Dictionary with validated file information
    """
    # Default max size 100MB
    max_size_mb = 100

    if config:
        if isinstance(config, dict):
            max_size_mb = config.get("MAX_FILE_SIZE_MB", 100)
        else:
            max_size_mb = getattr(config, "MAX_FILE_SIZE_MB", 100)

    validator = InputValidator(max_file_size_mb=max_size_mb)
    return validator.validate_file_upload(file, allowed_extensions, allowed_mime_types)
