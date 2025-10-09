"""
Validation utilities for TechDeck Flask service.

This module provides comprehensive input validation, data sanitization,
and security measures for HIPAA++ compliance.
"""

import re
import os
import mimetypes
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename


class ValidationError(Exception):
    """Custom validation error with detailed information."""

    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code or 'VALIDATION_ERROR'


class InputValidator:
    """
    Comprehensive input validator with security measures.

    Provides validation for various input types with HIPAA++
    compliance and security considerations.
    """

    # Common patterns for validation
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        'alphanumeric': r'^[a-zA-Z0-9_-]+$',
        'numeric': r'^\d+$',
        'float': r'^\d*\.?\d+$',
        'date_iso': r'^\d{4}-\d{2}-\d{2}$',
        'datetime_iso': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$'
    }

    # Sensitive field patterns (for sanitization)
    SENSITIVE_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        r'\b\d+\.\d+\.\d+\.\d+\b',  # IP address
    ]

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
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_empty: bool = False
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
                field_name
            )

        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} must not exceed {max_length} characters",
                field_name
            )

        # Check pattern if provided
        if pattern and not re.match(pattern, value):
            raise ValidationError(
                f"{field_name} format is invalid",
                field_name
            )

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
        return self.validate_string(
            email,
            field_name,
            pattern=self.PATTERNS['email']
        )

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
        return self.validate_string(
            uuid_str,
            field_name,
            pattern=self.PATTERNS['uuid']
        )

    def validate_integer(
        self,
        value: Union[int, str],
        field_name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
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
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer", field_name)

        return self._extracted_from_validate_float_28(
            min_value, int_value, field_name, max_value
        )

    def validate_float(
        self,
        value: Union[float, str],
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
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
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid number", field_name)

        return self._extracted_from_validate_float_28(
            min_value, float_value, field_name, max_value
        )

    # TODO Rename this here and in `validate_integer` and `validate_float`
    def _extracted_from_validate_float_28(self, min_value, arg1, field_name, max_value):
        if min_value is not None and arg1 < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name)
        if max_value is not None and arg1 > max_value:
            raise ValidationError(f"{field_name} must not exceed {max_value}", field_name)
        return arg1

    def validate_file_upload(
        self,
        file,
        allowed_extensions: Optional[List[str]] = None,
        allowed_mime_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        filename = getattr(file, 'filename', None)
        if not filename:
            raise ValidationError("File has no filename", "file")

        # Secure filename
        secure_name = secure_filename(filename)
        if not secure_name:
            raise ValidationError("Invalid filename", "file")

        # Check file extension
        file_extension = Path(secure_name).suffix.lower().lstrip('.')

        if allowed_extensions and file_extension not in allowed_extensions:
            raise ValidationError(
                f"File type '{file_extension}' not allowed. Allowed: {', '.join(allowed_extensions)}",
                "file"
            )

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer

        if file_size > self.max_file_size_bytes:
            raise ValidationError(
                f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)",
                "file"
            )

        # Check MIME type if provided
        if allowed_mime_types:
            mime_type, _ = mimetypes.guess_type(secure_name)
            if mime_type not in allowed_mime_types:
                raise ValidationError(
                    f"File MIME type '{mime_type}' not allowed",
                    "file"
                )

        return {
            'filename': secure_name,
            'extension': file_extension,
            'size_bytes': file_size,
            'mime_type': mime_type if allowed_mime_types else None
        }

    def validate_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dataset metadata.

        Args:
            metadata: Dataset metadata dictionary

        Returns:
            Validated metadata

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['name', 'description', 'format']

        # Check required fields
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                raise ValidationError(f"Missing required field: {field}", field)

        # Validate name
        name = self.validate_string(
            metadata['name'],
            'name',
            min_length=3,
            max_length=100
        )

        # Validate description
        description = self.validate_string(
            metadata['description'],
            'description',
            min_length=10,
            max_length=500
        )

        # Validate format
        allowed_formats = ['csv', 'json', 'jsonl', 'parquet']
        format_type = metadata['format'].lower()
        if format_type not in allowed_formats:
            raise ValidationError(
                f"Invalid format '{format_type}'. Allowed: {', '.join(allowed_formats)}",
                'format'
            )

        # Validate optional fields
        validated_metadata = {
            'name': name,
            'description': description,
            'format': format_type
        }

        # Validate tags if present
        if 'tags' in metadata:
            tags = metadata['tags']
            if not isinstance(tags, list):
                raise ValidationError("Tags must be a list", 'tags')

            validated_tags = []
            for tag in tags:
                validated_tag = self.validate_string(
                    tag,
                    'tag',
                    min_length=1,
                    max_length=50
                )
                validated_tags.append(validated_tag)

            validated_metadata['tags'] = validated_tags

        # Validate privacy level if present
        if 'privacy_level' in metadata:
            privacy_level = self.validate_string(
                metadata['privacy_level'],
                'privacy_level'
            )
            allowed_levels = ['public', 'internal', 'confidential', 'restricted']
            if privacy_level not in allowed_levels:
                raise ValidationError(
                    f"Invalid privacy level '{privacy_level}'. Allowed: {', '.join(allowed_levels)}",
                    'privacy_level'
                )
            validated_metadata['privacy_level'] = privacy_level

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
                    '[REDACTED]'
                    if any(
                        sensitive in key.lower()
                        for sensitive in self.SENSITIVE_FIELDS
                    )
                    else self.sanitize_for_output(value)
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize_for_output(item) for item in data]
        elif isinstance(data, str):
            # Remove sensitive patterns
            sanitized = data
            for pattern in self.SENSITIVE_PATTERNS:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
            return sanitized
        else:
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
        sanitized = value.replace('\x00', '')
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)

        # Trim whitespace
        sanitized = sanitized.strip()

        return sanitized


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

    def validate_csv_file(self, file_path: str) -> Dict[str, Any]:
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
            return self._extracted_from_validate_csv_file_16(file_path)
        except pd.errors.EmptyDataError:
            raise ValidationError("CSV file is empty", 'file')
        except pd.errors.ParserError as e:
            raise ValidationError(f"CSV parsing error: {str(e)}", 'file')
        except Exception as e:
            raise ValidationError(f"CSV validation error: {str(e)}", 'file')

    # TODO Rename this here and in `validate_csv_file`
    def _extracted_from_validate_csv_file_16(self, file_path):
        # Read first few rows to validate structure
        df = pd.read_csv(file_path, nrows=100)

        # Check column count
        if len(df.columns) > self.max_columns:
            raise ValidationError(
                f"Too many columns ({len(df.columns)}). Maximum allowed: {self.max_columns}",
                'columns'
            )

        # Check for required columns (if any)
        # This can be extended based on specific requirements

        # Validate column names
        for col in df.columns:
            if not isinstance(col, str) or not col.strip():
                raise ValidationError(
                    f"Invalid column name: {col}",
                    'column_names'
                )

            # Check for potentially sensitive column names
            sensitive_keywords = ['ssn', 'password', 'secret', 'key']
            if any(keyword in col.lower() for keyword in sensitive_keywords):
                raise ValidationError(
                    f"Potentially sensitive column name detected: {col}",
                    'column_names'
                )

        # Get full row count
        row_count = sum(1 for _ in open(file_path)) - 1  # Subtract header

        if row_count > self.max_rows:
            raise ValidationError(
                f"Too many rows ({row_count}). Maximum allowed: {self.max_rows}",
                'rows'
            )

        return {
            'columns': list(df.columns),
            'sample_rows': len(df),
            'total_rows': row_count,
            'validation_status': 'valid'
        }

    def validate_json_file(self, file_path: str) -> Dict[str, Any]:
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
            import json

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Basic structure validation
            if isinstance(data, list):
                if len(data) > self.max_rows:
                    raise ValidationError(
                        f"Too many records ({len(data)}). Maximum allowed: {self.max_rows}",
                        'records'
                    )

                if data and isinstance(data[0], dict) and len(data[0]) > self.max_columns:
                    raise ValidationError(
                        f"Too many fields ({len(data[0])}). Maximum allowed: {self.max_columns}",
                        'fields'
                    )

            return {
                'data_type': type(data).__name__,
                'record_count': len(data) if isinstance(data, list) else 1,
                'validation_status': 'valid'
            }

        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {str(e)}", 'file')
        except Exception as e:
            raise ValidationError(f"JSON validation error: {str(e)}", 'file')


# Convenience validation functions
def validate_dataset_name(name: str) -> str:
    """Validate dataset name."""
    validator = InputValidator()
    return validator.validate_string(
        name,
        'dataset_name',
        min_length=3,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$'
    )


def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pipeline configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Validated configuration

    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['stages', 'timeout']

    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}", field)

    # Validate stages
    stages = config['stages']
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValidationError("Stages must be a non-empty list", 'stages')

    allowed_stages = [
        'data_ingestion', 'preprocessing', 'feature_engineering',
        'model_training', 'validation', 'bias_detection', 'output_generation'
    ]

    for stage in stages:
        if stage not in allowed_stages:
            raise ValidationError(
                f"Invalid stage '{stage}'. Allowed: {', '.join(allowed_stages)}",
                'stages'
            )

    # Validate timeout
    timeout = config['timeout']
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValidationError("Timeout must be a positive integer", 'timeout')

    if timeout > 3600:  # 1 hour max
        raise ValidationError("Timeout cannot exceed 3600 seconds (1 hour)", 'timeout')

    return {
        'stages': stages,
        'timeout': timeout
    }


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
def validate_pipeline_input(config: Dict[str, Any], dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate pipeline configuration and dataset info.

    This is a thin wrapper around validate_pipeline_config and dataset validation
    used by older modules.
    """
    validated_config = validate_pipeline_config(config)

    if dataset_info:
        validator = InputValidator()
        # If dataset metadata present, validate it
        if isinstance(dataset_info, dict) and 'metadata' in dataset_info:
            validator.validate_dataset_metadata(dataset_info['metadata'])

    return {
        'config': validated_config,
        'dataset_info': dataset_info or {}
    }


def sanitize_input(data: Any) -> Any:
    """Compatibility wrapper for input sanitization used across the codebase."""
    return sanitize_user_input(data)


def validate_state_data(state: Any) -> Dict[str, Any]:
    """
    Backwards-compatible state validation expected by the pipeline orchestrator.

    Returns a dictionary with keys:
      - is_valid: bool
      - errors: list

    The implementation is intentionally permissive: it ensures the value is a
    mapping and checks for a few common required fields. More strict checks
    are possible but kept minimal to preserve compatibility across callers.
    """
    errors: List[str] = []

    if not isinstance(state, dict):
        return {"is_valid": False, "errors": ["state must be a dict"]}

    # Minimal required fields for a PipelineState-like object
    required = ["execution_id", "user_id", "status", "current_stage", "start_time"]
    for key in required:
        if key not in state:
            errors.append(f"missing required field: {key}")

    if errors:
        return {"is_valid": False, "errors": errors}

    # If start_time is a string, try to accept it (further parsing occurs elsewhere)
    return {"is_valid": True, "errors": []}
