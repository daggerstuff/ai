"""
Parameter validation utilities for MCP Server.

This module provides comprehensive parameter validation for tools, resources, and prompts.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ai.sourcing.journal.mcp.protocol import (
    MCPError,
    MCPErrorCode,
)

logger = None  # Will be set when logging is needed


class ValidationError(MCPError):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            value: Value that failed validation
            expected: Expected value or type
        """
        super().__init__(
            MCPErrorCode.TOOL_VALIDATION_ERROR,
            message,
            {"field": field, "value": value, "expected": expected},
        )
        self.field = field
        self.value = value
        self.expected = expected


class ParameterValidator:
    """Comprehensive parameter validator for MCP components."""

    @staticmethod
    def validate_required(
        params: Dict[str, Any],
        required_fields: List[str],
        context: str = "parameters",
    ) -> None:
        """
        Validate that all required fields are present.

        Args:
            params: Parameters to validate
            required_fields: List of required field names
            context: Context for error messages (e.g., "tool parameters")

        Raises:
            ValidationError: If any required field is missing
        """
        missing = [field for field in required_fields if field not in params]
        if missing:
            raise ValidationError(
                f"Missing required {context}: {', '.join(missing)}",
                field=",".join(missing),
            )

    @staticmethod
    def validate_type(
        value: Any,
        expected_type: Union[type, Tuple[type, ...]],
        field_name: str = "parameter",
    ) -> None:
        """
        Validate parameter type.

        Args:
            value: Value to validate
            expected_type: Expected type or tuple of types
            field_name: Field name for error messages

        Raises:
            ValidationError: If type doesn't match
        """
        if not isinstance(value, expected_type):
            type_names = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " or ".join(t.__name__ for t in expected_type)
            )
            raise ValidationError(
                f"{field_name} must be of type {type_names}, got {type(value).__name__}",
                field=field_name,
                value=value,
                expected=type_names,
            )

    @staticmethod
    def validate_string(
        value: Any,
        field_name: str = "parameter",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_empty: bool = True,
    ) -> None:
        """
        Validate string parameter.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_length: Minimum length
            max_length: Maximum length
            pattern: Optional regex pattern
            allow_empty: Whether empty strings are allowed

        Raises:
            ValidationError: If validation fails
        """
        ParameterValidator.validate_type(value, str, field_name)

        if not allow_empty and not value:
            raise ValidationError(
                f"{field_name} cannot be empty",
                field=field_name,
                value=value,
            )

        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field=field_name,
                value=value,
                expected=f"min_length={min_length}",
            )

        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters",
                field=field_name,
                value=value,
                expected=f"max_length={max_length}",
            )

        if pattern and not re.match(pattern, value):
            raise ValidationError(
                f"{field_name} does not match required pattern: {pattern}",
                field=field_name,
                value=value,
                expected=f"pattern={pattern}",
            )

    @staticmethod
    def validate_number(
        value: Any,
        field_name: str = "parameter",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        integer_only: bool = False,
    ) -> None:
        """
        Validate number parameter.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_value: Minimum value
            max_value: Maximum value
            integer_only: If True, only integers are allowed

        Raises:
            ValidationError: If validation fails
        """
        if integer_only:
            ParameterValidator.validate_type(value, int, field_name)
        else:
            ParameterValidator.validate_type(value, (int, float), field_name)

        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{field_name} must be at least {min_value}",
                field=field_name,
                value=value,
                expected=f"min_value={min_value}",
            )

        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} must be at most {max_value}",
                field=field_name,
                value=value,
                expected=f"max_value={max_value}",
            )

    @staticmethod
    def validate_array(
        value: Any,
        field_name: str = "parameter",
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        item_type: Optional[type] = None,
        unique: bool = False,
    ) -> None:
        """
        Validate array parameter.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_items: Minimum number of items
            max_items: Maximum number of items
            item_type: Expected type for array items
            unique: If True, all items must be unique

        Raises:
            ValidationError: If validation fails
        """
        ParameterValidator.validate_type(value, list, field_name)

        if min_items is not None and len(value) < min_items:
            raise ValidationError(
                f"{field_name} must have at least {min_items} items",
                field=field_name,
                value=value,
                expected=f"min_items={min_items}",
            )

        if max_items is not None and len(value) > max_items:
            raise ValidationError(
                f"{field_name} must have at most {max_items} items",
                field=field_name,
                value=value,
                expected=f"max_items={max_items}",
            )

        if item_type:
            for i, item in enumerate(value):
                try:
                    ParameterValidator.validate_type(item, item_type, f"{field_name}[{i}]")
                except ValidationError as e:
                    raise ValidationError(
                        f"{field_name} item at index {i} is invalid: {e.message}",
                        field=f"{field_name}[{i}]",
                        value=item,
                        expected=item_type.__name__,
                    ) from e

        if unique and len(value) != len(set(value)):
            raise ValidationError(
                f"{field_name} must contain unique items",
                field=field_name,
                value=value,
            )

    @staticmethod
    def validate_object(
        value: Any,
        field_name: str = "parameter",
        required_fields: Optional[List[str]] = None,
        allowed_fields: Optional[List[str]] = None,
    ) -> None:
        """
        Validate object parameter.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            required_fields: List of required field names
            allowed_fields: List of allowed field names (None = all allowed)

        Raises:
            ValidationError: If validation fails
        """
        ParameterValidator.validate_type(value, dict, field_name)

        if required_fields:
            missing = [f for f in required_fields if f not in value]
            if missing:
                raise ValidationError(
                    f"{field_name} missing required fields: {', '.join(missing)}",
                    field=field_name,
                    value=value,
                    expected=f"required_fields={required_fields}",
                )

        if allowed_fields:
            invalid = [f for f in value.keys() if f not in allowed_fields]
            if invalid:
                raise ValidationError(
                    f"{field_name} contains invalid fields: {', '.join(invalid)}",
                    field=field_name,
                    value=value,
                    expected=f"allowed_fields={allowed_fields}",
                )

    @staticmethod
    def validate_enum(
        value: Any,
        allowed_values: List[Any],
        field_name: str = "parameter",
    ) -> None:
        """
        Validate enum parameter.

        Args:
            value: Value to validate
            allowed_values: List of allowed values
            field_name: Field name for error messages

        Raises:
            ValidationError: If validation fails
        """
        if value not in allowed_values:
            raise ValidationError(
                f"{field_name} must be one of {allowed_values}, got {value}",
                field=field_name,
                value=value,
                expected=f"one of {allowed_values}",
            )

    @staticmethod
    def validate_json_schema(
        params: Dict[str, Any],
        schema: Dict[str, Any],
        context: str = "parameters",
    ) -> None:
        """
        Validate parameters against JSON Schema.

        Args:
            params: Parameters to validate
            schema: JSON Schema definition
            context: Context for error messages

        Raises:
            ValidationError: If validation fails
        """
        # Validate required fields
        required = schema.get("required", [])
        ParameterValidator.validate_required(params, required, context)

        # Validate properties
        properties = schema.get("properties", {})
        for field_name, value in params.items():
            if field_name not in properties:
                # Field not in schema - check if additionalProperties is allowed
                additional_props = schema.get("additionalProperties", True)
                if not additional_props:
                    raise ValidationError(
                        f"Unknown {context} field: {field_name}",
                        field=field_name,
                        value=value,
                    )
                continue

            prop_schema = properties[field_name]
            ParameterValidator._validate_property(
                field_name, value, prop_schema, context
            )

    @staticmethod
    def _validate_property(
        field_name: str,
        value: Any,
        prop_schema: Dict[str, Any],
        context: str = "parameter",
    ) -> None:
        """
        Validate a single property against its schema.

        Args:
            field_name: Field name
            value: Field value
            prop_schema: Property schema definition
            context: Context for error messages

        Raises:
            ValidationError: If validation fails
        """
        prop_type = prop_schema.get("type")

        if prop_type == "string":
            ParameterValidator.validate_string(
                value,
                field_name,
                min_length=prop_schema.get("minLength"),
                max_length=prop_schema.get("maxLength"),
                pattern=prop_schema.get("pattern"),
                allow_empty=prop_schema.get("allowEmpty", True),
            )
        elif prop_type == "integer":
            ParameterValidator.validate_number(
                value,
                field_name,
                min_value=prop_schema.get("minimum"),
                max_value=prop_schema.get("maximum"),
                integer_only=True,
            )
        elif prop_type == "number":
            ParameterValidator.validate_number(
                value,
                field_name,
                min_value=prop_schema.get("minimum"),
                max_value=prop_schema.get("maximum"),
                integer_only=False,
            )
        elif prop_type == "boolean":
            ParameterValidator.validate_type(value, bool, field_name)
        elif prop_type == "array":
            items_schema = prop_schema.get("items", {})
            item_type = None
            if isinstance(items_schema, dict) and "type" in items_schema:
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "object": dict,
                    "array": list,
                }
                item_type = type_map.get(items_schema["type"])

            ParameterValidator.validate_array(
                value,
                field_name,
                min_items=prop_schema.get("minItems"),
                max_items=prop_schema.get("maxItems"),
                item_type=item_type,
                unique=prop_schema.get("uniqueItems", False),
            )
        elif prop_type == "object":
            ParameterValidator.validate_object(
                value,
                field_name,
                required_fields=prop_schema.get("required"),
                allowed_fields=prop_schema.get("properties", {}).keys()
                if not prop_schema.get("additionalProperties", True)
                else None,
            )
        elif prop_type:
            # Unknown type - just check it's not None
            if value is None:
                raise ValidationError(
                    f"{field_name} cannot be None",
                    field=field_name,
                    value=value,
                )

        # Validate enum if specified
        if "enum" in prop_schema:
            ParameterValidator.validate_enum(
                value, prop_schema["enum"], field_name
            )


def validate_tool_parameters(
    params: Dict[str, Any], schema: Dict[str, Any]
) -> None:
    """
    Validate tool parameters against schema.

    Args:
        params: Tool parameters
        schema: Tool parameter schema

    Raises:
        ValidationError: If validation fails
    """
    ParameterValidator.validate_json_schema(params, schema, "tool parameter")


def validate_resource_parameters(
    params: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate resource parameters.

    Args:
        params: Resource parameters
        schema: Optional resource parameter schema

    Raises:
        ValidationError: If validation fails
    """
    if schema:
        ParameterValidator.validate_json_schema(params, schema, "resource parameter")
    # Basic validation: ensure params is a dict
    if not isinstance(params, dict):
        raise ValidationError(
            "Resource parameters must be a dictionary",
            value=params,
        )


def validate_prompt_arguments(
    args: Dict[str, Any], schema: Dict[str, Any]
) -> None:
    """
    Validate prompt arguments against schema.

    Args:
        args: Prompt arguments
        schema: Prompt argument schema

    Raises:
        ValidationError: If validation fails
    """
    ParameterValidator.validate_json_schema(args, schema, "prompt argument")

