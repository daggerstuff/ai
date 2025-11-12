"""
Base prompt class for MCP prompts.

This module provides the base MCPPrompt class that all MCP prompts must inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPPrompt(ABC):
    """Base class for MCP prompts."""

    def __init__(
        self,
        name: str,
        description: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize MCP prompt.

        Args:
            name: Prompt name (must be unique)
            description: Prompt description
            arguments: List of argument definitions (JSON Schema format)
        """
        self.name = name
        self.description = description
        self.arguments = arguments or []

    @property
    def prompt_schema(self) -> Dict[str, Any]:
        """
        Get prompt schema in MCP format.

        Returns:
            Prompt schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }

    @abstractmethod
    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render prompt template with provided parameters.

        Args:
            params: Optional parameters for prompt rendering

        Returns:
            Rendered prompt text

        Raises:
            ValueError: If required parameters are missing
        """
        pass

    def validate_arguments(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Validate prompt arguments against schema.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        if not params:
            params = {}

        # Validate required arguments
        for arg in self.arguments:
            arg_name = arg.get("name")
            required = arg.get("required", False)

            if required and arg_name not in params:
                raise ValueError(f"Missing required argument: {arg_name}")

            # Type validation (basic)
            if arg_name in params:
                expected_type = arg.get("type")
                if expected_type:
                    self._validate_type(arg_name, params[arg_name], expected_type)

    def _validate_type(self, field: str, value: Any, expected_type: str) -> None:
        """
        Validate argument type.

        Args:
            field: Field name
            value: Field value
            expected_type: Expected type (string, integer, number, boolean, array, object)

        Raises:
            ValueError: If type doesn't match
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type not in type_map:
            return  # Unknown type, skip validation

        expected_python_type = type_map[expected_type]
        if not isinstance(value, expected_python_type):
            raise ValueError(
                f"Argument '{field}' must be of type {expected_type}, "
                f"got {type(value).__name__}"
            )

    def format_template(
        self,
        template: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format template string with parameters.

        Args:
            template: Template string with {placeholder} syntax
            params: Parameters to substitute

        Returns:
            Formatted string
        """
        if not params:
            params = {}

        try:
            return template.format(**params)
        except KeyError as e:
            raise ValueError(f"Missing template parameter: {e}") from e

    def __repr__(self) -> str:
        """String representation of prompt."""
        return f"<MCPPrompt name={self.name!r}>"

