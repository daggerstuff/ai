"""
Base tool class for MCP tools.

This module provides the base MCPTool class that all MCP tools must inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MCPTool(ABC):
    """Base class for MCP tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Initialize MCP tool.

        Args:
            name: Tool name (must be unique)
            description: Tool description
            parameters: JSON Schema for tool parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """
        Get tool schema in MCP format.

        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            params: Tool parameters (validated)

        Returns:
            Tool execution result

        Raises:
            ValueError: If parameters are invalid
            Exception: If tool execution fails
        """
        pass

    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate tool parameters against schema.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        # Basic validation - check required fields
        schema = self.parameters
        required = schema.get("required", [])

        for field in required:
            if field not in params:
                raise ValueError(f"Missing required parameter: {field}")

        # Type validation (basic)
        properties = schema.get("properties", {})
        for field, value in params.items():
            if field in properties:
                prop = properties[field]
                expected_type = prop.get("type")
                if expected_type:
                    self._validate_type(field, value, expected_type)

    def _validate_type(self, field: str, value: Any, expected_type: str) -> None:
        """
        Validate parameter type.

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
                f"Parameter '{field}' must be of type {expected_type}, "
                f"got {type(value).__name__}"
            )

    def __repr__(self) -> str:
        """String representation of tool."""
        return f"<MCPTool name={self.name!r}>"

