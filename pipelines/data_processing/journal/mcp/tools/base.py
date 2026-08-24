"""
Base tool class for MCP tools.

This module provides the base MCPTool class that all MCP tools must inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ai.pipelines.data_processing.journal.mcp.utils.validation import (
    ValidationError,
    validate_tool_parameters,
)

logger = logging.getLogger(__name__)


class MCPTool(ABC):
    """Base class for MCP tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
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
    def tool_schema(self) -> dict[str, Any]:
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
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            params: Tool parameters (validated)

        Returns:
            Tool execution result

        Raises:
            ValidationError: If parameters are invalid
            Exception: If tool execution fails
        """

    def validate_parameters(self, params: dict[str, Any]) -> None:
        """
        Validate tool parameters against schema.

        Args:
            params: Parameters to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            validate_tool_parameters(params, self.parameters)
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ValidationError(
                f"Parameter validation failed: {e!s}",
                value=params,
            ) from e
