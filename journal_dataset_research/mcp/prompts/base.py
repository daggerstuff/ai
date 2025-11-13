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
            ValidationError: If parameters are invalid
        """
        from ai.journal_dataset_research.mcp.utils.validation import (
            ValidationError,
            validate_prompt_arguments,
        )

        if not params:
            params = {}

        # Convert arguments list to JSON schema format
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for arg in self.arguments:
            arg_name = arg.get("name")
            if not arg_name:
                continue

            schema["properties"][arg_name] = {
                "type": arg.get("type", "string"),
                "description": arg.get("description", ""),
            }

            if arg.get("required", False):
                schema["required"].append(arg_name)

        try:
            validate_prompt_arguments(params, schema)
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ValidationError(
                f"Prompt argument validation failed: {str(e)}",
                value=params,
            ) from e

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

