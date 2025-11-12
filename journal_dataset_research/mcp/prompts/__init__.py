"""
MCP Prompts for Journal Dataset Research System.

This module provides prompt templates for common research workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.mcp.prompts.base import MCPPrompt

logger = logging.getLogger(__name__)

__all__ = [
    "PromptRegistry",
    "MCPPrompt",
]


class PromptRegistry:
    """Registry for MCP prompts."""

    def __init__(self) -> None:
        """Initialize prompt registry."""
        self._prompts: Dict[str, MCPPrompt] = {}

    def register(self, prompt: MCPPrompt) -> None:
        """
        Register an MCP prompt.

        Args:
            prompt: Prompt to register

        Raises:
            ValueError: If prompt with same name already exists
        """
        if prompt.name in self._prompts:
            raise ValueError(f"Prompt with name '{prompt.name}' already registered")

        self._prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")

    def unregister(self, name: str) -> None:
        """
        Unregister an MCP prompt.

        Args:
            name: Prompt name to unregister
        """
        if name in self._prompts:
            del self._prompts[name]
            logger.info(f"Unregistered prompt: {name}")

    def get(self, name: str) -> Optional[MCPPrompt]:
        """
        Get prompt by name.

        Args:
            name: Prompt name

        Returns:
            Prompt instance or None if not found
        """
        return self._prompts.get(name)

    def list_prompts(self) -> List[MCPPrompt]:
        """
        List all registered prompts.

        Returns:
            List of registered prompts
        """
        return list(self._prompts.values())

    def list_prompt_names(self) -> List[str]:
        """
        List all registered prompt names.

        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())

    def get_prompt_schemas(self) -> List[Dict[str, Any]]:
        """
        Get prompt schemas for all registered prompts.

        Returns:
            List of prompt schemas in MCP format
        """
        return [prompt.prompt_schema for prompt in self._prompts.values()]

    def has_prompt(self, name: str) -> bool:
        """
        Check if prompt is registered.

        Args:
            name: Prompt name

        Returns:
            True if prompt is registered, False otherwise
        """
        return name in self._prompts

    def find_by_pattern(self, pattern: str) -> List[MCPPrompt]:
        """
        Find prompts matching name pattern.

        Args:
            pattern: Name pattern to match (supports wildcards)

        Returns:
            List of matching prompts
        """
        import fnmatch
        return [
            prompt
            for name, prompt in self._prompts.items()
            if fnmatch.fnmatch(name, pattern)
        ]

    def clear(self) -> None:
        """Clear all registered prompts."""
        self._prompts.clear()
        logger.info("Cleared all prompts")

    def __len__(self) -> int:
        """Get number of registered prompts."""
        return len(self._prompts)

    def __contains__(self, name: str) -> bool:
        """Check if prompt is registered."""
        return name in self._prompts

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"<PromptRegistry prompts={len(self._prompts)}>"
