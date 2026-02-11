"""
Tool Registry for MCP Server.

This module provides tool registration, discovery, and lookup functionality.
"""

import logging
from typing import Dict, List, Optional

from ai.sourcing.journal.mcp.tools.base import MCPTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for MCP tools."""

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """
        Register an MCP tool.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """
        Unregister an MCP tool.

        Args:
            name: Tool name to unregister
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Optional[MCPTool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[MCPTool]:
        """
        List all registered tools.

        Returns:
            List of registered tools
        """
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, any]]:
        """
        Get tool schemas for all registered tools.

        Returns:
            List of tool schemas in MCP format
        """
        return [tool.tool_schema for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        """
        Check if tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool is registered, False otherwise
        """
        return name in self._tools

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        logger.info("Cleared all tools")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"<ToolRegistry tools={len(self._tools)}>"

