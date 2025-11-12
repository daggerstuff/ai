"""
Tool execution framework for MCP Server.

This module provides tool execution with parameter validation, error handling,
and result formatting.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.mcp.protocol import (
    JSONRPCErrorCode,
    MCPError,
    MCPErrorCode,
)
from ai.journal_dataset_research.mcp.tools.base import MCPTool
from ai.journal_dataset_research.mcp.tools.registry import ToolRegistry
from ai.journal_dataset_research.mcp.utils.error_handling import MCPErrorHandler

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles tool execution with validation and error handling."""

    def __init__(self, registry: ToolRegistry) -> None:
        """
        Initialize tool executor.

        Args:
            registry: Tool registry instance
        """
        self.registry = registry

    async def execute_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            params: Tool parameters
            timeout: Optional timeout in seconds

        Returns:
            Tool execution result

        Raises:
            MCPError: If tool execution fails
        """
        # Get tool from registry
        tool = self.registry.get(tool_name)
        if tool is None:
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Tool not found: {tool_name}",
                {"tool_name": tool_name},
            )

        # Validate parameters
        params = params or {}
        try:
            tool.validate_parameters(params)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Parameter validation failed: {str(e)}",
                {"tool_name": tool_name, "params": params, "error": str(e)},
            ) from e

        # Execute tool with optional timeout
        try:
            if timeout:
                result = await asyncio.wait_for(
                    tool.execute(params),
                    timeout=timeout,
                )
            else:
                result = await tool.execute(params)

            # Validate result is a dictionary
            if not isinstance(result, dict):
                logger.warning(
                    f"Tool {tool_name} returned non-dict result, wrapping in dict"
                )
                result = {"result": result}

            return result

        except asyncio.TimeoutError as e:
            raise MCPError(
                MCPErrorCode.TOOL_TIMEOUT,
                f"Tool execution timed out after {timeout} seconds",
                {"tool_name": tool_name, "timeout": timeout},
            ) from e

        except MCPError:
            # Re-raise MCP errors
            raise

        except Exception as e:
            # Wrap other exceptions
            logger.exception(f"Error executing tool {tool_name}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Tool execution failed: {str(e)}",
                {"tool_name": tool_name, "error_type": type(e).__name__},
            ) from e

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools with their schemas.

        Returns:
            List of tool schemas
        """
        return self.registry.get_tool_schemas()

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool schema by name.

        Args:
            tool_name: Tool name

        Returns:
            Tool schema or None if not found
        """
        tool = self.registry.get(tool_name)
        if tool:
            return tool.tool_schema
        return None

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if tool is available.

        Args:
            tool_name: Tool name

        Returns:
            True if tool exists, False otherwise
        """
        return self.registry.has_tool(tool_name)



