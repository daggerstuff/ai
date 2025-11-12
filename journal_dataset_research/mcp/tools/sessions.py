"""
Session management tools for MCP Server.

This module provides tools for managing research sessions through the MCP protocol.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.journal_dataset_research.mcp.protocol import MCPError, MCPErrorCode
from ai.journal_dataset_research.mcp.tools.base import MCPTool

logger = logging.getLogger(__name__)


class CreateSessionTool(MCPTool):
    """Tool for creating a new research session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize CreateSessionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="create_session",
            description="Create a new research session with target sources and search keywords",
            parameters={
                "type": "object",
                "properties": {
                    "target_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target sources (e.g., ['pubmed', 'doaj'])",
                    },
                    "search_keywords": {
                        "type": "object",
                        "description": "Dictionary of search keywords by category",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "example": {
                            "therapeutic": ["therapy", "counseling"],
                            "dataset": ["dataset", "conversation"],
                        },
                    },
                    "weekly_targets": {
                        "type": "object",
                        "description": "Dictionary of weekly targets (optional)",
                        "additionalProperties": {"type": "integer"},
                        "example": {
                            "sources_identified": 10,
                            "datasets_evaluated": 5,
                        },
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional custom session ID (auto-generated if not provided)",
                    },
                },
                "required": ["target_sources", "search_keywords"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute create_session tool.

        Args:
            params: Tool parameters

        Returns:
            Session details
        """
        try:
            target_sources = params.get("target_sources", [])
            search_keywords = params.get("search_keywords", {})
            weekly_targets = params.get("weekly_targets")
            session_id = params.get("session_id")

            # Validate parameters
            if not target_sources:
                raise ValueError("target_sources cannot be empty")
            if not search_keywords:
                raise ValueError("search_keywords cannot be empty")

            # Create session
            session = self.service.create_session(
                target_sources=target_sources,
                search_keywords=search_keywords,
                weekly_targets=weekly_targets,
                session_id=session_id,
            )

            # Convert session to dict
            return {
                "session_id": session.session_id,
                "start_date": session.start_date.isoformat(),
                "target_sources": session.target_sources,
                "search_keywords": session.search_keywords,
                "weekly_targets": session.weekly_targets,
                "current_phase": session.current_phase,
                "progress_metrics": session.progress_metrics,
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error creating session: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to create session: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e


class ListSessionsTool(MCPTool):
    """Tool for listing all research sessions."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize ListSessionsTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="list_sessions",
            description="List all research sessions",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute list_sessions tool.

        Args:
            params: Tool parameters (empty)

        Returns:
            List of sessions with metadata
        """
        try:
            sessions = self.service.list_sessions()

            # Convert sessions to dicts
            sessions_list = []
            for session in sessions:
                sessions_list.append(
                    {
                        "session_id": session.session_id,
                        "start_date": session.start_date.isoformat(),
                        "target_sources": session.target_sources,
                        "search_keywords": session.search_keywords,
                        "weekly_targets": session.weekly_targets,
                        "current_phase": session.current_phase,
                        "progress_metrics": session.progress_metrics,
                    }
                )

            return {
                "sessions": sessions_list,
                "count": len(sessions_list),
            }
        except Exception as e:
            logger.exception(f"Error listing sessions: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to list sessions: {str(e)}",
                {"error": str(e)},
            ) from e


class GetSessionTool(MCPTool):
    """Tool for getting a specific research session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetSessionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_session",
            description="Get details of a specific research session",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_session tool.

        Args:
            params: Tool parameters

        Returns:
            Session details
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Get session
            session = self.service.get_session(session_id)

            # Convert session to dict
            return {
                "session_id": session.session_id,
                "start_date": session.start_date.isoformat(),
                "target_sources": session.target_sources,
                "search_keywords": session.search_keywords,
                "weekly_targets": session.weekly_targets,
                "current_phase": session.current_phase,
                "progress_metrics": session.progress_metrics,
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting session: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get session: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e


class UpdateSessionTool(MCPTool):
    """Tool for updating a research session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize UpdateSessionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="update_session",
            description="Update a research session configuration",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "target_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target sources (optional)",
                    },
                    "search_keywords": {
                        "type": "object",
                        "description": "Dictionary of search keywords by category (optional)",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "weekly_targets": {
                        "type": "object",
                        "description": "Dictionary of weekly targets (optional)",
                        "additionalProperties": {"type": "integer"},
                    },
                    "current_phase": {
                        "type": "string",
                        "description": "Current phase (discovery, evaluation, acquisition, integration)",
                        "enum": ["discovery", "evaluation", "acquisition", "integration"],
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute update_session tool.

        Args:
            params: Tool parameters

        Returns:
            Updated session details
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            target_sources = params.get("target_sources")
            search_keywords = params.get("search_keywords")
            weekly_targets = params.get("weekly_targets")
            current_phase = params.get("current_phase")

            # Update session
            session = self.service.update_session(
                session_id=session_id,
                target_sources=target_sources,
                search_keywords=search_keywords,
                weekly_targets=weekly_targets,
                current_phase=current_phase,
            )

            # Convert session to dict
            return {
                "session_id": session.session_id,
                "start_date": session.start_date.isoformat(),
                "target_sources": session.target_sources,
                "search_keywords": session.search_keywords,
                "weekly_targets": session.weekly_targets,
                "current_phase": session.current_phase,
                "progress_metrics": session.progress_metrics,
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error updating session: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to update session: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e


class DeleteSessionTool(MCPTool):
    """Tool for deleting a research session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize DeleteSessionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="delete_session",
            description="Delete a research session",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute delete_session tool.

        Args:
            params: Tool parameters

        Returns:
            Deletion result
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Delete session
            self.service.delete_session(session_id)

            return {
                "session_id": session_id,
                "deleted": True,
                "message": f"Session {session_id} deleted successfully",
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error deleting session: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to delete session: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

