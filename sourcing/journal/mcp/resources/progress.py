"""
Progress resources for MCP Server.

This module provides resources for accessing research progress data.
"""

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.api.services.command_handler_service import (
  CommandHandlerService,
)
from ai.sourcing.journal.mcp.protocol import (
  JSONRPCErrorCode,
  MCPError,
  MCPErrorCode,
)
from ai.sourcing.journal.mcp.resources.base import MCPResource

logger = logging.getLogger(__name__)


class ProgressMetricsResource(MCPResource):
  """Resource for accessing current progress metrics for a session."""

  def __init__(self, service: CommandHandlerService) -> None:
    """
    Initialize ProgressMetricsResource.

    Args:
      service: CommandHandlerService instance
    """
    self.service = service
    super().__init__(
      uri="research://progress/metrics/{session_id}",
      name="Progress Metrics",
      description="Current progress metrics for a research session",
      mime_type="application/json",
    )

  async def read(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read progress metrics for a session.

    Args:
      params: Parameters containing session_id

    Returns:
      Progress metrics content

    Raises:
      MCPError: If session_id is missing or session not found
    """
    if not params or "session_id" not in params:
      raise MCPError(
        JSONRPCErrorCode.INVALID_PARAMS,
        "Missing required parameter: session_id",
      )

    session_id = params["session_id"]
    if not isinstance(session_id, str):
      raise MCPError(
        JSONRPCErrorCode.INVALID_PARAMS,
        "session_id must be a string",
      )

    try:
      orchestrator = self.service.orchestrator
      progress = orchestrator.get_progress(session_id)

      # Convert to dict for JSON serialization
      progress_dict = asdict(progress)

      content = self.format_content(progress_dict)
      return {"contents": [content]}

    except KeyError:
      raise MCPError(
        MCPErrorCode.RESOURCE_NOT_FOUND,
        f"Session {session_id} not found",
      )
    except Exception as e:
      logger.exception(f"Error reading progress metrics for session {session_id}")
      raise MCPError(
        JSONRPCErrorCode.INTERNAL_ERROR,
        f"Failed to read progress metrics: {str(e)}",
      )

  def validate_uri(self, uri: str) -> bool:
    """
    Validate if URI matches this resource pattern.

    Args:
      uri: URI to validate

    Returns:
      True if URI matches pattern, False otherwise
    """
    return uri.startswith("research://progress/metrics/")

  def extract_session_id(self, uri: str) -> Optional[str]:
    """
    Extract session ID from URI.

    Args:
      uri: Resource URI

    Returns:
      Session ID or None if invalid
    """
    if not uri.startswith("research://progress/metrics/"):
      return None
    return uri.replace("research://progress/metrics/", "")


class ProgressHistoryResource(MCPResource):
  """Resource for accessing progress history for a session."""

  def __init__(self, service: CommandHandlerService) -> None:
    """
    Initialize ProgressHistoryResource.

    Args:
      service: CommandHandlerService instance
    """
    self.service = service
    super().__init__(
      uri="research://progress/history/{session_id}",
      name="Progress History",
      description="Historical progress snapshots for a research session",
      mime_type="application/json",
    )

  async def read(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read progress history for a session.

    Args:
      params: Parameters containing session_id

    Returns:
      Progress history content

    Raises:
      MCPError: If session_id is missing or session not found
    """
    if not params or "session_id" not in params:
      raise MCPError(
        JSONRPCErrorCode.INVALID_PARAMS,
        "Missing required parameter: session_id",
      )

    session_id = params["session_id"]
    if not isinstance(session_id, str):
      raise MCPError(
        JSONRPCErrorCode.INVALID_PARAMS,
        "session_id must be a string",
      )

    try:
      orchestrator = self.service.orchestrator
      history = orchestrator.get_progress_history(session_id)

      # Convert snapshots to dicts for JSON serialization
      history_dicts = [asdict(snapshot) for snapshot in history]

      content = self.format_content(history_dicts)
      return {"contents": [content]}

    except KeyError:
      raise MCPError(
        MCPErrorCode.RESOURCE_NOT_FOUND,
        f"Session {session_id} not found",
      )
    except Exception as e:
      logger.exception(f"Error reading progress history for session {session_id}")
      raise MCPError(
        JSONRPCErrorCode.INTERNAL_ERROR,
        f"Failed to read progress history: {str(e)}",
      )

  def validate_uri(self, uri: str) -> bool:
    """
    Validate if URI matches this resource pattern.

    Args:
      uri: URI to validate

    Returns:
      True if URI matches pattern, False otherwise
    """
    return uri.startswith("research://progress/history/")

  def extract_session_id(self, uri: str) -> Optional[str]:
    """
    Extract session ID from URI.

    Args:
      uri: Resource URI

    Returns:
      Session ID or None if invalid
    """
    if not uri.startswith("research://progress/history/"):
      return None
    return uri.replace("research://progress/history/", "")

