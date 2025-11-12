"""
Session resources for MCP Server.

This module provides resources for accessing research session state.
"""

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from ai.journal_dataset_research.api.services.command_handler_service import (
  CommandHandlerService,
)
from ai.journal_dataset_research.mcp.protocol import (
  JSONRPCErrorCode,
  MCPError,
  MCPErrorCode,
)
from ai.journal_dataset_research.mcp.resources.base import MCPResource

logger = logging.getLogger(__name__)


class SessionStateResource(MCPResource):
  """Resource for accessing session state for a research session."""

  def __init__(self, service: CommandHandlerService) -> None:
    """
    Initialize SessionStateResource.

    Args:
      service: CommandHandlerService instance
    """
    self.service = service
    super().__init__(
      uri="research://sessions/{session_id}/state",
      name="Session State",
      description="Complete session state including sources, evaluations, acquisitions, and integration plans",
      mime_type="application/json",
    )

  async def read(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read session state.

    Args:
      params: Parameters containing session_id

    Returns:
      Session state content

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
      state = orchestrator.get_session_state(session_id)

      # Convert to dict for JSON serialization
      state_dict = asdict(state)

      content = self.format_content(state_dict)
      return {"contents": [content]}

    except KeyError:
      raise MCPError(
        MCPErrorCode.RESOURCE_NOT_FOUND,
        f"Session {session_id} not found",
      )
    except Exception as e:
      logger.exception(f"Error reading session state for session {session_id}")
      raise MCPError(
        JSONRPCErrorCode.INTERNAL_ERROR,
        f"Failed to read session state: {str(e)}",
      )

  def validate_uri(self, uri: str) -> bool:
    """
    Validate if URI matches this resource pattern.

    Args:
      uri: URI to validate

    Returns:
      True if URI matches pattern, False otherwise
    """
    return uri.startswith("research://sessions/") and uri.endswith("/state")

  def extract_session_id(self, uri: str) -> Optional[str]:
    """
    Extract session ID from URI.

    Args:
      uri: Resource URI

    Returns:
      Session ID or None if invalid
    """
    if not uri.startswith("research://sessions/") or not uri.endswith("/state"):
      return None
    # Extract session_id from: research://sessions/{session_id}/state
    parts = uri.replace("research://sessions/", "").replace("/state", "")
    return parts if parts else None

