"""
Metrics resources for MCP Server.

This module provides resources for accessing session metrics and activity logs.
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


class SessionMetricsResource(MCPResource):
  """Resource for accessing session metrics including activity logs and error logs."""

  def __init__(self, service: CommandHandlerService) -> None:
    """
    Initialize SessionMetricsResource.

    Args:
      service: CommandHandlerService instance
    """
    self.service = service
    super().__init__(
      uri="research://sessions/{session_id}/metrics",
      name="Session Metrics",
      description="Session metrics including activity logs, error logs, and progress reports",
      mime_type="application/json",
    )

  async def read(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read session metrics.

    Args:
      params: Parameters containing session_id and optional metric_type

    Returns:
      Session metrics content

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

    metric_type = params.get("metric_type", "all")  # all, activity, errors, report

    try:
      orchestrator = self.service.orchestrator

      metrics: Dict[str, Any] = {}

      if metric_type in ("all", "activity"):
        activity_log = orchestrator.get_activity_log(session_id)
        metrics["activity_log"] = [asdict(log) for log in activity_log]

      if metric_type in ("all", "errors"):
        error_log = orchestrator.get_error_log(session_id)
        metrics["error_log"] = error_log

      if metric_type in ("all", "report"):
        progress_report = orchestrator.generate_progress_report(session_id)
        metrics["progress_report"] = progress_report

      content = self.format_content(metrics)
      return {"contents": [content]}

    except KeyError:
      raise MCPError(
        MCPErrorCode.RESOURCE_NOT_FOUND,
        f"Session {session_id} not found",
      )
    except Exception as e:
      logger.exception(f"Error reading session metrics for session {session_id}")
      raise MCPError(
        JSONRPCErrorCode.INTERNAL_ERROR,
        f"Failed to read session metrics: {str(e)}",
      )

  def validate_uri(self, uri: str) -> bool:
    """
    Validate if URI matches this resource pattern.

    Args:
      uri: URI to validate

    Returns:
      True if URI matches pattern, False otherwise
    """
    return uri.startswith("research://sessions/") and uri.endswith("/metrics")

  def extract_session_id(self, uri: str) -> Optional[str]:
    """
    Extract session ID from URI.

    Args:
      uri: Resource URI

    Returns:
      Session ID or None if invalid
    """
    if not uri.startswith("research://sessions/") or not uri.endswith("/metrics"):
      return None
    # Extract session_id from: research://sessions/{session_id}/metrics
    parts = uri.replace("research://sessions/", "").replace("/metrics", "")
    return parts if parts else None

