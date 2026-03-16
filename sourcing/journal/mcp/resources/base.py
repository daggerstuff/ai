"""
Base resource class for MCP resources.

This module provides the base MCPResource class that all MCP resources must inherit from.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPResource(ABC):
  """Base class for MCP resources."""

  def __init__(
    self,
    uri: str,
    name: str,
    description: str,
    mime_type: str = "application/json",
  ) -> None:
    """
    Initialize MCP resource.

    Args:
      uri: Resource URI (must be unique)
      name: Resource name
      description: Resource description
      mime_type: MIME type of resource content
    """
    self.uri = uri
    self.name = name
    self.description = description
    self.mime_type = mime_type

  @property
  def resource_schema(self) -> Dict[str, Any]:
    """
    Get resource schema in MCP format.

    Returns:
      Resource schema dictionary
    """
    return {
      "uri": self.uri,
      "name": self.name,
      "description": self.description,
      "mimeType": self.mime_type,
    }

  @abstractmethod
  async def read(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read resource content.

    Args:
      params: Optional parameters for reading resource

    Returns:
      Resource content in MCP format:
      {
        "contents": [
          {
            "uri": str,
            "mimeType": str,
            "text": str (optional),
            "blob": str (optional, base64 encoded)
          }
        ]
      }

    Raises:
      ValueError: If parameters are invalid
      Exception: If resource read fails
    """
    pass

  async def list(self) -> List[Dict[str, Any]]:
    """
    List sub-resources (if applicable).

    Returns:
      List of sub-resource schemas in MCP format
    """
    # Default implementation returns empty list
    # Override in subclasses that support listing
    return []

  def validate_uri(self, uri: str) -> bool:
    """
    Validate if URI matches this resource.

    Args:
      uri: URI to validate

    Returns:
      True if URI matches, False otherwise
    """
    return uri == self.uri

  def format_content(
    self,
    content: Any,
    mime_type: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Format content for MCP response.

    Args:
      content: Content to format
      mime_type: Optional MIME type override

    Returns:
      Formatted content dictionary
    """
    mime = mime_type or self.mime_type

    # Format based on MIME type
    if mime == "application/json":
      if isinstance(content, (dict, list)):
        text = json.dumps(content, indent=2, default=str)
      else:
        text = str(content)
    elif mime.startswith("text/"):
      text = str(content)
    else:
      # For binary content, would need base64 encoding
      # For now, convert to string
      text = str(content)

    return {
      "uri": self.uri,
      "mimeType": mime,
      "text": text,
    }

