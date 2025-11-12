"""
MCP Resources for Journal Dataset Research System.

This module provides resource implementations for research data access.
"""

import logging
from typing import Dict, List, Optional

from ai.journal_dataset_research.mcp.resources.base import MCPResource
from ai.journal_dataset_research.mcp.resources.metrics import SessionMetricsResource
from ai.journal_dataset_research.mcp.resources.progress import (
  ProgressHistoryResource,
  ProgressMetricsResource,
)
from ai.journal_dataset_research.mcp.resources.sessions import SessionStateResource

logger = logging.getLogger(__name__)

__all__ = [
  "ResourceRegistry",
  "MCPResource",
  "ProgressMetricsResource",
  "ProgressHistoryResource",
  "SessionStateResource",
  "SessionMetricsResource",
]


class ResourceRegistry:
  """Registry for MCP resources."""

  def __init__(self) -> None:
    """Initialize resource registry."""
    self._resources: Dict[str, MCPResource] = {}

  def register(self, resource: MCPResource) -> None:
    """
    Register an MCP resource.

    Args:
      resource: Resource to register

    Raises:
      ValueError: If resource with same URI already exists
    """
    if resource.uri in self._resources:
      raise ValueError(f"Resource with URI '{resource.uri}' already registered")

    self._resources[resource.uri] = resource
    logger.info(f"Registered resource: {resource.uri}")

  def unregister(self, uri: str) -> None:
    """
    Unregister an MCP resource.

    Args:
      uri: Resource URI to unregister
    """
    if uri in self._resources:
      del self._resources[uri]
      logger.info(f"Unregistered resource: {uri}")

  def get(self, uri: str) -> Optional[MCPResource]:
    """
    Get resource by URI.

    Args:
      uri: Resource URI

    Returns:
      Resource instance or None if not found
    """
    return self._resources.get(uri)

  def list_resources(self) -> List[MCPResource]:
    """
    List all registered resources.

    Returns:
      List of registered resources
    """
    return list(self._resources.values())

  def list_resource_uris(self) -> List[str]:
    """
    List all registered resource URIs.

    Returns:
      List of resource URIs
    """
    return list(self._resources.keys())

  def get_resource_schemas(self) -> List[Dict[str, any]]:
    """
    Get resource schemas for all registered resources.

    Returns:
      List of resource schemas in MCP format
    """
    return [resource.resource_schema for resource in self._resources.values()]

  def has_resource(self, uri: str) -> bool:
    """
    Check if resource is registered.

    Args:
      uri: Resource URI

    Returns:
      True if resource is registered, False otherwise
    """
    return uri in self._resources

  def find_by_pattern(self, pattern: str) -> List[MCPResource]:
    """
    Find resources matching URI pattern.

    Args:
      pattern: URI pattern to match (supports wildcards)

    Returns:
      List of matching resources
    """
    import fnmatch
    return [
      resource
      for uri, resource in self._resources.items()
      if fnmatch.fnmatch(uri, pattern)
    ]

  def clear(self) -> None:
    """Clear all registered resources."""
    self._resources.clear()
    logger.info("Cleared all resources")

  def __len__(self) -> int:
    """Get number of registered resources."""
    return len(self._resources)

  def __contains__(self, uri: str) -> bool:
    """Check if resource is registered."""
    return uri in self._resources

  def __repr__(self) -> str:
    """String representation of registry."""
    return f"<ResourceRegistry resources={len(self._resources)}>"
