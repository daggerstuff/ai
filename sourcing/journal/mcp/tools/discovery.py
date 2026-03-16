"""
Source discovery tools for MCP Server.

This module provides tools for discovering and managing dataset sources through the MCP protocol.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.tools.base import MCPTool
from ai.sourcing.journal.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class DiscoverSourcesTool(MCPTool):
    """Tool for discovering dataset sources."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize DiscoverSourcesTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="discover_sources",
            description="Discover dataset sources for a research session using keywords and target sources",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search keywords (e.g., ['therapy', 'counseling', 'dataset'])",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target sources (e.g., ['pubmed', 'doaj', 'dryad'])",
                        "enum": ["pubmed", "pubmed_central", "doaj", "dryad", "zenodo", "clinical_trials"],
                    },
                },
                "required": ["session_id", "keywords", "sources"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute discover_sources tool.

        Args:
            params: Tool parameters

        Returns:
            Discovery result with progress updates
        """
        try:
            session_id = params.get("session_id")
            keywords = params.get("keywords", [])
            sources = params.get("sources", [])

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not keywords:
                raise ValueError("keywords cannot be empty")
            if not sources:
                raise ValueError("sources cannot be empty")

            # Initiate discovery
            result = self.service.initiate_discovery(
                session_id=session_id,
                keywords=keywords,
                sources=sources,
            )

            # Get discovered sources
            sources_list = self.service.get_sources(session_id)

            # Convert sources to dicts
            sources_data = []
            for source in sources_list:
                sources_data.append(self._source_to_dict(source))

            return {
                "session_id": session_id,
                "total_sources": result.get("total_sources", len(sources_data)),
                "sources": sources_data,
                "status": "completed",
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error discovering sources: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to discover sources: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date.isoformat(),
            "discovery_method": source.discovery_method,
        }


class GetSourcesTool(MCPTool):
    """Tool for getting all sources for a session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetSourcesTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_sources",
            description="Get all discovered sources for a research session",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters for sources",
                        "properties": {
                            "source_type": {
                                "type": "string",
                                "enum": ["journal", "repository", "clinical_trial", "training_material"],
                            },
                            "open_access": {
                                "type": "boolean",
                            },
                            "data_availability": {
                                "type": "string",
                                "enum": ["available", "upon_request", "restricted", "unknown"],
                            },
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["publication_date", "discovery_date", "title"],
                        "description": "Field to sort by",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order",
                        "default": "desc",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_sources tool.

        Args:
            params: Tool parameters

        Returns:
            List of sources with optional filtering and sorting
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Get sources
            sources_list = self.service.get_sources(session_id)

            # Apply filters
            filters = params.get("filters", {})
            if filters:
                sources_list = self._apply_filters(sources_list, filters)

            # Apply sorting
            sort_by = params.get("sort_by")
            sort_order = params.get("sort_order", "desc")
            if sort_by:
                sources_list = self._apply_sorting(sources_list, sort_by, sort_order)

            # Convert sources to dicts
            sources_data = []
            for source in sources_list:
                sources_data.append(self._source_to_dict(source))

            return {
                "session_id": session_id,
                "sources": sources_data,
                "count": len(sources_data),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting sources: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get sources: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _apply_filters(self, sources: List[DatasetSource], filters: Dict[str, Any]) -> List[DatasetSource]:
        """Apply filters to sources list."""
        filtered = sources

        if "source_type" in filters:
            filtered = [s for s in filtered if s.source_type == filters["source_type"]]

        if "open_access" in filters:
            filtered = [s for s in filtered if s.open_access == filters["open_access"]]

        if "data_availability" in filters:
            filtered = [s for s in filtered if s.data_availability == filters["data_availability"]]

        return filtered

    def _apply_sorting(
        self, sources: List[DatasetSource], sort_by: str, sort_order: str
    ) -> List[DatasetSource]:
        """Apply sorting to sources list."""
        reverse = sort_order == "desc"

        if sort_by == "publication_date":
            return sorted(sources, key=lambda s: s.publication_date, reverse=reverse)
        elif sort_by == "discovery_date":
            return sorted(sources, key=lambda s: s.discovery_date, reverse=reverse)
        elif sort_by == "title":
            return sorted(sources, key=lambda s: s.title.lower(), reverse=reverse)
        else:
            return sources

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date.isoformat(),
            "discovery_method": source.discovery_method,
        }


class GetSourceTool(MCPTool):
    """Tool for getting a specific source."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetSourceTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_source",
            description="Get details of a specific dataset source",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Source ID",
                    },
                },
                "required": ["session_id", "source_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_source tool.

        Args:
            params: Tool parameters

        Returns:
            Source details
        """
        try:
            session_id = params.get("session_id")
            source_id = params.get("source_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not source_id:
                raise ValueError("source_id is required")

            # Get source
            source = self.service.get_source(session_id, source_id)

            # Convert source to dict
            return self._source_to_dict(source)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting source: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get source: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date.isoformat(),
            "discovery_method": source.discovery_method,
        }


class FilterSourcesTool(MCPTool):
    """Tool for filtering sources with advanced filtering logic."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize FilterSourcesTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="filter_sources",
            description="Filter sources with advanced filtering criteria",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Filter criteria",
                        "properties": {
                            "source_type": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["journal", "repository", "clinical_trial", "training_material"],
                                },
                                "description": "Filter by source types",
                            },
                            "open_access": {
                                "type": "boolean",
                                "description": "Filter by open access status",
                            },
                            "data_availability": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["available", "upon_request", "restricted", "unknown"],
                                },
                                "description": "Filter by data availability",
                            },
                            "keyword_match": {
                                "type": "string",
                                "description": "Filter by keyword match in title or abstract",
                            },
                            "author_match": {
                                "type": "string",
                                "description": "Filter by author name match",
                            },
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start_date": {"type": "string", "format": "date"},
                                    "end_date": {"type": "string", "format": "date"},
                                },
                                "description": "Filter by publication date range",
                            },
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["publication_date", "discovery_date", "title"],
                        "description": "Field to sort by",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order",
                        "default": "desc",
                    },
                },
                "required": ["session_id", "filters"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute filter_sources tool.

        Args:
            params: Tool parameters

        Returns:
            Filtered source list
        """
        try:
            session_id = params.get("session_id")
            filters = params.get("filters", {})

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not filters:
                raise ValueError("filters cannot be empty")

            # Get sources
            sources_list = self.service.get_sources(session_id)

            # Apply filters
            filtered_sources = self._apply_advanced_filters(sources_list, filters)

            # Apply sorting
            sort_by = params.get("sort_by")
            sort_order = params.get("sort_order", "desc")
            if sort_by:
                filtered_sources = self._apply_sorting(filtered_sources, sort_by, sort_order)

            # Convert sources to dicts
            sources_data = []
            for source in filtered_sources:
                sources_data.append(self._source_to_dict(source))

            return {
                "session_id": session_id,
                "sources": sources_data,
                "count": len(sources_data),
                "filters_applied": filters,
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error filtering sources: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to filter sources: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _apply_advanced_filters(
        self, sources: List[DatasetSource], filters: Dict[str, Any]
    ) -> List[DatasetSource]:
        """Apply advanced filters to sources list."""
        filtered = sources

        # Filter by source type
        if "source_type" in filters:
            source_types = filters["source_type"]
            if isinstance(source_types, str):
                source_types = [source_types]
            filtered = [s for s in filtered if s.source_type in source_types]

        # Filter by open access
        if "open_access" in filters:
            filtered = [s for s in filtered if s.open_access == filters["open_access"]]

        # Filter by data availability
        if "data_availability" in filters:
            availabilities = filters["data_availability"]
            if isinstance(availabilities, str):
                availabilities = [availabilities]
            filtered = [s for s in filtered if s.data_availability in availabilities]

        # Filter by keyword match
        if "keyword_match" in filters:
            keyword = filters["keyword_match"].lower()
            filtered = [
                s
                for s in filtered
                if keyword in s.title.lower() or keyword in s.abstract.lower()
            ]

        # Filter by author match
        if "author_match" in filters:
            author = filters["author_match"].lower()
            filtered = [
                s
                for s in filtered
                if any(author in a.lower() for a in s.authors)
            ]

        # Filter by date range
        if "date_range" in filters:
            date_range = filters["date_range"]
            start_date = date_range.get("start_date")
            end_date = date_range.get("end_date")

            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    filtered = [s for s in filtered if s.publication_date >= start_dt]
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid start_date format: {start_date}")

            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    filtered = [s for s in filtered if s.publication_date <= end_dt]
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid end_date format: {end_date}")

        return filtered

    def _apply_sorting(
        self, sources: List[DatasetSource], sort_by: str, sort_order: str
    ) -> List[DatasetSource]:
        """Apply sorting to sources list."""
        reverse = sort_order == "desc"

        if sort_by == "publication_date":
            return sorted(sources, key=lambda s: s.publication_date, reverse=reverse)
        elif sort_by == "discovery_date":
            return sorted(sources, key=lambda s: s.discovery_date, reverse=reverse)
        elif sort_by == "title":
            return sorted(sources, key=lambda s: s.title.lower(), reverse=reverse)
        else:
            return sources

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date.isoformat(),
            "discovery_method": source.discovery_method,
        }

