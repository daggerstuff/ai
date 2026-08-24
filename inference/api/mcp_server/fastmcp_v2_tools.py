"""
Improved Foresight MCP server tools following MCP best practices.

This module provides memory management tools with:
- Proper tool annotations (readOnlyHint, destructiveHint, etc.)
- Structured output with outputSchema
- Response format support (JSON/Markdown)
- Pagination metadata
- Actionable error messages
"""

from __future__ import annotations

import json
from enum import StrEnum

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from ai.inference.api.mcp_server.fastmcp_protocols import MemoryRemover, MemoryUpdater
from ai.inference.api.mcp_server.memory_scope import memory_in_scope
from ai.inference.api.memory.memory_status_service import build_memory_status_summary

from .fastmcp_parsing import parse_metadata, parse_scope_context
from .fastmcp_presenters import memory_store_success_message
from .fastmcp_search import get_scoped_recent_memories, search_scoped_memories
from .fastmcp_shared import _stdio_trust_enabled, authorized_tool_context_from_json, stdio_trusted_tool_context
from .fastmcp_store import (
    ScopeEnrichedMemoryCreator,
    build_memory_store_plan,
    persist_memory_store_plan,
    scope_config_from_parsed,
)

# =============================================================================
# Enums
# =============================================================================


class ResponseFormat(StrEnum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


# =============================================================================
# Pydantic Models
# =============================================================================


class MemoryStoreInput(BaseModel):
    """Input model for storing a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    content: str = Field(
        ...,
        description="The significant fact, preference, or insight to store",
        min_length=1,
        max_length=10000,
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user (e.g., 'user-123', 'vivi')",
        min_length=1,
    )
    category: str = Field(
        default="fact",
        description="Category for the memory (e.g., 'fact', 'preference', 'context')",
    )
    metadata: str | None = Field(
        default=None,
        description="Additional metadata as a JSON string",
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON (org_id, project_id, session_id, etc.)",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (human-readable) or 'json' (machine-readable)",
    )


class MemoryQueryInput(BaseModel):
    """Input model for querying memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    query: str = Field(
        ...,
        description="Search query to find relevant memories",
        min_length=1,
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0,
        description="Number of results to skip for pagination",
        ge=0,
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


class MemoryGetInput(BaseModel):
    """Input model for retrieving a single memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    memory_id: str = Field(
        ...,
        description="Unique identifier of the memory to retrieve",
        min_length=1,
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


class MemoryListInput(BaseModel):
    """Input model for listing all memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    limit: int = Field(
        default=20,
        description="Maximum number of memories to return",
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0,
        description="Number of memories to skip for pagination",
        ge=0,
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


class MemoryUpdateInput(BaseModel):
    """Input model for updating a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    memory_id: str = Field(
        ...,
        description="Unique identifier of the memory to update",
        min_length=1,
    )
    content: str = Field(
        ...,
        description="Updated content for the memory",
        min_length=1,
        max_length=10000,
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    metadata: str | None = Field(
        default=None,
        description="Updated metadata as JSON string",
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


class MemoryDeleteInput(BaseModel):
    """Input model for deleting a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    memory_id: str = Field(
        ...,
        description="Unique identifier of the memory to delete",
        min_length=1,
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


class MemoryStatusInput(BaseModel):
    """Input model for memory status."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    scope_context: str | None = Field(
        default=None,
        description="Scope context as JSON",
    )
    auth_context: str | None = Field(
        default=None,
        description="HMAC authentication context as JSON",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _format_error(message: str, suggestion: str | None = None) -> str:
    """Format error messages with actionable guidance."""
    if suggestion:
        return f"Error: {message}\nSuggestion: {suggestion}"
    return f"Error: {message}"


def _format_memory_results_json(
    results: list[dict],
    user_id: str,
    query: str,
    limit: int,
    offset: int,
) -> str:
    """Format search results as JSON with pagination metadata."""
    return json.dumps(
        {
            "user_id": user_id,
            "query": query,
            "total": len(results),
            "count": min(len(results), limit),
            "offset": offset,
            "has_more": len(results) > limit + offset,
            "next_offset": offset + limit if len(results) > limit + offset else None,
            "memories": [
                {
                    "id": r.get("id"),
                    "content": r.get("memory") or r.get("content") or r.get("text"),
                    "score": r.get("score", 0.0),
                    "category": r.get("metadata", {}).get("category"),
                    "created_at": r.get("created_at"),
                }
                for r in results[offset : offset + limit]
            ],
        },
        indent=2,
    )


def _format_memory_results_markdown(
    results: list[dict],
    user_id: str,
    query: str,
    limit: int,
    offset: int,
) -> str:
    """Format search results as human-readable markdown."""
    lines = [
        f"### Memory Search Results: '{query}'",
        f"**User:** {user_id}",
        f"**Showing:** {min(len(results) - offset, limit)} of {len(results)} results",
        "",
    ]

    for i, r in enumerate(results[offset : offset + limit], start=offset + 1):
        content = r.get("memory") or r.get("content") or r.get("text", "N/A")
        score = r.get("score", 0.0)
        category = r.get("metadata", {}).get("category", "unknown")
        lines.append(f"{i}. [{score:.2f}] {content}")
        lines.append(f"   - Category: {category}")
        lines.append("")

    if len(results) > limit + offset:
        lines.append(f"_Use offset={offset + limit} to see more results._")

    return "\n".join(lines)


# =============================================================================
# Tool Definitions
# =============================================================================


async def foresight_store_memory(params: MemoryStoreInput) -> str:
    """Store a significant fact, preference, or insight in long-term memory.

    This tool persists information to the user's memory bank, making it available
    for future retrieval via search or listing operations. Memories can be scoped
    to specific contexts (org, project, session) for targeted retrieval.

    Args:
        params: Validated input containing:
            - content: The information to store (fact, preference, insight)
            - user_id: User identifier for memory ownership
            - category: Classification of memory type
            - metadata: Optional additional context as JSON
            - scope_context: Optional scoping for organized retrieval
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Confirmation with memory ID and stored content.

    Examples:
        - Store project context: content="Project uses Python 3.11", category="context"
        - Store user preference: content="Prefers dark mode", category="preference"
        - Store scoped fact: content="API key is stored in .env", scope_context='{"project_id": "myapp"}'
    """
    metadata_dict = parse_metadata(params.metadata)
    context = authorized_tool_context_from_json(
        tool_name="foresight_store_memory",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "content": params.content,
            "user_id": params.user_id,
            "category": params.category,
            "metadata": metadata_dict,
        },
        visibility_default="private",
    )
    authorized_user_id = context.scope.user_id
    scope_config = scope_config_from_parsed(parse_scope_context(params.scope_context))

    try:
        plan = build_memory_store_plan(
            content=params.content,
            user_id=authorized_user_id,
            category=params.category,
            metadata_dict=metadata_dict,
            scope=scope_config,
        )
        result = persist_memory_store_plan(
            creator=ScopeEnrichedMemoryCreator(manager=context.manager),
            plan=plan,
        )

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "success": True,
                    "user_id": plan.user_id,
                    "content": plan.content,
                    "category": plan.category,
                    "memory_id": result if isinstance(result, str) else None,
                },
                indent=2,
            )

        return memory_store_success_message(
            user_id=plan.user_id,
            content=plan.content,
            category=plan.category,
            result=result,
        )
    except Exception as exc:
        return _format_error(
            f"Failed to store memory: {exc}", "Check that the content is valid and you have write permissions."
        )


async def foresight_query_memories(params: MemoryQueryInput) -> str:
    """Search long-term memory for relevant information using semantic search.

    This tool performs semantic search across stored memories, returning results
    ranked by relevance. Use this when you need to recall specific information
    that matches a query or concept.

    Args:
        params: Validated input containing:
            - query: Search terms or natural language query
            - user_id: User whose memories to search
            - limit: Maximum results to return (1-100, default 5)
            - offset: Pagination offset for large result sets
            - scope_context: Optional scope filter (org, project, session)
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Search results with relevance scores and pagination metadata.

    Examples:
        - Find project decisions: query="why did we choose PostgreSQL"
        - Find recent context: query="last week's meeting notes", limit=10
        - Scoped search: query="API endpoint", scope_context='{"project_id": "myapp"}'
    """
    context = authorized_tool_context_from_json(
        tool_name="foresight_query_memories",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "query": params.query,
            "user_id": params.user_id,
            "limit": params.limit,
            "offset": params.offset,
            "scope_context": params.scope_context,
        },
    )

    try:
        results = search_scoped_memories(
            manager=context.manager,
            query=params.query,
            user_id=context.scope.user_id,
            scope=context.scope,
            limit=params.limit + params.offset + 10,  # Fetch extra for pagination
        )

        if not results:
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(
                    {
                        "user_id": context.scope.user_id,
                        "query": params.query,
                        "total": 0,
                        "memories": [],
                    },
                    indent=2,
                )
            return f"No memories found matching '{params.query}' for user {params.user_id}."

        if params.response_format == ResponseFormat.JSON:
            return _format_memory_results_json(
                results, context.scope.user_id, params.query, params.limit, params.offset
            )
        return _format_memory_results_markdown(
            results, context.scope.user_id, params.query, params.limit, params.offset
        )

    except Exception as exc:
        return _format_error(f"Failed to query memories: {exc}", "Check your query syntax and try with simpler terms.")


async def foresight_get_memory(params: MemoryGetInput) -> str:
    """Retrieve a single memory entry by its unique identifier.

    This tool fetches the complete content and metadata of a specific memory
    when you know its ID. Use this for direct access to stored information.

    Args:
        params: Validated input containing:
            - memory_id: Unique identifier of the memory
            - user_id: User who owns the memory
            - scope_context: Optional scope filter
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Complete memory details including content, metadata, and timestamps.

    Examples:
        - Get by ID: memory_id="mem-abc123", user_id="vivi"
    """
    context = authorized_tool_context_from_json(
        tool_name="foresight_get_memory",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "memory_id": params.memory_id,
            "user_id": params.user_id,
        },
    )

    try:
        # Check if memory is in scope before returning
        if not memory_in_scope(
            manager=context.manager,
            scope=context.scope,
            memory_id=params.memory_id,
        ):
            return _format_error(
                f"Memory '{params.memory_id}' not found in your scope.",
                "Verify the memory ID is correct and you have access to it.",
            )

        # Get the memory record
        if not hasattr(context.manager, "get_memory"):
            return _format_error(
                "Memory retrieval not supported by this backend.",
                "Use foresight_query_memories or foresight_list_memories instead.",
            )

        record = context.manager.get_memory(params.memory_id)
        if not record:
            return _format_error(
                f"Memory '{params.memory_id}' not found.", "The memory may have been deleted or the ID is incorrect."
            )

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "id": record.get("id"),
                    "content": record.get("content") or record.get("memory") or record.get("text"),
                    "user_id": context.scope.user_id,
                    "category": record.get("metadata", {}).get("category"),
                    "metadata": record.get("metadata", {}),
                    "created_at": record.get("created_at"),
                },
                indent=2,
            )

        content = record.get("content") or record.get("memory") or record.get("text", "N/A")
        category = record.get("metadata", {}).get("category", "unknown")
        created = record.get("created_at", "unknown")

        return "\n".join(
            [
                f"### Memory: {params.memory_id}",
                f"**User:** {context.scope.user_id}",
                f"**Category:** {category}",
                f"**Created:** {created}",
                "",
                "**Content:**",
                content,
            ]
        )

    except Exception as exc:
        return _format_error(f"Failed to retrieve memory: {exc}", "Check the memory ID and your access permissions.")


async def foresight_list_memories(params: MemoryListInput) -> str:
    """List all memories for a user with pagination support.

    This tool retrieves a paginated list of all memories, optionally filtered
    by scope. Use this when you need to browse or enumerate memories rather
    than search for specific content.

    Args:
        params: Validated input containing:
            - user_id: User whose memories to list
            - limit: Maximum memories to return (1-100, default 20)
            - offset: Pagination offset
            - scope_context: Optional scope filter
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Paginated list of memories with content previews.

    Examples:
        - List all: user_id="vivi", limit=50
        - Paginated: user_id="vivi", limit=20, offset=20
        - Scoped list: scope_context='{"project_id": "myapp"}'
    """
    context = authorized_tool_context_from_json(
        tool_name="foresight_list_memories",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "user_id": params.user_id,
            "limit": params.limit,
            "offset": params.offset,
        },
    )

    try:
        memories = get_scoped_recent_memories(
            manager=context.manager,
            user_id=context.scope.user_id,
            scope=context.scope,
            limit=params.limit + params.offset + 10,  # Fetch extra for pagination
        )

        if not memories:
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(
                    {
                        "user_id": context.scope.user_id,
                        "total": 0,
                        "memories": [],
                    },
                    indent=2,
                )
            return f"No memories found for user {params.user_id}."

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "user_id": context.scope.user_id,
                    "total": len(memories),
                    "count": min(len(memories) - params.offset, params.limit),
                    "offset": params.offset,
                    "has_more": len(memories) > params.limit + params.offset,
                    "next_offset": params.offset + params.limit
                    if len(memories) > params.limit + params.offset
                    else None,
                    "memories": [
                        {
                            "id": m.get("id"),
                            "content": (m.get("content") or m.get("memory") or m.get("text", ""))[:200],
                            "category": m.get("metadata", {}).get("category"),
                            "created_at": m.get("created_at"),
                        }
                        for m in memories[params.offset : params.offset + params.limit]
                    ],
                },
                indent=2,
            )

        lines = [
            f"### Memory List for {params.user_id}",
            f"**Showing:** {min(len(memories) - params.offset, params.limit)} of {len(memories)} memories",
            "",
        ]

        for i, m in enumerate(memories[params.offset : params.offset + params.limit], start=params.offset + 1):
            content = (m.get("content") or m.get("memory") or m.get("text", "N/A"))[:100]
            category = m.get("metadata", {}).get("category", "unknown")
            memory_id = m.get("id", "unknown")
            lines.append(f"{i}. [{memory_id}] {content}...")
            lines.append(f"   Category: {category}")
            lines.append("")

        if len(memories) > params.limit + params.offset:
            lines.append(f"_Use offset={params.offset + params.limit} to see more._")

        return "\n".join(lines)

    except Exception as exc:
        return _format_error(f"Failed to list memories: {exc}", "Try with a smaller limit or check your permissions.")


async def foresight_update_memory(params: MemoryUpdateInput) -> str:
    """Update the content of an existing memory entry.

    This tool modifies a stored memory's content while preserving its identity
    and relationships. Use this to correct or refine previously stored information.

    Args:
        params: Validated input containing:
            - memory_id: Unique identifier of memory to update
            - content: New content for the memory
            - user_id: User who owns the memory
            - metadata: Optional updated metadata as JSON
            - scope_context: Optional scope filter
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Confirmation of successful update.

    Examples:
        - Correct fact: memory_id="mem-123", content="Corrected information"
        - Refine preference: memory_id="mem-456", content="Updated preference details"
    """
    metadata_dict = parse_metadata(params.metadata)
    context = authorized_tool_context_from_json(
        tool_name="foresight_update_memory",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "memory_id": params.memory_id,
            "content": params.content,
            "user_id": params.user_id,
            "metadata": metadata_dict,
        },
    )

    try:
        authorized_user_id = context.scope.user_id
        if not isinstance(context.manager, MemoryUpdater):
            return _format_error(
                "Memory backend does not support updates.", "This backend is read-only. Contact your administrator."
            )

        if not memory_in_scope(
            manager=context.manager,
            scope=context.scope,
            memory_id=params.memory_id,
        ):
            return _format_error(
                f"Memory '{params.memory_id}' not found in your scope.",
                "Verify the memory ID and your access permissions.",
            )

        if context.manager.update_memory(
            params.memory_id,
            new_content=params.content,
            metadata=metadata_dict,
            user_id=authorized_user_id,
        ):
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(
                    {
                        "success": True,
                        "memory_id": params.memory_id,
                        "user_id": authorized_user_id,
                        "content_length": len(params.content),
                    },
                    indent=2,
                )
            return f"Memory updated successfully (ID: {params.memory_id})"

        return _format_error("Update failed for unknown reason.", "Try again or check the memory content for issues.")

    except Exception as exc:
        return _format_error(f"Failed to update memory: {exc}", "Check the memory ID and your update permissions.")


async def foresight_delete_memory(params: MemoryDeleteInput) -> str:
    """Permanently remove a memory entry from storage.

    This tool irreversibly deletes a memory. Use with caution as deleted
    memories cannot be recovered. Consider if you need to backup important
    information before deletion.

    Args:
        params: Validated input containing:
            - memory_id: Unique identifier of memory to delete
            - user_id: User who owns the memory
            - scope_context: Optional scope filter
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Confirmation of deletion.

    Examples:
        - Delete outdated: memory_id="mem-123", user_id="vivi"
        - Remove incorrect: memory_id="mem-456", user_id="vivi"
    """
    context = authorized_tool_context_from_json(
        tool_name="foresight_delete_memory",
        user_id=params.user_id,
        auth_context=params.auth_context,
        scope_context=params.scope_context,
        payload={
            "memory_id": params.memory_id,
            "user_id": params.user_id,
        },
    )

    try:
        authorized_user_id = context.scope.user_id
        if not isinstance(context.manager, MemoryRemover):
            return _format_error(
                "Memory backend does not support deletion.", "This backend is read-only. Contact your administrator."
            )

        if not memory_in_scope(
            manager=context.manager,
            scope=context.scope,
            memory_id=params.memory_id,
        ):
            return _format_error(
                f"Memory '{params.memory_id}' not found in your scope.",
                "Verify the memory ID and your access permissions.",
            )

        if context.manager.delete_memory(params.memory_id, user_id=authorized_user_id):
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(
                    {
                        "success": True,
                        "memory_id": params.memory_id,
                        "user_id": authorized_user_id,
                    },
                    indent=2,
                )
            return f"Memory deleted successfully (ID: {params.memory_id})"

        return _format_error("Deletion failed for unknown reason.", "The memory may have already been deleted.")

    except Exception as exc:
        return _format_error(f"Failed to delete memory: {exc}", "Check the memory ID and your delete permissions.")


async def foresight_memory_status(params: MemoryStatusInput) -> str:
    """Get statistics and health information for the user's memory bank.

    This tool provides an overview of stored memories including counts by
    category and overall health status. Use this to understand memory
    usage and identify potential issues.

    Args:
        params: Validated input containing:
            - user_id: User whose status to check
            - scope_context: Optional scope filter
            - auth_context: HMAC authentication (or use STDIO_TRUST mode)
            - response_format: 'markdown' or 'json' output

    Returns:
        str: Memory statistics including total count and category breakdown.

    Examples:
        - Check status: user_id="vivi"
        - Scoped status: scope_context='{"project_id": "myapp"}'
    """
    if not params.auth_context:
        if _stdio_trust_enabled():
            context = stdio_trusted_tool_context(
                user_id=params.user_id,
                scope_context=params.scope_context,
            )
        else:
            return _format_error(
                "Authentication required.", "Provide auth_context or enable HINDSIGHT_MCP_STDIO_TRUST."
            )
    else:
        context = authorized_tool_context_from_json(
            tool_name="foresight_memory_status",
            user_id=params.user_id,
            auth_context=params.auth_context,
            scope_context=params.scope_context,
            payload={
                "user_id": params.user_id,
            },
        )

    try:
        summary = build_memory_status_summary(
            manager=context.manager,
            scope=context.scope,
            user_id=context.scope.user_id,
        )

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "user_id": context.scope.user_id,
                    "total_memories": summary.total_memories,
                    "is_sampled": summary.is_sampled,
                    "health": summary.health,
                    "categories": dict(summary.categories),
                },
                indent=2,
            )

        if summary.total_memories == 0:
            return f"### Memory Status: {context.scope.user_id}\n\nNo memories stored yet."

        total_label = "Sampled Memories" if summary.is_sampled else "Total Memories"
        category_lines = "\n".join(f"- **{key}:** {value}" for key, value in summary.categories.items())

        return "\n".join(
            [
                f"### Memory Status: {context.scope.user_id}",
                "",
                f"**{total_label}:** {summary.total_memories}",
                f"**Health:** {summary.health}",
                "",
                "**Category Breakdown:**",
                category_lines,
            ]
        )

    except Exception as exc:
        return _format_error(f"Failed to get memory status: {exc}", "Check your authentication and try again.")


# =============================================================================
# Registration Function
# =============================================================================


def register_memory_tools_v2(mcp: FastMCP) -> None:
    """Register all improved Foresight memory tools with the MCP server.

    This function registers tools following MCP best practices:
    - Service-prefixed names (foresight_*)
    - Proper tool annotations
    - Pydantic input validation
    - Structured output support
    """
    # Store - creates new memories
    mcp.tool(
        name="foresight_store_memory",
        annotations={
            "title": "Store Memory",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )(foresight_store_memory)

    # Query - search memories
    mcp.tool(
        name="foresight_query_memories",
        annotations={
            "title": "Query Memories",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )(foresight_query_memories)

    # Get - retrieve single memory
    mcp.tool(
        name="foresight_get_memory",
        annotations={
            "title": "Get Memory",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )(foresight_get_memory)

    # List - enumerate all memories
    mcp.tool(
        name="foresight_list_memories",
        annotations={
            "title": "List Memories",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )(foresight_list_memories)

    # Update - modify existing memory
    mcp.tool(
        name="foresight_update_memory",
        annotations={
            "title": "Update Memory",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )(foresight_update_memory)

    # Delete - remove memory
    mcp.tool(
        name="foresight_delete_memory",
        annotations={
            "title": "Delete Memory",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )(foresight_delete_memory)

    # Status - memory statistics
    mcp.tool(
        name="foresight_memory_status",
        annotations={
            "title": "Memory Status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )(foresight_memory_status)
