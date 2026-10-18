#!/usr/bin/env python3
"""
MCP Memory Server - Stdio Protocol Implementation.

Provides MCP-compliant memory operations via stdio (not HTTP).
This is the correct implementation for MCP client integration.
"""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ai.api.mcp_server.memory_scope import (
    build_scope_metadata,
    filter_memories_by_scope,
    scope_from_kwargs,
    scope_input_schema_properties,
    search_with_overfetch,
)
from ai.memory.manager_factory import get_required_memory_manager

logger = logging.getLogger(__name__)


# Initialize server
app = Server("pixelated-memory")

# Memory client initialization
_memory_client = None


def get_memory_client():
    """Get or create the configured shared memory manager."""
    global _memory_client
    if _memory_client is None:
        _memory_client = get_required_memory_manager()
        logger.info(
            "Initialized shared memory manager: %s",
            type(_memory_client).__name__,
        )
    return _memory_client


def _scope_from_arguments(arguments: Any):
    return scope_from_kwargs(
        user_id=arguments["user_id"],
        org_id=arguments.get("org_id"),
        project_id=arguments.get("project_id"),
        session_id=arguments.get("session_id"),
        agent_id=arguments.get("agent_id"),
        run_id=arguments.get("run_id"),
        visibility=arguments.get("visibility", "private"),
        include_shared=arguments.get("include_shared", True),
    )


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools."""
    return [
        Tool(
            name="add_memory",
            description="Store information in long-term memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to remember"},
                    "user_id": {"type": "string", "description": "User identifier"},
                    "category": {"type": "string", "description": "Memory category (optional)"},
                    **scope_input_schema_properties(include_visibility=True),
                },
                "required": ["content", "user_id"],
            },
        ),
        Tool(
            name="search_memory",
            description="Search memories using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "user_id": {"type": "string", "description": "User identifier"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                    **scope_input_schema_properties(include_visibility=False),
                },
                "required": ["query", "user_id"],
            },
        ),
        Tool(
            name="get_all_memories",
            description="Get all memories for a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    **scope_input_schema_properties(include_visibility=False),
                },
                "required": ["user_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    client = get_memory_client()

    try:
        if name == "add_memory":
            scope = _scope_from_arguments(arguments)
            memory_id = client.add_memory(
                arguments["content"],
                user_id=arguments["user_id"],
                metadata=build_scope_metadata(
                    scope=scope,
                    incoming_metadata={},
                    category=arguments.get("category"),
                ),
                category=arguments.get("category"),
            )

            return [
                TextContent(
                    type="text",
                    text=f"✅ Memory stored successfully (ID: {memory_id})",
                )
            ]

        if name == "search_memory":
            scope = _scope_from_arguments(arguments)
            limit = arguments.get("limit", 10)
            memories = search_with_overfetch(
                manager=client,
                query=arguments["query"],
                user_id=arguments["user_id"],
                requested_limit=limit,
                scope=scope,
            )

            if not memories:
                return [TextContent(type="text", text="No memories found matching your query.")]

            text = f"Found {len(memories)} memories:\n\n"
            for i, mem in enumerate(memories, 1):
                content = mem.get("memory", mem.get("content", "N/A"))
                text += f"{i}. {content}\n"

            return [TextContent(type="text", text=text)]

        if name == "get_all_memories":
            scope = _scope_from_arguments(arguments)
            result = client.get_all_memories(user_id=arguments["user_id"])
            memories = result.get("results", []) if isinstance(result, dict) else result
            memories = filter_memories_by_scope(scope=scope, memories=memories or [])

            if not memories:
                return [TextContent(type="text", text="No memories stored for this user.")]

            text = f"Total memories: {len(memories)}\n\n"
            for i, mem in enumerate(memories, 1):
                content = mem.get("memory", mem.get("content", "N/A"))
                text += f"{i}. {content}\n"

            return [TextContent(type="text", text=text)]

        return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text="❌ An internal error occurred while processing the tool call.")]


async def main():
    """Run the MCP server."""
    logger.info("Starting Pixelated Memory MCP Server")

    # Initialize memory client
    get_memory_client()

    async with stdio_server() as streams:
        read_stream, write_stream = streams
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


def run():
    """Entry point for CLI (synchronous wrapper)."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
