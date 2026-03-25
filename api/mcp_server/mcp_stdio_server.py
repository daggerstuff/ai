#!/usr/bin/env python3
"""
MCP Memory Server - Stdio Protocol Implementation.

Provides MCP-compliant memory operations via stdio (not HTTP).
This is the correct implementation for MCP client integration.
"""

import asyncio
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ai.api.mcp_server.memory_scope import (
    build_scope_metadata,
    filter_memories_by_scope,
    search_with_overfetch,
    scope_from_kwargs,
    scope_input_schema_properties,
)

logger = logging.getLogger(__name__)


# Initialize server
app = Server("pixelated-memory")

# Memory client initialization
_mem0_client = None


def get_mem0_client():
    """Get or create Mem0 client with null fallback."""
    global _mem0_client
    if _mem0_client is None:
        api_key = os.environ.get("MEM0_API_KEY")

        if api_key:
            try:
                from mem0 import MemoryClient

                _mem0_client = MemoryClient(api_key=api_key)
                logger.info("Initialized Mem0 Platform API client")
                return _mem0_client
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client: {e}")
        else:
            logger.info("No MEM0_API_KEY, using null memory")

        # Null implementation
        class NullMemory:
            """Complete null memory for development."""

            def add(self, content: str, user_id: str, metadata: dict = None, **kwargs):
                return {"results": [{"id": f"null-{hash(content) % 10000}"}]}

            def search(self, query: str, user_id: str, limit: int = 10, **kwargs):
                return {"results": []}

            def get_all(self, user_id: str, **kwargs):
                return {"results": []}

            def update(self, memory_id: str, text: str, **kwargs):
                return {"message": "updated"}

            def delete(self, memory_id: str, **kwargs):
                return {"message": "deleted"}

        _mem0_client = NullMemory()

    return _mem0_client


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
    client = get_mem0_client()

    try:
        if name == "add_memory":
            scope = _scope_from_arguments(arguments)
            result = client.add(
                arguments["content"],
                user_id=arguments["user_id"],
                metadata=build_scope_metadata(
                    scope=scope,
                    incoming_metadata={},
                    category=arguments.get("category"),
                ),
            )
            memory_id = "unknown"
            if isinstance(result, dict):
                if result.get("id"):
                    memory_id = result.get("id")
                elif "results" in result and result["results"]:
                    memory_id = result["results"][0].get("id", "unknown")
            elif isinstance(result, list) and result:
                memory_id = result[0].get("id", "unknown")

            return [
                TextContent(
                    type="text",
                    text=f"✅ Memory stored successfully (ID: {memory_id})",
                )
            ]

        elif name == "search_memory":
            scope = _scope_from_arguments(arguments)
            limit = arguments.get("limit", 10)
            memories = search_with_overfetch(
                manager=client,
                query=arguments["query"],
                user_id=arguments["user_id"],
                requested_limit=limit,
            )
            memories = filter_memories_by_scope(
                scope=scope,
                memories=memories or [],
                limit=limit,
            )

            if not memories:
                return [TextContent(type="text", text="No memories found matching your query.")]

            text = f"Found {len(memories)} memories:\n\n"
            for i, mem in enumerate(memories, 1):
                content = mem.get("memory", mem.get("content", "N/A"))
                text += f"{i}. {content}\n"

            return [TextContent(type="text", text=text)]

        elif name == "get_all_memories":
            scope = _scope_from_arguments(arguments)
            result = client.get_all(user_id=arguments["user_id"])
            memories = result.get("results", []) if isinstance(result, dict) else result
            memories = filter_memories_by_scope(scope=scope, memories=memories or [])

            if not memories:
                return [TextContent(type="text", text="No memories stored for this user.")]

            text = f"Total memories: {len(memories)}\n\n"
            for i, mem in enumerate(memories, 1):
                content = mem.get("memory", mem.get("content", "N/A"))
                text += f"{i}. {content}\n"

            return [TextContent(type="text", text=text)]

        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting Pixelated Memory MCP Server")

    # Initialize memory client
    get_mem0_client()

    async with stdio_server() as (read_stream, write_stream):
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
