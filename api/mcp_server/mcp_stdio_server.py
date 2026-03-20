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
                    "category": {
                        "type": "string",
                        "description": "Memory category (optional)",
                    },
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
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10,
                    },
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
            result = client.add(
                arguments["content"],
                user_id=arguments["user_id"],
                metadata=(
                    {"category": arguments.get("category")}
                    if arguments.get("category")
                    else None
                ),
            )
            memory_id = "unknown"
            if isinstance(result, dict) and "results" in result and result["results"]:
                memory_id = result["results"][0].get("id", "unknown")

            return [
                TextContent(
                    type="text",
                    text=f"✅ Memory stored successfully (ID: {memory_id})",
                )
            ]

        elif name == "search_memory":
            result = client.search(
                arguments["query"],
                user_id=arguments["user_id"],
                limit=arguments.get("limit", 10),
            )
            memories = result.get("results", []) if isinstance(result, dict) else result

            if not memories:
                return [
                    TextContent(
                        type="text", text="No memories found matching your query."
                    )
                ]

            text = f"Found {len(memories)} memories:\n\n"
            for i, mem in enumerate(memories, 1):
                content = mem.get("memory", mem.get("content", "N/A"))
                text += f"{i}. {content}\n"

            return [TextContent(type="text", text=text)]

        elif name == "get_all_memories":
            result = client.get_all(user_id=arguments["user_id"])
            memories = result.get("results", []) if isinstance(result, dict) else result

            if not memories:
                return [
                    TextContent(type="text", text="No memories stored for this user.")
                ]

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
