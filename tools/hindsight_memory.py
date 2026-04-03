#!/usr/bin/env python3
"""
Hindsight Memory MCP Tools for Claude Code.

These tools provide direct access to the Hindsight memory system via MCP protocol.
They wrap the MCP server calls as native Python functions.
"""

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _get_server_params() -> StdioServerParameters:
    """Get the MCP server parameters for hindsight memory."""
    return StdioServerParameters(
        command="/home/vivi/pixelated/ai/.venv/bin/python3",
        args=["-m", "ai.api.mcp_server.fastmcp_app"],
        env={
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/home/vivi/pixelated",
            "MEMORY_PROVIDER": "local_hindsight",
            "HINDSIGHT_LOCAL_DB_PATH": "/home/vivi/pixelated/memory.db",
            "HINDSIGHT_MCP_STDIO_TRUST": "true",
            "HINDSIGHT_COMPAT_DEFAULT_USER_ID": "vivi",
            "HINDSIGHT_COMPAT_BEARER_ACTOR_ID": "local-hindsight-cli",
        }
    )


async def _call_mcp_tool(tool_name: str, params: dict) -> str:
    """Call an MCP tool and return the result."""
    server_params = _get_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, params)
            if result.content:
                return result.content[0].text
            return "No result"


def memory_store(content: str, user_id: str = "vivi", category: str = "fact") -> str:
    """Store a significant fact, preference, or insight in long-term memory.

    Args:
        content: The memory content to store
        user_id: The user ID (default: "vivi")
        category: Category for the memory (default: "fact")

    Returns:
        Success message with memory ID or error message
    """
    params = {
        "content": content,
        "user_id": user_id,
        "category": category,
    }
    return asyncio.run(_call_mcp_tool("memory_store", params))


def memory_query(query: str, user_id: str = "vivi", limit: int = 5) -> str:
    """Search long-term memory for relevant information.

    Args:
        query: Search query string
        user_id: The user ID (default: "vivi")
        limit: Maximum number of results (default: 5)

    Returns:
        Formatted search results or error message
    """
    params = {
        "query": query,
        "user_id": user_id,
        "limit": limit,
    }
    return asyncio.run(_call_mcp_tool("memory_query", params))


def memory_status(user_id: str = "vivi") -> str:
    """Get high-level statistics for the user's stored memories.

    Args:
        user_id: The user ID (default: "vivi")

    Returns:
        Memory statistics or error message
    """
    params = {"user_id": user_id}
    return asyncio.run(_call_mcp_tool("memory_status", params))


def memory_update(memory_id: str, content: str, user_id: str = "vivi") -> str:
    """Refine or correct an existing memory entry.

    Args:
        memory_id: The ID of the memory to update
        content: New content for the memory
        user_id: The user ID (default: "vivi")

    Returns:
        Success message or error message
    """
    params = {
        "memory_id": memory_id,
        "content": content,
        "user_id": user_id,
    }
    return asyncio.run(_call_mcp_tool("memory_update", params))


def memory_delete(memory_id: str, user_id: str = "vivi") -> str:
    """Purge an obsolete or incorrect memory entry.

    Args:
        memory_id: The ID of the memory to delete
        user_id: The user ID (default: "vivi")

    Returns:
        Success message or error message
    """
    params = {
        "memory_id": memory_id,
        "user_id": user_id,
    }
    return asyncio.run(_call_mcp_tool("memory_delete", params))


# CLI interface for direct testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: hindsight_memory.py <command> [args...]")
        print("Commands:")
        print("  store <content> [category]  - Store a memory")
        print("  query <query> [limit]       - Search memories")
        print("  status                      - Get memory status")
        print("  update <id> <content>       - Update a memory")
        print("  delete <id>                 - Delete a memory")
        sys.exit(1)

    command = sys.argv[1]

    if command == "store":
        if len(sys.argv) < 3:
            print("Error: content required")
            sys.exit(1)
        content = " ".join(sys.argv[2:])
        category = sys.argv[3] if len(sys.argv) > 3 else "fact"
        print(memory_store(content, category=category))

    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: query required")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        print(memory_query(query, limit=limit))

    elif command == "status":
        print(memory_status())

    elif command == "update":
        if len(sys.argv) < 4:
            print("Error: memory_id and content required")
            sys.exit(1)
        memory_id = sys.argv[2]
        content = " ".join(sys.argv[3:])
        print(memory_update(memory_id, content))

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: memory_id required")
            sys.exit(1)
        memory_id = sys.argv[2]
        print(memory_delete(memory_id))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
