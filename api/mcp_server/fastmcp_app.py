"""
FastMCP Server for Pixelated Memory.

Exposes memory capabilities as standard MCP tools.
Can be run directly via `uv run` to serve over stdio (default) or SSE.

Usage:
    uv run ai/api/mcp_server/fastmcp_app.py
"""

import logging
import os
import sys

# Add 'ai' to path if running directly to find siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mcp.server.fastmcp import FastMCP

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP
mcp = FastMCP("Pixelated Memory")

# --- Manager Initialization Patterns ---


def get_best_manager():
    """
    Initialize the best available memory manager based on env vars.
    Replicates logic from memory_server.py
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    mem0_key = os.environ.get("MEM0_API_KEY")

    try:
        # 1. Try GeminiMem0Manager (Preferred - Smart)
        if gemini_key:
            from ai.memory.mem0_gemini.manager import (
                GeminiMem0Config,
                GeminiMem0Manager,
            )

            logger.info("Initializing GeminiMem0Manager")
            return GeminiMem0Manager(
                GeminiMem0Config(
                    gemini_api_key=gemini_key,
                    mem0_api_key=mem0_key,
                    user_id="mcp_stdio_user",
                )
            )

        # 2. Try Standard MemoryManager (Mem0 Wrapper)
        if mem0_key:
            from ai.api.memory.memory_manager import get_memory_manager

            logger.info("Initializing Standard MemoryManager")
            return (
                get_memory_manager()
            )  # Will handle mem0 init internaly if env var set

        # 3. Fallback
        from ai.api.memory.null_memory import NullMemoryManager

        logger.warning("No API keys found. Using NullMemory.")
        return NullMemoryManager()

    except Exception as e:
        logger.error(f"Error initializing manager: {e}")
        from ai.api.memory.null_memory import NullMemoryManager

        return NullMemoryManager()


# Global manager instance (lazy loaded)
_manager = None


def get_manager():
    global _manager
    if not _manager:
        _manager = get_best_manager()
    return _manager


# --- Tools ---


@mcp.tool()
async def add_memory(
    content: str, user_id: str, metadata: str = None, category: str = None
) -> str:
    """
    Add information to long-term memory.

    Args:
        content: The text to remember.
        user_id: The ID of the user.
        metadata: Optional JSON string of metadata.
        category: Optional category (e.g. 'preference', 'fact').
    """
    import contextlib
    import json

    manager = get_manager()

    meta_dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            meta_dict = json.loads(metadata)

    # Handle different manager signatures
    try:
        if hasattr(manager, "add_memory"):  # GeminiMem0Manager
            res = manager.add_memory(
                content, user_id, metadata=meta_dict, category=category
            )
            return f"Memory stored. ID: {res}"

        elif hasattr(manager, "add_message"):  # MemoryManager
            if category:
                meta_dict["category"] = category
            manager.add_message(user_id, "default", content, "user", metadata=meta_dict)
            return "Memory stored."

        return "Error: Incompatible memory manager."
    except Exception as e:
        return f"Error adding memory: {str(e)}"


@mcp.tool()
async def search_memory(query: str, user_id: str) -> str:
    """
    Search for memories relevant to a query.
    """
    manager = get_manager()
    try:
        results = []
        if hasattr(manager, "search_memories"):
            results = manager.search_memories(query, user_id)
        elif hasattr(manager, "client") and hasattr(manager.client, "search"):
            # MemoryManager wrapping Mem0
            results = manager.client.search(query, user_id=user_id)

        return str(results)
    except Exception as e:
        return f"Error searching: {str(e)}"


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """Delete a memory by ID."""
    manager = get_manager()
    try:
        if hasattr(manager, "delete_memory"):
            manager.delete_memory(memory_id)
            return "Deleted."
        elif hasattr(manager, "client") and hasattr(manager.client, "delete"):
            manager.client.delete(memory_id)
            return "Deleted."
        return "Delete not supported."
    except Exception as e:
        return f"Error deleting: {str(e)}"


if __name__ == "__main__":
    mcp.run()
