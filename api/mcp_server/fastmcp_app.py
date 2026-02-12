"""
FastMCP Server for Pixelated Memory.

Exposes memory capabilities as standard MCP tools, resources, and prompts.
Can be run directly via `uv run` to serve over stdio (default) or SSE.

Usage:
    uv run ai/api/mcp_server/fastmcp_app.py
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

# Add 'ai' to path if running directly to find siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mcp.server.fastmcp import FastMCP

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP
mcp = FastMCP(
    "Pixelated Memory",
    dependencies=["mem0ai", "google-genai", "pydantic", "e2b"],
)

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
            return get_memory_manager()

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


# --- Resources ---


@mcp.resource("memory://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """
    Get a summary of the user's profile and key preferences.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No profile data found for user: {user_id}"

    # Filter for profile-like categories if they exist, otherwise use all
    profile_data = [
        m.get("memory") or m.get("content", "")
        for m in memories
        if m.get("metadata", {}).get("category") in ["preference", "fact", "bio"]
    ] or [m.get("memory") or m.get("content", "") for m in memories[:10]]

    return f"### 👤 User Profile: {user_id}\n\n" + "\n".join(
        f"- {item}" for item in profile_data
    )


@mcp.resource("memory://{user_id}/history")
def get_user_history(user_id: str) -> str:
    """
    Get the recent conversation history and emotional context.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No history found for user: {user_id}"

    # Sort by timestamp if available
    sorted_memories = sorted(
        memories,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
        reverse=True,
    )

    history = [
        m.get("memory") or m.get("content", "")
        for m in sorted_memories[:20]  # Last 20 interactions
    ]

    return f"### 📜 Recent History: {user_id}\n\n" + "\n".join(
        f"- {item}" for item in history
    )


# --- Prompts ---


@mcp.prompt()
def empathy_onboarding(name: str = "Friend") -> List[Dict[str, Any]]:
    """
    A premium onboarding prompt for Pixelated Empathy.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are Pixelated Empathy, a compassionate AI companion trained in "
                "therapeutic dialogue. Your purpose is to ensure the user feels "
                "genuinely heard. Use the EARS framework: Empathize (name emotions), "
                "Acknowledge (validate without minimizing), Reflect (briefly "
                "summarize), and Support (offer collaborative next steps).\n\n"
                "Maintain presence: 'I'm here with you right now.'\n"
                "Prioritize connection before direction. Avoid toxic positivity."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Hello! My name is {name}. I'm here to explore the Empathy Gym. "
                "Can you introduce yourself and tell me how our journey begins?"
            ),
        },
    ]


@mcp.prompt()
def session_reflection(user_id: str) -> List[Dict[str, Any]]:
    """
    Reflect on the current session's emotional growth and breakthroughs.
    """
    return [
        {
            "role": "system",
            "content": (
                "Analyze the recent interaction history for this user. "
                "Provide a compassionate reflection focused on emotional growth "
                "and shared breakthroughs today. Use the user's specific themes "
                "(mirroring their words) to validate their journey. Offer presence "
                "and maintain a non-judgmental, autonomous tone."
            ),
        },
        {
            "role": "user",
            "content": (
                "I'd like to reflect on our session today. "
                f"Can you look at my journey (User ID: {user_id}) and share "
                "what we've discovered together?"
            ),
        },
    ]


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

    manager = get_manager()

    meta_dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            meta_dict = json.loads(metadata) or {}

    # Handle different manager signatures
    try:
        if hasattr(manager, "add_memory"):  # GeminiMem0Manager
            res = manager.add_memory(
                content, user_id, metadata=meta_dict, category=category
            )
            return (
                f"### ✨ Memory Crystallized\n\n"
                f"I've updated my internal map for **{user_id}**.\n\n"
                f"**Insight:** {content}\n"
                f"**Category:** {category or 'General'}\n\n"
                f"*Memory ID: {res}*"
            )

        elif hasattr(manager, "add_message"):  # MemoryManager
            if category:
                meta_dict["category"] = category
            manager.add_message(user_id, "default", content, "user", metadata=meta_dict)
            return f"### ✅ Memory Saved\n\nInsight for **{user_id}** has been stored."

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

        if not results:
            return (
                f"### 🔍 Search: {query}\n\n"
                f"No matching memories found for **{user_id}**."
            )

        formatted_results = []
        for r in results:
            content = r.get("memory") or r.get("content", "Unknown")
            score = r.get("score", 0.0)
            formatted_results.append(f"- **[{score:.2f}]** {content}")

        return (
            f"### 🔍 Relevant Insights for {user_id}\n\n"
            + "\n".join(formatted_results)
            + f"\n\n*Based on query: '{query}'*"
        )
    except Exception as e:
        return f"Error searching: {str(e)}"


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """Delete a memory by ID."""
    manager = get_manager()
    try:
        success = False
        if hasattr(manager, "delete_memory"):
            success = manager.delete_memory(memory_id)
        elif hasattr(manager, "client") and hasattr(manager.client, "delete"):
            manager.client.delete(memory_id)
            success = True

        if success:
            return "### 🗑️ Memory Released\n\nThe insight has been removed from context."
        return "Delete not supported or failed."
    except Exception as e:
        return f"Error deleting: {str(e)}"


@mcp.tool()
async def get_empathy_score(user_id: str) -> str:
    """
    Calculate the current empathy alignment score for a user.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)

    if not memories:
        return "### 📊 Empathy Status\n\nNo data yet. Let's start a conversation!"

    # Mock logic for score calculation based on memory count and variety
    score = min(1.0, len(memories) / 20.0)
    status = "Attuned" if score > 0.7 else "Developing" if score > 0.3 else "Learning"

    return (
        f"### 💜 Empathy Alignment: {score:.2%}\n\n"
        f"Status: **{status}**\n\n"
        f"We've shared {len(memories)} significant insights so far."
    )


@mcp.tool()
async def execute_in_sandbox(
    code: str,
    language: str = "python",
    timeout_seconds: int = 30,
) -> str:
    """
    Execute code in a secure, ephemeral E2B sandbox.

    Runs code in an isolated Linux VM that is destroyed after
    execution. No secrets or sensitive data are forwarded.

    Args:
        code: The code to execute.
        language: 'python' or 'shell'. Defaults to 'python'.
        timeout_seconds: Max execution time (1-120s). Defaults to 30.
    """
    from ai.api.mcp_server.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor()

    try:
        result = executor.execute_code(
            language=language,
            code=code,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        return f"### ❌ Validation Error\n\n{exc}"

    data = result.to_dict()

    if data.get("error"):
        return f"### ⚠️ Sandbox Error\n\n{data['error']}"

    parts = [f"### 🖥️ Sandbox Execution ({language})"]

    if data["stdout"]:
        parts.append(f"\n**stdout:**\n```\n{data['stdout']}\n```")
    if data["stderr"]:
        parts.append(f"\n**stderr:**\n```\n{data['stderr']}\n```")

    parts.append(
        f"\n**Exit code:** {data['exit_code']} | **Duration:** {data['duration_ms']}ms"
    )

    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run()
