# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "mcp>=1.26.0",
#   "fastmcp>=2.3.3",
#   "mem0ai>=1.0.3",
#   "google-genai>=1.62.0",
#   "pydantic>=2.11.7",
# ]
# ///
"""
FastMCP Server for Pixelated Memory.

Exposes memory capabilities as standard MCP tools, resources, and prompts.
Refined for autonomous agent use with consolidated, high-utility tools.
"""

import json
import inspect
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

# Add 'ai' to path if running directly to find siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ai.memory.manager_factory import get_memory_manager

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")
_manager_instance = None

# Initialize FastMCP
mcp = FastMCP(
    "Pixelated Memory",
    dependencies=["mem0ai", "google-genai", "pydantic", "openai"],
)


def get_manager():
    """Retrieve the global memory manager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = get_memory_manager()

    manager = _manager_instance
    if not manager:
        raise RuntimeError("No memory manager configured (NVIDIA or Gemini)")
    return manager


def _get_recent_memories(manager: Any, user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Prefer bounded retrieval when the manager supports it."""
    try:
        return manager.get_all_memories(user_id, limit=limit)
    except TypeError:
        memories = manager.get_all_memories(user_id)
        return memories[-limit:]


async def _generate_analysis_text(manager: Any, prompt: str, mode: str) -> str:
    """Generate analysis text across sync and async memory manager variants."""
    response: Any

    if hasattr(manager, "generate_analysis"):
        response = manager.generate_analysis(prompt, mode=mode)
    elif hasattr(manager, "generate_content"):
        response = manager.generate_content(prompt)
    elif hasattr(manager, "generate"):
        response = manager.generate(prompt=prompt)
    else:
        client = getattr(manager, "client", None)
        config = getattr(manager, "config", None)
        model_name = getattr(config, "model_name", None)

        if client and model_name and hasattr(client, "chat"):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        if client and model_name and hasattr(client, "models"):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text

        raise RuntimeError("Analysis requires an AI-capable memory manager.")

    if inspect.isawaitable(response):
        response = await response

    return response


# --- Resources ---


@mcp.resource("memory://{user_id}/context")
def get_memory_context(user_id: str) -> str:
    """
    Get a concise context summary for the user.
    Useful for quick session restarts or context bootstrapping.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No context data found for user: {user_id}"

    # Filter for high-value categories
    context_data = [
        m.get("memory") or m.get("content", "")
        for m in memories
        if m.get("metadata", {}).get("category")
        in ["fact", "preference", "goal", "identity", "project_context"]
    ] or [m.get("memory") or m.get("content", "") for m in memories[:15]]

    return f"### 🧠 Memory Context: {user_id}\n\n" + "\n".join(
        f"- {item}" for item in context_data
    )


# --- Prompts ---


@mcp.prompt()
def session_start(user_id: str) -> List[Dict[str, Any]]:
    """
    Prepare for a new session by recalling the user's state and goals.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an empathetic agent equipped with long-term memory. "
                f"Recall the core facts and goals for user '{user_id}' using the "
                "memory_query tool. Identify the 'Emotional Tide' or current "
                "project status before responding."
            ),
        }
    ]


# --- Tools ---


@mcp.tool()
async def memory_store(
    content: str, user_id: str, category: str = "fact", metadata: str = None
) -> str:
    """
    Store a significant fact, preference, or insight in long-term memory.

    AUTONOMOUS TRIGGER: Call this whenever you learn something new about the
    user or their project that should persist across sessions.
    Don't wait for permission.

    Args:
        content: The information to remember.
        user_id: The ID of the user/entity.
        category: 'fact', 'goal', 'preference', 'project_context', 'identity',
                  or 'insight'.
        metadata: Optional JSON string for extra attributes.
    """
    import contextlib

    manager = get_manager()
    meta_dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            meta_dict = json.loads(metadata) or {}

    if "timestamp" not in meta_dict:
        meta_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        res = manager.add_memory(
            content, user_id, metadata=meta_dict, category=category
        )
        return (
            f"✅ **Memory Secured** for {user_id}\n"
            f"- **Content:** {content}\n"
            f"- **Category:** {category}\n"
            f"- **ID:** {res}"
        )
    except Exception as e:
        return f"❌ Error storing memory: {str(e)}"


@mcp.tool()
async def memory_query(query: str, user_id: str, limit: int = 5) -> str:
    """
    Search long-term memory for relevant information.

    AUTONOMOUS TRIGGER: Call this if you're unsure about a past detail or need 
    to align with established user preferences/goals.

    Args:
        query: Search term or thematic query.
        user_id: The ID of the user to search for.
        limit: Max number of relevant results (default 5).
    """
    manager = get_manager()
    try:
        results = manager.search_memories(query, user_id)

        if not results:
            return f"🔍 No relevant matches for '{query}' in {user_id}'s memory."

        limited_results = results[:limit]

        formatted_results = [
            f"- [{r.get('score', 0.0):.2f}] "
            f"{r.get('memory') or r.get('content', 'N/A')}"
            for r in limited_results
        ]

        return (
            f"### 🔍 Memory Retrieval for {user_id}\n\n"
            + "\n".join(formatted_results)
        )
    except Exception as e:
        return f"❌ Error querying memory: {str(e)}"


@mcp.tool()
async def memory_update(
    memory_id: str, content: str, metadata: str = None
) -> str:
    """
    Refine or correct an existing memory entry.

    Args:
        memory_id: The unique ID from a previous store or query.
        content: The updated information.
        metadata: Optional JSON string for updated metadata.
    """
    import contextlib

    manager = get_manager()
    meta_dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            meta_dict = json.loads(metadata) or {}

    try:
        if manager.update_memory(memory_id, new_content=content, metadata=meta_dict):
            return f"🔄 **Memory Updated** (ID: {memory_id})"
        return "❌ Update failed or not supported."
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
async def memory_delete(memory_id: str) -> str:
    """
    Purge an obsolete or incorrect memory entry.
    """
    manager = get_manager()
    try:
        if manager.delete_memory(memory_id):
            return f"🗑️ **Memory Released** (ID: {memory_id})"
        return "❌ Deletion failed."
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
async def memory_sync_workspace(user_id: str, context_summary: str) -> str:
    """
    Sync the current workspace/project context into long-term memory.
    Useful for 'Serena-style' indexing of project evolution.

    AUTONOMOUS TRIGGER: Call this at the end of a significant task or session
    to 'checkpoint' the project narrative.
    """
    manager = get_manager()
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "workspace_sync"
    }
    try:
        res = manager.add_memory(
            context_summary, user_id, metadata=meta, category="project_context"
        )
        return f"🚀 **Workspace Context Synced** (ID: {res})"
    except Exception as e:
        return f"❌ Sync Error: {str(e)}"


@mcp.tool()
async def memory_analyze(user_id: str, mode: str = "themes") -> str:
    """
    Perform deep analysis on user memory for systemic insights.

    Modes:
    - 'themes': Extract recursive patterns and core beliefs.
    - 'evolution': Map narrative shifts and growth edges.
    - 'dissonance': Detect contradictions between stored facts.
    - 'forensics': Detailed extraction of entities and artifacts.

    Args:
        user_id: ID of the user to analyze.
        mode: Analysis mode ('themes', 'evolution', 'dissonance', 'forensics').
    """
    manager = get_manager()
    memories = _get_recent_memories(manager, user_id, limit=100)
    if not memories:
        return f"No memories for **{user_id}** to analyze."

    mem_list = memories[-100:]
    history = "\n".join(
        f"[{m.get('metadata', {}).get('timestamp', 'unk')}] "
        f"{m.get('memory') or m.get('content', '')}"
        for m in mem_list
    )

    prompts = {
        "themes": (
            "Identify recursive patterns, core beliefs, and persistent emotional "
            f"signals in these memories:\n\n{history}"
        ),
        "evolution": (
            "Map the narrative evolution and growth shifts in this user's journey:"
            f"\n\n{history}"
        ),
        "dissonance": (
            "Find contradictions or narrative conflicts in these memory entries:"
            f"\n\n{json.dumps(mem_list, indent=2)}"
        ),
        "forensics": (
            "Extract structured artifacts (People, Places, Tech, Goals) from this "
            f"memory set:\n\n{history}"
        ),
    }

    if mode not in prompts:
        return f"❌ Invalid mode: {mode}"

    try:
        response_text = await _generate_analysis_text(manager, prompts[mode], mode)
        return f"### 🧩 Memory Analysis ({mode}): {user_id}\n\n{response_text}"
    except Exception as e:
        return f"❌ Analysis Error: {str(e)}"


@mcp.tool()
async def memory_status(user_id: str) -> str:
    """
    Get high-level statistics and health of the user's memory cartography.
    """
    manager = get_manager()
    memories = _get_recent_memories(manager, user_id, limit=250)
    
    if not memories:
        return f"### 📊 Memory Status: {user_id}\n\nCartography is empty."

    count = len(memories)
    categories = {}
    for m in memories:
        cat = m.get("metadata", {}).get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
    
    cat_lines = "\n".join(f"- **{k}:** {v}" for k, v in categories.items())
    
    return (
        f"### 📊 Memory Status: {user_id}\n\n"
        f"**Total Anchors:** {count}\n"
        f"**Health:** {'Stable' if count > 10 else 'Developing'}\n\n"
        f"**Category Breakdown:**\n{cat_lines}"
    )


if __name__ == "__main__":
    mcp.run()
