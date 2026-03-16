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
from datetime import datetime, timezone
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
    dependencies=["mem0ai", "google-genai", "pydantic"],
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
    Follows Memory Forensics principles by extracting core identity artifacts.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No profile data found for user: {user_id}"

    # Filter for profile-like categories if they exist, otherwise use all
    profile_data = [
        m.get("memory") or m.get("content", "")
        for m in memories
        if m.get("metadata", {}).get("category")
        in ["preference", "fact", "bio", "identity"]
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

    # Auto-inject timestamp for temporal features
    if "timestamp" not in meta_dict:
        meta_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

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
async def update_memory(memory_id: str, content: str, metadata: str = None) -> str:
    """
    Update an existing memory with new content.

    Args:
        memory_id: The ID of the memory to update.
        content: The new text content.
        metadata: Optional JSON string of updated metadata.
    """
    import contextlib

    manager = get_manager()
    meta_dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            meta_dict = json.loads(metadata) or {}

    try:
        if hasattr(manager, "update_memory") and (
            manager.update_memory(memory_id, content, metadata=meta_dict)
        ):
            return (
                f"### 🔄 Memory Refined\n\n"
                f"I've updated the insight (ID: {memory_id}).\n\n"
                f"**New Content:** {content}"
            )
        return "Update not supported by current manager or failed."
    except Exception as e:
        return f"Error updating memory: {str(e)}"


@mcp.tool()
async def get_memory(memory_id: str) -> str:
    """
    Retrieve a specific memory by its ID.
    """
    manager = get_manager()
    try:
        if mem := manager.get_memory(memory_id):
            return f"### 🧠 Memory Retrieval\n\n{json.dumps(mem, indent=2)}"
        return f"Memory with ID {memory_id} not found."
    except Exception as e:
        return f"Error retrieving: {str(e)}"


@mcp.tool()
async def extract_memory_artifacts(memory_id: str) -> str:
    """
    Perform deep forensic analysis on a specific memory to extract
    structured artifacts (entities, relationships, emotional tone).

    This implements 'Memory Forensics' by identifying hidden context
    within raw memory strings.
    """
    manager = get_manager()
    if not hasattr(manager, "get_memory") or not hasattr(manager, "client"):
        return "Artifact extraction not supported."

    if not (memory := manager.get_memory(memory_id)):
        return f"Memory {memory_id} not found."

    content = memory.get("memory") or memory.get("content", "")
    prompt = (
        "Extract structured artifacts from the following memory content. "
        "Identify: Entities (People, Places, Things), Relationships (A is a B), "
        "Emotional Tone (Primary emotion), and Scientific Validity "
        "(Is it a fact or feeling?).\n\n"
        f"CONTENT: {content}\n\n"
        "Return the result as a Markdown list of artifacts."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return (
            f"### 🔍 Memory Artifacts: {memory_id}\n\n"
            f"**Raw:** {content}\n\n"
            f"{response.text}"
        )
    except Exception as e:
        return f"Error extracting artifacts: {str(e)}"


@mcp.tool()
async def list_memory_history(user_id: str, limit: int = 50) -> str:
    """
    Perform a forensic chronological listing of all memories for a user.
    Useful for auditing narrative evolution and identifying state changes.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)

    if not memories:
        return f"No memories found for user: {user_id}"

    # Sort chronologically
    sorted_mems = sorted(
        memories,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
    )

    lines = []
    for m in sorted_mems[:limit]:
        ts = m.get("metadata", {}).get("timestamp", "unknown")
        cid = m.get("id", "unk")
        content = m.get("memory") or m.get("content", "")
        cat = m.get("metadata", {}).get("category", "general")
        lines.append(f"| {ts} | {cid} | {cat} | {content} |")

    return (
        f"### 📊 Memory Audit Log: {user_id}\n\n"
        "| Timestamp | ID | Category | Content |\n"
        "| :--- | :--- | :--- | :--- |\n" + "\n".join(lines)
    )


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
        if hasattr(manager, "delete_memory"):
            res = manager.delete_memory(memory_id)
            if res:
                return (
                    "### 🗑️ Memory Released\n\n"
                    "The insight has been removed from context."
                )
            return "Delete failed."
        elif hasattr(manager, "client") and hasattr(manager.client, "delete"):
            manager.client.delete(memory_id)
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
async def detect_cognitive_conflicts(user_id: str) -> str:
    """
    EXPERIMENTAL: Analyze all memories for a user to detect contradictions
    or emergent conflicts in their narrative.

    This uses Gemini to perform a deep scan of the 'Emotional Cartography'
    for a given user ID.
    """
    manager = get_manager()
    if not hasattr(manager, "get_all_memories") or not hasattr(manager, "client"):
        return "Conflict detection not supported by current manager."

    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No memories found for **{user_id}** to analyze."

    # Prepare memories for analysis
    mem_list = [
        {"id": m.get("id", "unk"), "content": m.get("memory") or m.get("content", "")}
        for m in memories
    ]

    prompt = (
        "You are an expert at Narrative Consistency and Cognitive Dissonance analysis. "
        "Examine the following list of memories (id and content) for a single user. "
        "Identify any contradictions, obsolete facts, or emergent conflicts. "
        "A conflict is when two memories can't both be true at the same time, "
        "or when a later memory suggests an earlier one has been superseded.\n\n"
        f"USER MEMORIES for {user_id}:\n"
        f"{json.dumps(mem_list, indent=2)}\n\n"
        "Return a structured report in Markdown. For each conflict, list the "
        "affected memory IDs, the nature of the contradiction, and a recommendation. "
        "If no conflicts are found, say 'No narrative conflicts detected.'"
    )

    try:
        # Use manager's Gemini client directly if available
        # manager.client is genai.Client
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return f"### 🧩 Cognitive Conflict Report: {user_id}\n\n{response.text}"
    except Exception as e:
        return f"Error during conflict detection: {str(e)}"


@mcp.tool()
async def analyze_narrative_evolution(user_id: str) -> str:
    """
    OUTSIDE THE BOX: Analyze the chronological evolution of the user's narrative.
    Looks for shifting perspectives, growth patterns, or recurring themes over time.

    This implements 'Emotional Cartography' by mapping the journey through memory.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)
    if not memories:
        return f"No memories found for **{user_id}** to analyze journey."

    # Sort memories chronologically
    sorted_memories = sorted(
        memories,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
    )

    history_str = ""
    for m in sorted_memories:
        ts = m.get("metadata", {}).get("timestamp", "unknown")
        content = m.get("memory") or m.get("content", "")
        history_str += f"[{ts}] {content}\n"

    prompt = (
        "You are a specialist in Narrative Therapy and Longitudinal Data Analysis. "
        "Analyze the following chronological sequence of memories for a user. "
        "Identify: 1) Major narrative shifts, 2) Recurring themes that persist, "
        "3) Signs of emotional growth or stagnation, 4) The evolution of their "
        "'Self' concept.\n\n"
        f"USER JOURNEY for {user_id}:\n{history_str}\n\n"
        "Return a deep reflective report in Markdown titled "
        "'Narrative Evolution of [User ID]'."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error analyzing journey: {str(e)}"


@mcp.tool()
async def weave_resonances(memory_id: str) -> str:
    """
    Find semantic and emotional echoes between a specific memory
    and all other memories for that user.

    Returns a "resonance map" showing which memories echo the target
    and why they are connected.

    Args:
        memory_id: The ID of the memory to find resonances for.
    """
    manager = get_manager()
    target = manager.get_memory(memory_id)
    if not target:
        return f"Memory `{memory_id}` not found."

    target_content = target.get("memory") or target.get("content", "")
    user_id = target.get("user_id", "user")
    all_memories = manager.get_all_memories(user_id)

    others = [
        {
            "id": m.get("id", "unk"),
            "content": m.get("memory") or m.get("content", ""),
        }
        for m in all_memories
        if m.get("id") != memory_id
    ]

    if not others:
        return "No other memories to compare against."

    prompt = (
        "You are a Narrative Resonance Analyst trained in therapeutic "
        "pattern recognition. Given a TARGET memory and a list of OTHER "
        "memories from the same user, identify semantic and emotional "
        "echoes — memories that resonate with the target.\n\n"
        "A resonance is when two memories share emotional themes, "
        "recurring language, connected life events, or complementary "
        "perspectives on the same core experience.\n\n"
        f"TARGET MEMORY (ID: {memory_id}):\n"
        f"{target_content}\n\n"
        f"OTHER MEMORIES:\n{json.dumps(others, indent=2)}\n\n"
        "Return a Markdown report titled 'Resonance Map'. For each "
        "connected memory, list its ID, the nature of the echo, and "
        "the emotional thread linking them. If few resonances exist, "
        "say so honestly."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return f"### 🕸️ Resonance Map: `{memory_id}`\n\n{response.text}"
    except Exception as e:
        return f"Error finding resonances: {str(e)}"


@mcp.tool()
async def get_emotional_tide(user_id: str, window: str = "30d") -> str:
    """
    Extract emotional valence from memories within a time window
    and render a temporal trajectory of emotional movement.

    Uses simplified Plutchik categories: Joy, Sadness, Anger, Fear,
    Trust, Surprise.

    Args:
        user_id: The ID of the user.
        window: Time window (e.g. '7d', '30d', '90d'). Defaults to '30d'.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)

    if not memories:
        return f"No memories for **{user_id}** to analyze."

    sorted_memories = sorted(
        memories,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
    )

    timeline = ""
    for m in sorted_memories:
        ts = m.get("metadata", {}).get("timestamp", "unknown")
        content = m.get("memory") or m.get("content", "")
        timeline += f"[{ts}] {content}\n"

    prompt = (
        "You are an Emotional Cartographer specializing in mapping "
        "emotional trajectories over time. Analyze the following "
        "chronological memories and produce an 'Emotional Tide' report.\n\n"
        "For each memory, identify the dominant emotion from the "
        "simplified Plutchik categories: Joy, Sadness, Anger, Fear, "
        "Trust, Surprise, Anticipation, Disgust.\n\n"
        f"MEMORIES (chronological, window: {window}):\n{timeline}\n\n"
        "Return a Markdown report with:\n"
        "1. A timeline showing the emotional category for each memory\n"
        "2. The overall emotional trajectory (rising, falling, cycling)\n"
        "3. Key emotional turning points\n"
        "4. A one-sentence 'Emotional Weather' summary\n\n"
        "Be compassionate and non-judgmental."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return f"### 🌊 Emotional Tide: {user_id}\n\n{response.text}"
    except Exception as e:
        return f"Error analyzing emotional tide: {str(e)}"


@mcp.tool()
async def reconstruct_self(user_id: str, before_date: str) -> str:
    """
    Reconstruct who the user was at a specific point in time.

    Filters memories to only those that existed before the given date,
    then synthesizes a narrative of the user's identity at that moment.
    A time machine for emotional growth.

    Args:
        user_id: The ID of the user.
        before_date: ISO date string (e.g. '2026-01-15').
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)

    if not memories:
        return f"No memories for **{user_id}** to reconstruct."

    filtered = [
        m for m in memories if m.get("metadata", {}).get("timestamp", "") <= before_date
    ]

    if not filtered:
        return (
            f"No memories found before **{before_date}** for {user_id}. "
            "Either no memories existed then, or they lack timestamps."
        )

    sorted_memories = sorted(
        filtered,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
    )

    history = ""
    for m in sorted_memories:
        ts = m.get("metadata", {}).get("timestamp", "unknown")
        content = m.get("memory") or m.get("content", "")
        history += f"[{ts}] {content}\n"

    prompt = (
        "You are a Temporal Identity Reconstructor. Given a set of "
        "memories that existed BEFORE a specific date, reconstruct "
        "who this person was at that point in time.\n\n"
        f"MEMORIES BEFORE {before_date} for {user_id}:\n{history}\n\n"
        "Produce a compassionate, reflective report titled "
        f"'Who Was {user_id} on {before_date}?' that synthesizes:\n"
        "1. Core beliefs and perspectives at that time\n"
        "2. Emotional state and dominant feelings\n"
        "3. Key relationships and concerns\n"
        "4. What they seemed to be working through\n\n"
        "Write in a warm, therapeutic tone. This is a mirror, "
        "not a judgment."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error reconstructing self: {str(e)}"


@mcp.tool()
async def distill_themes(user_id: str) -> str:
    """
    Synthesize all memories into core themes, recurring patterns,
    and emotional anchors.

    This is synthesis only — it never modifies or deletes originals.
    Like dreaming for AI memory: finding the signal across all memories.

    Args:
        user_id: The ID of the user.
    """
    manager = get_manager()
    memories = manager.get_all_memories(user_id)

    if not memories:
        return f"No memories for **{user_id}** to distill."

    mem_list = [
        {
            "id": m.get("id", "unk"),
            "content": m.get("memory") or m.get("content", ""),
            "category": m.get("metadata", {}).get("category", "general"),
            "timestamp": m.get("metadata", {}).get("timestamp", "unknown"),
        }
        for m in memories
    ]

    prompt = (
        "You are a Narrative Distillation Specialist. Analyze ALL of "
        "the following memories for a single user and produce a "
        "'Theme Distillation' report.\n\n"
        f"ALL MEMORIES for {user_id} ({len(mem_list)} total):\n"
        f"{json.dumps(mem_list, indent=2)}\n\n"
        "Produce a Markdown report with:\n"
        "1. **Core Themes** — The 3-5 dominant themes across all memories\n"
        "2. **Emotional Anchors** — Emotions that appear repeatedly\n"
        "3. **Recurring Language** — Phrases or words the user returns to\n"
        "4. **Unresolved Tensions** — Contradictions or open questions\n"
        "5. **Growth Edges** — Areas where change is emerging\n\n"
        "This is a therapist's session notes, written by memory itself. "
        "Be warm, insightful, and honest."
    )

    try:
        response = manager.client.models.generate_content(
            model=manager.config.model_name,
            contents=prompt,
        )
        return f"### 💎 Theme Distillation: {user_id}\n\n{response.text}"
    except Exception as e:
        return f"Error distilling themes: {str(e)}"


if __name__ == "__main__":
    mcp.run()
