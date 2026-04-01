from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .fastmcp_shared import (
    authorized_tool_context_from_json,
    get_recent_memories,
)


def _get_scoped_recent_memories(*, manager, scope, user_id: str, limit: int) -> list[dict[str, Any]]:
    if hasattr(manager, "get_all_memories_scoped"):
        return manager.get_all_memories_scoped(
            user_id=user_id,
            org_id=scope.org_id,
            project_id=scope.project_id,
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            include_shared=scope.include_shared,
            limit=limit,
        )
    memories = get_recent_memories(manager, user_id, limit=limit)
    from ai.api.mcp_server.memory_scope import filter_memories_by_scope

    return filter_memories_by_scope(scope=scope, memories=memories, limit=limit)


async def memory_status(
    user_id: str,
    auth_context: str,
    scope_context: str | None = None,
) -> str:
    """Get high-level statistics for the user's stored memories."""
    context = authorized_tool_context_from_json(
        tool_name="memory_status",
        user_id=user_id,
        auth_context=auth_context,
        scope_context=scope_context,
        payload={
            "user_id": user_id,
            "scope_context": scope_context,
        },
    )
    memories = _get_scoped_recent_memories(
        manager=context.manager,
        scope=context.scope,
        user_id=user_id,
        limit=250,
    )

    if not memories:
        return f"### Memory Status: {user_id}\n\nCartography is empty."

    categories: Dict[str, int] = {}
    for memory in memories:
        category = memory.get("metadata", {}).get("category", "general")
        categories[category] = categories.get(category, 0) + 1

    category_lines = "\n".join(f"- **{key}:** {value}" for key, value in categories.items())
    health = "Stable" if len(memories) > 10 else "Developing"
    return (
        f"### Memory Status: {user_id}\n\n"
        f"**Total Anchors:** {len(memories)}\n"
        f"**Health:** {health}\n\n"
        f"**Category Breakdown:**\n{category_lines}"
    )


def register_context_surfaces(mcp: FastMCP) -> None:
    mcp.tool()(memory_status)
