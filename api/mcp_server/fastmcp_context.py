from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ai.api.memory.memory_status_summary import summarize_memory_status

from .fastmcp_shared import (
    authorized_tool_context_from_json,
)


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
    summary = summarize_memory_status(
        manager=context.manager,
        scope=context.scope,
        user_id=user_id,
    )
    if summary.total_anchors == 0:
        return f"### Memory Status: {user_id}\n\nCartography is empty."
    category_lines = "\n".join(
        f"- **{key}:** {value}" for key, value in summary.categories.items()
    )
    return (
        f"### Memory Status: {user_id}\n\n"
        f"**Total Anchors:** {summary.total_anchors}\n"
        f"**Health:** {summary.health}\n\n"
        f"**Category Breakdown:**\n{category_lines}"
    )


def register_context_surfaces(mcp: FastMCP) -> None:
    mcp.tool()(memory_status)
