from __future__ import annotations

from fastmcp import FastMCP

from ai.inference.api.memory.memory_status_service import build_memory_status_summary

from .fastmcp_shared import (
    _stdio_trust_enabled,
    authorized_tool_context_from_json,
    stdio_trusted_tool_context,
)


async def memory_status(
    user_id: str,
    auth_context: str | None = None,
    scope_context: str | None = None,
) -> str:
    """Get high-level statistics for the user's stored memories."""
    if not auth_context and _stdio_trust_enabled():
        context = stdio_trusted_tool_context(
            user_id=user_id,
            scope_context=scope_context,
        )
    else:
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
    authorized_user_id = context.scope.user_id
    summary = build_memory_status_summary(
        manager=context.manager,
        scope=context.scope,
        user_id=authorized_user_id,
    )
    if summary.total_memories == 0:
        return f"### Memory Status: {authorized_user_id}\n\nCartography is empty."
    total_label = "Sampled Memories" if summary.is_sampled else "Total Memories"
    category_lines = "\n".join(f"- **{key}:** {value}" for key, value in summary.categories.items())
    return (
        f"### Memory Status: {authorized_user_id}\n\n"
        f"**{total_label}:** {summary.total_memories}\n"
        f"**Health:** {summary.health}\n\n"
        f"**Category Breakdown:**\n{category_lines}"
    )


def register_context_surfaces(mcp: FastMCP) -> None:
    mcp.tool()(memory_status)
