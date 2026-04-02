from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from ai.api.mcp_server.memory_scope import memory_in_scope

from .fastmcp_parsing import parse_auth_context, parse_metadata, parse_scope_context
from .fastmcp_presenters import memory_store_success_message
from .fastmcp_protocols import MemoryRemover, MemoryUpdater
from .fastmcp_search import search_scoped_memories
from .fastmcp_shared import authorized_tool_context_from_json, authorized_tool_context_from_parts
from .fastmcp_store import (
    ScopeEnrichedMemoryCreator,
    build_memory_store_plan,
    persist_memory_store_plan,
    scope_config_from_parsed,
)


def _search_scoped_memories(
    *,
    manager,
    query: str,
    user_id: str,
    scope,
    limit: int,
):
    return search_scoped_memories(
        manager=manager,
        query=query,
        user_id=user_id,
        scope=scope,
        limit=limit,
    )


async def memory_store(
    content: str,
    user_id: str,
    auth_context: str,
    category: str = "fact",
    metadata: Optional[str] = None,
    scope_context: Optional[str] = None,
) -> str:
    """Store a significant fact, preference, or insight in long-term memory."""
    auth = parse_auth_context(auth_context)
    metadata_dict = parse_metadata(metadata)
    context = authorized_tool_context_from_parts(
        tool_name="memory_store",
        user_id=user_id,
        auth=auth,
        scope=parse_scope_context(scope_context),
        payload={
            "content": content,
            "user_id": user_id,
            "category": category,
            "metadata": metadata_dict,
        },
        visibility_default="private",
    )
    authorized_user_id = context.scope.user_id
    scope_config = scope_config_from_parsed(parse_scope_context(scope_context))

    try:
        plan = build_memory_store_plan(
            content=content,
            user_id=authorized_user_id,
            category=category,
            metadata_dict=metadata_dict,
            scope=scope_config,
        )
        result = persist_memory_store_plan(
            creator=ScopeEnrichedMemoryCreator(manager=context.manager),
            plan=plan,
        )
        return memory_store_success_message(
            user_id=plan.user_id,
            content=plan.content,
            category=plan.category,
            result=result,
        )
    except Exception as exc:
        return f"❌ Error storing memory: {exc}"


async def memory_query(
    query: str,
    user_id: str,
    auth_context: str,
    limit: int = 5,
    scope_context: Optional[str] = None,
) -> str:
    """Search long-term memory for relevant information."""
    context = authorized_tool_context_from_json(
        tool_name="memory_query",
        user_id=user_id,
        auth_context=auth_context,
        scope_context=scope_context,
        payload={
            "query": query,
            "user_id": user_id,
            "limit": limit,
            "scope_context": scope_context,
        },
    )
    try:
        results = _search_scoped_memories(
            manager=context.manager,
            query=query,
            user_id=context.scope.user_id,
            scope=context.scope,
            limit=limit,
        )
        if not results:
            return (
                f"🔍 No relevant matches for '{query}' within the requested memory scope "
                f"for {user_id}."
            )
        formatted = [
            f"- [{item.get('score', 0.0):.2f}] {item.get('memory') or item.get('content', 'N/A')}"
            for item in results[:limit]
        ]
        return f"### Memory Retrieval for {user_id}\n\n" + "\n".join(formatted)
    except Exception as exc:
        return f"❌ Error querying memory: {exc}"


async def memory_update(
    memory_id: str,
    content: str,
    user_id: str,
    auth_context: str,
    metadata: Optional[str] = None,
    scope_context: Optional[str] = None,
) -> str:
    """Refine or correct an existing memory entry."""
    metadata_dict = parse_metadata(metadata)
    context = authorized_tool_context_from_json(
        tool_name="memory_update",
        user_id=user_id,
        auth_context=auth_context,
        scope_context=scope_context,
        payload={
            "memory_id": memory_id,
            "content": content,
            "user_id": user_id,
            "metadata": metadata_dict,
            "scope_context": scope_context,
        },
    )

    try:
        authorized_user_id = context.scope.user_id
        if not isinstance(context.manager, MemoryUpdater):
            return "❌ Update failed: memory backend is not writable."
        if not memory_in_scope(manager=context.manager, scope=context.scope, memory_id=memory_id):
            return "❌ Update denied: memory not found in provided scope."
        if context.manager.update_memory(
            memory_id,
            new_content=content,
            metadata=metadata_dict,
            user_id=authorized_user_id,
        ):
            return f"🔄 **Memory Updated** (ID: {memory_id})"
        return "❌ Update failed or not supported."
    except Exception as exc:
        return f"❌ Error: {exc}"


async def memory_delete(
    memory_id: str,
    user_id: str,
    auth_context: str,
    scope_context: Optional[str] = None,
) -> str:
    """Purge an obsolete or incorrect memory entry."""
    context = authorized_tool_context_from_json(
        tool_name="memory_delete",
        user_id=user_id,
        auth_context=auth_context,
        scope_context=scope_context,
        payload={
            "memory_id": memory_id,
            "user_id": user_id,
            "scope_context": scope_context,
        },
    )
    try:
        authorized_user_id = context.scope.user_id
        if not isinstance(context.manager, MemoryRemover):
            return "❌ Deletion failed: memory backend is not writable."
        if not memory_in_scope(manager=context.manager, scope=context.scope, memory_id=memory_id):
            return "❌ Delete denied: memory not found in provided scope."
        if context.manager.delete_memory(memory_id, user_id=authorized_user_id):
            return f"🗑️ **Memory Released** (ID: {memory_id})"
        return "❌ Deletion failed."
    except Exception as exc:
        return f"❌ Error: {exc}"


def register_memory_tools(mcp: FastMCP) -> None:
    mcp.tool()(memory_store)
    mcp.tool()(memory_query)
    mcp.tool()(memory_update)
    mcp.tool()(memory_delete)
