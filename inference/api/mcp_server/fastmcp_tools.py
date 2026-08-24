from __future__ import annotations

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ai.inference.api.mcp_server.memory_scope import memory_in_scope

from .fastmcp_parsing import parse_metadata, parse_scope_context
from .fastmcp_presenters import memory_store_success_message
from .fastmcp_protocols import MemoryRemover, MemoryUpdater
from .fastmcp_search import search_scoped_memories
from .fastmcp_shared import authorized_tool_context_from_json
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


class MemoryStoreRequest(BaseModel):
    content: str = Field(description="The significant fact, preference, or insight to store.")
    user_id: str = Field(description="Unique identifier for the user.")
    auth_context: str | None = Field(default=None, description="HMAC authentication context.")
    category: str = Field(default="fact", description="Category for the memory (e.g., 'fact', 'preference').")
    metadata: str | None = Field(default=None, description="Additional metadata context as a JSON string.")
    scope_context: str | None = Field(default=None, description="Detailed scope context as a JSON string.")


class MemoryUpdateRequest(BaseModel):
    memory_id: str = Field(description="The unique identifier of the memory entry to update.")
    content: str = Field(description="The updated text content for the memory entry.")
    user_id: str = Field(description="Unique identifier for the user.")
    auth_context: str | None = Field(default=None, description="HMAC authentication context.")
    metadata: str | None = Field(default=None, description="Updated metadata context as a JSON string.")
    scope_context: str | None = Field(default=None, description="Updated scope context as a JSON string.")


async def memory_store(req: MemoryStoreRequest) -> str:
    """Store a significant fact, preference, or insight in long-term memory."""
    metadata_dict = parse_metadata(req.metadata)
    context = authorized_tool_context_from_json(
        tool_name="memory_store",
        user_id=req.user_id,
        auth_context=req.auth_context,
        scope_context=req.scope_context,
        payload={
            "content": req.content,
            "user_id": req.user_id,
            "category": req.category,
            "metadata": metadata_dict,
        },
        visibility_default="private",
    )
    authorized_user_id = context.scope.user_id
    scope_config = scope_config_from_parsed(parse_scope_context(req.scope_context))

    try:
        plan = build_memory_store_plan(
            content=req.content,
            user_id=authorized_user_id,
            category=req.category,
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
    auth_context: str | None = None,
    limit: int = 5,
    scope_context: str | None = None,
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
            return f"🔍 No relevant matches for '{query}' within the requested memory scope for {user_id}."
        formatted = [
            f"- [{item.get('score', 0.0):.2f}] {item.get('memory') or item.get('content') or item.get('text', 'N/A')}"
            for item in results[:limit]
        ]
        return f"### Memory Retrieval for {user_id}\n\n" + "\n".join(formatted)
    except Exception as exc:
        return f"❌ Error querying memory: {exc}"


async def memory_update(req: MemoryUpdateRequest) -> str:
    """Refine or correct an existing memory entry."""
    metadata_dict = parse_metadata(req.metadata)
    context = authorized_tool_context_from_json(
        tool_name="memory_update",
        user_id=req.user_id,
        auth_context=req.auth_context,
        scope_context=req.scope_context,
        payload={
            "memory_id": req.memory_id,
            "content": req.content,
            "user_id": req.user_id,
            "metadata": metadata_dict,
            "scope_context": req.scope_context,
        },
    )

    try:
        authorized_user_id = context.scope.user_id
        if not isinstance(context.manager, MemoryUpdater):
            return "❌ Update failed: memory backend is not writable."
        if not memory_in_scope(
            manager=context.manager,
            scope=context.scope,
            memory_id=req.memory_id,
        ):
            return "❌ Update denied: memory not found in provided scope."
        if context.manager.update_memory(
            req.memory_id,
            new_content=req.content,
            metadata=metadata_dict,
            user_id=authorized_user_id,
        ):
            return f"🔄 **Memory Updated** (ID: {req.memory_id})"
        return "❌ Update failed or not supported."
    except Exception as exc:
        return f"❌ Error: {exc}"


async def memory_delete(
    memory_id: str,
    user_id: str,
    auth_context: str | None = None,
    scope_context: str | None = None,
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
        if not memory_in_scope(
            manager=context.manager,
            scope=context.scope,
            memory_id=memory_id,
        ):
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
