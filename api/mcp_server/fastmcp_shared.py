from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from ai.api.mcp_server.memory_auth import authorize_memory_access
from ai.api.mcp_server.memory_scope import MemoryScope, scope_from_kwargs
from .fastmcp_parsing import (
    ParsedAuthContext,
    ParsedScopeContext,
    parse_auth_context,
    parse_scope_context,
)
from ai.memory.manager_factory import get_required_memory_manager

logger = logging.getLogger("mcp_server")
_manager_instance = None


@dataclass(frozen=True)
class AuthorizedToolContext:
    manager: Any
    scope: MemoryScope


def get_manager() -> Any:
    """Retrieve the global memory manager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = get_required_memory_manager()

    return _manager_instance


def get_recent_memories(
    manager: Any, user_id: str, limit: int
) -> List[Dict[str, Any]]:
    """Retrieve a bounded slice of recent memories."""
    return manager.get_all_memories(user_id, limit=limit)


def authorize_tool_access(
    *,
    tool_name: str,
    actor_id: str,
    user_id: str,
    timestamp: str,
    nonce: str,
    signature: str,
    payload: Dict[str, Any],
):
    """Authorize MCP tool invocations with the shared actor-policy model."""
    request_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return authorize_memory_access(
        actor_id=actor_id,
        user_id=user_id,
        request_method="MCP",
        request_target=f"mcp://pixelated-memory/{tool_name}",
        request_body=request_body,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
    )


def authorized_tool_context(
    *,
    tool_name: str,
    actor_id: str,
    user_id: str,
    timestamp: str,
    nonce: str,
    signature: str,
    payload: Dict[str, Any],
    org_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    include_shared: bool = True,
    visibility: str = "private",
) -> AuthorizedToolContext:
    authorize_tool_access(
        tool_name=tool_name,
        actor_id=actor_id,
        user_id=user_id,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
        payload=payload,
    )
    return AuthorizedToolContext(
        manager=get_manager(),
        scope=scope_from_kwargs(
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            include_shared=include_shared,
            visibility=visibility,
        ),
    )


def authorized_tool_context_from_parts(
    *,
    tool_name: str,
    user_id: str,
    auth: ParsedAuthContext,
    scope: ParsedScopeContext,
    payload: Dict[str, Any],
    visibility_default: str = "private",
) -> AuthorizedToolContext:
    return authorized_tool_context(
        tool_name=tool_name,
        actor_id=auth.actor_id,
        user_id=user_id,
        timestamp=auth.timestamp,
        nonce=auth.nonce,
        signature=auth.signature,
        payload=payload,
        org_id=scope.org_id,
        project_id=scope.project_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        session_id=scope.session_id,
        include_shared=scope.include_shared,
        visibility=scope.visibility or visibility_default,
    )


def authorized_tool_context_from_json(
    *,
    tool_name: str,
    user_id: str,
    auth_context: str,
    scope_context: str | None,
    payload: Dict[str, Any],
    visibility_default: str = "private",
) -> AuthorizedToolContext:
    auth = parse_auth_context(auth_context)
    scope = parse_scope_context(scope_context)
    return authorized_tool_context_from_parts(
        tool_name=tool_name,
        user_id=user_id,
        auth=auth,
        scope=scope,
        payload=payload,
        visibility_default=visibility_default,
    )
