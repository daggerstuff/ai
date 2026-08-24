from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from ai.inference.api.mcp_server.memory_auth import (
    authorize_memory_access,
    configured_actor_policies,
    configured_actor_tokens,
)
from ai.inference.api.mcp_server.memory_scope import MemoryScope, scope_from_kwargs
from ai.research.manager_factory import get_required_memory_manager

from .fastmcp_parsing import (
    ParsedAuthContext,
    ParsedScopeContext,
    parse_auth_context,
    parse_scope_context,
)

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


def get_recent_memories(manager: Any, user_id: str, limit: int) -> list[dict[str, Any]]:
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
    payload: dict[str, Any],
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
    payload: dict[str, Any],
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
    payload: dict[str, Any],
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


def _stdio_trust_enabled() -> bool:
    """Return True when the server allows unauthenticated MCP stdio calls."""
    flag = os.environ.get("HINDSIGHT_MCP_STDIO_TRUST", "").strip().lower()
    return flag in ("1", "true", "yes")


def _stdio_trusted_user() -> str:
    """Resolve the implicit user identity for stdio-trusted calls."""
    for key in (
        "HINDSIGHT_COMPAT_DEFAULT_USER_ID",
        "SUBCONSCIOUS_USER_ID",
        "USER",
        "USERNAME",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise HTTPException(
        status_code=503,
        detail=(
            "HINDSIGHT_MCP_STDIO_TRUST is enabled but no user identity is configured. "
            "Set HINDSIGHT_COMPAT_DEFAULT_USER_ID."
        ),
    )


def _stdio_trusted_actor() -> str:
    """Resolve the implicit actor identity for stdio-trusted calls."""
    configured = os.environ.get("HINDSIGHT_COMPAT_BEARER_ACTOR_ID", "").strip()
    if configured:
        return configured
    tokens = configured_actor_tokens()
    if len(tokens) == 1:
        return next(iter(tokens))
    raise HTTPException(
        status_code=503,
        detail=(
            "HINDSIGHT_MCP_STDIO_TRUST is enabled with multiple actors configured. "
            "Set HINDSIGHT_COMPAT_BEARER_ACTOR_ID to select one."
        ),
    )


def stdio_trusted_tool_context(
    *,
    user_id: str | None,
    scope_context: str | None = None,
    visibility_default: str = "private",
) -> AuthorizedToolContext:
    """Create an AuthorizedToolContext without HMAC auth for stdio-trusted callers.

    Only callable when HINDSIGHT_MCP_STDIO_TRUST=true.  The actor identity and
    user identity are resolved from environment variables so that the transport
    boundary (stdio) is the trust boundary.
    """
    actor_id = _stdio_trusted_actor()
    resolved_user = (user_id or "").strip() or _stdio_trusted_user()
    policies = configured_actor_policies()
    actor_key = actor_id
    if actor_key not in policies:
        actor_key = actor_id.lower().replace("-", "_")
    policy = policies.get(actor_key)
    if policy is not None and not policy.allows_user(resolved_user):
        raise HTTPException(
            status_code=403,
            detail=(f"Stdio-trusted actor '{actor_id}' is not permitted to act for user '{resolved_user}'."),
        )
    logger.info(
        "stdio-trusted MCP call accepted: actor=%s user=%s",
        actor_id,
        resolved_user,
    )
    scope = parse_scope_context(scope_context)
    return AuthorizedToolContext(
        manager=get_manager(),
        scope=scope_from_kwargs(
            user_id=resolved_user,
            org_id=scope.org_id,
            project_id=scope.project_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            session_id=scope.session_id,
            include_shared=scope.include_shared,
            visibility=scope.visibility or visibility_default,
        ),
    )


def authorized_tool_context_from_json(
    *,
    tool_name: str,
    user_id: str,
    auth_context: str | None,
    scope_context: str | None,
    payload: dict[str, Any],
    visibility_default: str = "private",
) -> AuthorizedToolContext:
    if not auth_context:
        if _stdio_trust_enabled():
            return stdio_trusted_tool_context(
                user_id=user_id,
                scope_context=scope_context,
                visibility_default=visibility_default,
            )
        raise HTTPException(
            status_code=401,
            detail=("auth_context is required when HINDSIGHT_MCP_STDIO_TRUST is not enabled."),
        )
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
