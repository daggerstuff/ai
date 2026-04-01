from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai.api.mcp_server.memory_scope import build_scope_metadata

from .fastmcp_parsing import parse_metadata
from .fastmcp_protocols import ScopedMemoryWriter
from .fastmcp_shared import authorized_tool_context


@dataclass(frozen=True)
class MemoryScopeConfig:
    org_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    include_shared: bool = True
    visibility: str = "private"


@dataclass(frozen=True)
class MemoryAuthConfig:
    actor_id: str
    timestamp: str
    nonce: str
    signature: str


@dataclass(frozen=True)
class MemoryStoreRequest:
    content: str
    user_id: str
    category: str
    metadata_dict: dict
    actor_id: str
    timestamp: str
    nonce: str
    signature: str
    scope_config: MemoryScopeConfig


class MemoryStoreRequestFactory:
    @staticmethod
    def from_inputs(
        *,
        content: str,
        user_id: str,
        category: str,
        metadata: Optional[str],
        auth: dict,
        scope: dict,
    ) -> MemoryStoreRequest:
        return MemoryStoreRequest(
            content=content,
            user_id=user_id,
            category=category,
            metadata_dict=parse_metadata(metadata),
            actor_id=auth["actor_id"],
            timestamp=auth["timestamp"],
            nonce=auth["nonce"],
            signature=auth["signature"],
            scope_config=MemoryScopeConfig(
                org_id=scope["org_id"],
                project_id=scope["project_id"],
                agent_id=scope["agent_id"],
                run_id=scope["run_id"],
                session_id=scope["session_id"],
                include_shared=scope["include_shared"],
                visibility=scope["visibility"],
            ),
        )


def memory_store_payload(request: MemoryStoreRequest) -> dict:
    return {
        "content": request.content,
        "user_id": request.user_id,
        "category": request.category,
        "metadata": request.metadata_dict,
    }


def authorize_memory_store_request(request: MemoryStoreRequest):
    return authorized_tool_context(
        tool_name="memory_store",
        actor_id=request.actor_id,
        user_id=request.user_id,
        timestamp=request.timestamp,
        nonce=request.nonce,
        signature=request.signature,
        payload=memory_store_payload(request),
        org_id=request.scope_config.org_id,
        project_id=request.scope_config.project_id,
        agent_id=request.scope_config.agent_id,
        run_id=request.scope_config.run_id,
        session_id=request.scope_config.session_id,
        include_shared=request.scope_config.include_shared,
        visibility=request.scope_config.visibility or "private",
    )


class MemoryStorePersistenceService:
    @staticmethod
    def persist(*, request: MemoryStoreRequest, manager, scope):
        if isinstance(manager, ScopedMemoryWriter):
            return manager.add_memory_scoped(
                content=request.content,
                user_id=request.user_id,
                metadata=request.metadata_dict,
                category=request.category,
                scope_metadata=scope.to_metadata(),
            )

        enriched_metadata = build_scope_metadata(
            scope=scope,
            incoming_metadata=request.metadata_dict,
            category=request.category,
        )
        return manager.add_memory(
            request.content,
            request.user_id,
            metadata=enriched_metadata,
            category=request.category,
        )


def memory_store_result_id(result) -> str | None:
    if isinstance(result, str):
        normalized = result.strip()
        return normalized or None
    if isinstance(result, dict):
        if isinstance(result.get("id"), str) and result["id"].strip():
            return result["id"].strip()
        results = result.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                record_id = first.get("id")
                if isinstance(record_id, str) and record_id.strip():
                    return record_id.strip()
    return None


def memory_store_success_message(
    *,
    user_id: str,
    content: str,
    category: str,
    result,
) -> str:
    lines = [
        f"✅ **Memory Secured** for {user_id}",
        f"- **Content:** {content}",
        f"- **Category:** {category}",
    ]
    record_id = memory_store_result_id(result)
    if record_id:
        lines.append(f"- **ID:** {record_id}")
    return "\n".join(lines)
