from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from ai.api.mcp_server.memory_scope import build_scope_metadata

from .fastmcp_parsing import parse_metadata
from .fastmcp_protocols import MemoryCreator, MemoryScopeProvider, ScopedMemoryCreator


@dataclass(frozen=True)
class MemoryScopeConfig:
    org_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    include_shared: bool = True
    visibility: str = "private"

    def to_metadata(self) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {"visibility": self.visibility}
        if self.org_id:
            metadata["org_id"] = self.org_id
        if self.project_id:
            metadata["project_id"] = self.project_id
        if self.agent_id:
            metadata["agent_id"] = self.agent_id
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.session_id:
            metadata["session_id"] = self.session_id
        return metadata


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


@dataclass(frozen=True)
class MemoryStorePayload:
    content: str
    user_id: str
    category: str
    metadata_dict: dict


@dataclass(frozen=True)
class PreparedMemoryStorePayload:
    content: str
    user_id: str
    category: str
    metadata_for_scoped_creator: dict
    metadata_for_basic_creator: dict
    scope_metadata: dict | None


@dataclass(frozen=True)
class AuthorizedMemoryStoreOperation:
    prepared_payload: PreparedMemoryStorePayload
    creator: "MemoryStoreCreator"


class MemoryStoreCreator(Protocol):
    def create_memory(self, payload: PreparedMemoryStorePayload): ...


class MemoryStoreMetadataFactory:
    @staticmethod
    def prepare(
        *,
        payload: MemoryStorePayload,
        scope: MemoryScopeProvider,
    ) -> PreparedMemoryStorePayload:
        metadata = dict(payload.metadata_dict)
        if payload.category:
            metadata.setdefault("category", payload.category)
        scoped_metadata = dict(metadata)
        basic_metadata = build_scope_metadata(
            scope=scope,
            incoming_metadata=metadata,
            category=None,
        )
        return PreparedMemoryStorePayload(
            content=payload.content,
            user_id=payload.user_id,
            category=payload.category,
            metadata_for_scoped_creator=scoped_metadata,
            metadata_for_basic_creator=basic_metadata,
            scope_metadata=_scope_metadata_dict(scope),
        )


@dataclass(frozen=True)
class ScopeEnrichedMemoryCreator:
    manager: Any

    def create_memory(self, payload: PreparedMemoryStorePayload):
        manager = self.manager
        if isinstance(manager, ScopedMemoryCreator):
            return manager.add_memory_scoped(
                content=payload.content,
                user_id=payload.user_id,
                metadata=payload.metadata_for_scoped_creator,
                category=payload.category,
                scope_metadata=payload.scope_metadata,
            )

        if not isinstance(manager, MemoryCreator):
            raise TypeError("Memory manager does not support write operations.")

        return manager.add_memory(
            content=payload.content,
            user_id=payload.user_id,
            metadata=payload.metadata_for_basic_creator,
            category=payload.category,
        )


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
    payload = MemoryStorePayload(
        content=request.content,
        user_id=request.user_id,
        category=request.category,
        metadata_dict=request.metadata_dict,
    )
    return {
        "content": payload.content,
        "user_id": payload.user_id,
        "category": payload.category,
        "metadata": payload.metadata_dict,
    }


def build_memory_store_payload(request: MemoryStoreRequest) -> MemoryStorePayload:
    return MemoryStorePayload(
        content=request.content,
        user_id=request.user_id,
        category=request.category,
        metadata_dict=request.metadata_dict,
    )
def _scope_metadata_dict(scope: MemoryScopeProvider) -> dict | None:
    metadata = scope.to_metadata()
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise TypeError("Scope metadata must be a dictionary.")
    return metadata


class MemoryStorePersistenceService:
    @staticmethod
    def persist(*, operation: AuthorizedMemoryStoreOperation):
        return operation.creator.create_memory(operation.prepared_payload)


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
