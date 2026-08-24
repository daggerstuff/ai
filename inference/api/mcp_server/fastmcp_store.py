from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai.inference.api.mcp_server.memory_scope import build_scope_metadata, scope_metadata_dict

from .fastmcp_parsing import ParsedScopeContext
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
        """Satisfy MemoryScopeProvider protocol."""
        return scope_metadata_dict(self)


@dataclass(frozen=True)
class MemoryStorePlan:
    content: str
    user_id: str
    category: str
    basic_metadata: dict
    scoped_metadata: dict
    scope_metadata: dict | None


class MemoryStoreCreator(Protocol):
    def create_memory(self, plan: MemoryStorePlan) -> str: ...


class _BasicMemoryWriter:
    def __init__(self, manager: MemoryCreator) -> None:
        self.manager = manager

    def create_memory(self, plan: MemoryStorePlan) -> str:
        return self.manager.add_memory(
            content=plan.content,
            user_id=plan.user_id,
            metadata=plan.basic_metadata,
            category=plan.category,
        )


class _ScopedMemoryWriter:
    def __init__(self, manager: ScopedMemoryCreator) -> None:
        self.manager = manager

    def create_memory(self, plan: MemoryStorePlan) -> str:
        return self.manager.add_memory_scoped(
            content=plan.content,
            user_id=plan.user_id,
            metadata=plan.scoped_metadata,
            category=plan.category,
            scope_metadata=plan.scope_metadata,
        )


@dataclass(frozen=True)
class ScopeEnrichedMemoryCreator:
    manager: Any

    def create_memory(self, plan: MemoryStorePlan) -> str:
        writer = self._resolve_writer()
        return writer.create_memory(plan)

    def _resolve_writer(self) -> MemoryStoreCreator:
        """Prefer scoped writers so scope metadata is never silently dropped."""
        manager = self.manager
        if isinstance(manager, ScopedMemoryCreator):
            return _ScopedMemoryWriter(manager)
        if isinstance(manager, MemoryCreator):
            return _BasicMemoryWriter(manager)
        raise TypeError("Memory manager does not support write operations.")


def scope_config_from_parsed(scope: ParsedScopeContext) -> MemoryScopeConfig:
    return MemoryScopeConfig(
        org_id=scope.org_id,
        project_id=scope.project_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        session_id=scope.session_id,
        include_shared=scope.include_shared,
        visibility=scope.visibility,
    )


def build_memory_store_plan(
    *,
    content: str,
    user_id: str,
    category: str,
    metadata_dict: dict,
    scope: MemoryScopeProvider,
) -> MemoryStorePlan:
    basic_metadata = _metadata_with_category(
        category=category,
        metadata_dict=metadata_dict,
    )
    return MemoryStorePlan(
        content=content,
        user_id=user_id,
        category=category,
        basic_metadata=basic_metadata,
        scoped_metadata=build_scope_metadata(
            scope=scope,
            incoming_metadata=basic_metadata,
            category=category,
        ),
        scope_metadata=_scope_metadata(scope),
    )


def _metadata_with_category(*, category: str, metadata_dict: dict) -> dict:
    metadata = dict(metadata_dict)
    if category:
        metadata.setdefault("category", category)
    return metadata


def _scope_metadata(scope: MemoryScopeProvider) -> dict | None:
    metadata = scope_metadata_dict(scope)
    if not isinstance(metadata, dict):
        raise TypeError("Scope metadata must be a dictionary.")
    return metadata


def persist_memory_store_plan(*, creator: MemoryStoreCreator, plan: MemoryStorePlan) -> str:
    return creator.create_memory(plan)


def memory_store_result_id(result) -> str | None:
    if isinstance(result, str):
        normalized = result.strip()
        return normalized or None
    if isinstance(result, dict):
        results = result.get("results")
        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if isinstance(first, dict):
                record_id = first.get("id")
                if isinstance(record_id, str) and record_id.strip():
                    return record_id.strip()
        if isinstance(result.get("id"), str) and result["id"].strip():
            return result["id"].strip()
    return None
