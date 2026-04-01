from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryCreator(Protocol):
    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
    ) -> str: ...


@runtime_checkable
class MemoryUpdater(Protocol):
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> bool: ...


@runtime_checkable
class MemoryRemover(Protocol):
    def delete_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> bool: ...


@runtime_checkable
class ScopedMemoryCreator(MemoryCreator, Protocol):
    def add_memory_scoped(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
        scope_metadata: Optional[dict] = None,
    ) -> str: ...


@runtime_checkable
class ScopedMemorySearcher(Protocol):
    def search_memories_scoped(
        self,
        *,
        query: str,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class ScopedMemoryLister(Protocol):
    def get_all_memories_scoped(
        self,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class ScopedMemoryCategoryCounter(Protocol):
    def count_memories_by_category_scoped(
        self,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, int]: ...


@runtime_checkable
class MemoryScopeProvider(Protocol):
    org_id: str | None
    project_id: str | None
    agent_id: str | None
    run_id: str | None
    session_id: str | None
    include_shared: bool
    visibility: str

    def to_metadata(self) -> dict[str, Any] | None: ...
