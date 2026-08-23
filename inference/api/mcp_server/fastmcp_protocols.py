from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryCreator(Protocol):
    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        category: str | None = None,
    ) -> str: ...


@runtime_checkable
class MemoryUpdater(Protocol):
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: dict | None = None,
        user_id: str | None = None,
    ) -> bool: ...


@runtime_checkable
class MemoryRemover(Protocol):
    def delete_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> bool: ...


@runtime_checkable
class MemoryReader(Protocol):
    def get_memory(self, memory_id: str) -> dict[str, Any] | None: ...


@runtime_checkable
class ScopedMemoryCreator(MemoryCreator, Protocol):
    def add_memory_scoped(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        category: str | None = None,
        scope_metadata: dict | None = None,
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
class BasicMemorySearcher(Protocol):
    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]] | dict[str, Any]: ...


@runtime_checkable
class LegacyMemorySearcher(Protocol):
    def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]] | dict[str, Any]: ...


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
class BasicMemoryLister(Protocol):
    def get_all_memories(
        self,
        user_id: str,
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
class MemoryQueryServiceProvider(Protocol):
    queries: ScopedMemoryCategoryCounter


@runtime_checkable
class MemoryScopeProvider(Protocol):
    @property
    def org_id(self) -> str | None: ...

    @property
    def project_id(self) -> str | None: ...

    @property
    def agent_id(self) -> str | None: ...

    @property
    def run_id(self) -> str | None: ...

    @property
    def session_id(self) -> str | None: ...

    @property
    def include_shared(self) -> bool: ...

    @property
    def visibility(self) -> str: ...

    def to_metadata(self) -> dict[str, Any] | None: ...
