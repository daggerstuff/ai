from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ScopedMemoryWriter(Protocol):
    def add_memory_scoped(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
        scope_metadata: Optional[dict] = None,
    ) -> str: ...

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> bool: ...

    def delete_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> bool: ...


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
