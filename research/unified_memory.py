"""
Compatibility layer for the previous unified memory interface.

The project has moved toward local reflection-based memory primitives, but a number
of tests and downstream imports still target `ai.research.unified_memory`.
This module re-exports the shared types and exposes a minimal `MemoryProvider`
protocol so legacy imports remain stable during phased refactors.
"""

from __future__ import annotations

from typing import Protocol

from .reflection_types import CrisisSeverity, Memory, MemoryCategory, MemoryMetadata

__all__ = [
    "CrisisSeverity",
    "Memory",
    "MemoryCategory",
    "MemoryMetadata",
    "MemoryProvider",
]


class MemoryProvider(Protocol):
    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str: ...

    async def get_memory(self, memory_id: str) -> Memory | None: ...

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = ...,
        metadata: MemoryMetadata | None = ...,
    ) -> None: ...

    async def delete_memory(self, memory_id: str) -> None: ...

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[Memory]: ...

    async def get_memories_by_user(self, user_id: str, limit: int = 100) -> list[Memory]: ...

    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: str | None = ...,
        limit: int = 100,
    ) -> list[Memory]: ...
