from __future__ import annotations

from typing import Any

from .null_memory_command_service import NullMemoryCommandService
from .null_memory_query_service import NullMemoryQueryService


class NullMemoryLegacyService:
    """Legacy dict-shaped memory API layered on top of command/query services."""

    def __init__(
        self,
        *,
        commands: NullMemoryCommandService,
        queries: NullMemoryQueryService,
    ) -> None:
        self.commands = commands
        self.queries = queries

    def add(self, content: str, user_id: str, metadata: dict | None = None, **_kwargs: Any) -> dict:
        record = self.commands.add(content=content, user_id=user_id, metadata=metadata)
        return {"results": [{"id": record["id"]}]}

    def search(self, query: str, user_id: str, **kwargs: Any) -> dict:
        return {
            "results": self.queries.search_memories(
                query=query,
                user_id=user_id,
                limit=kwargs.get("limit", 1000),
            )
        }

    def get_all(self, user_id: str, **kwargs: Any) -> dict:
        return {
            "results": self.queries.get_all_memories(
                user_id=user_id,
                limit=kwargs.get("limit", 1000),
            )
        }

    def get(self, memory_id: str, **kwargs: Any) -> dict | None:
        user_id = kwargs.get("user_id")
        if not user_id:
            return None
        return self.commands.get(memory_id=memory_id, user_id=user_id)

    def update(self, memory_id: str, new_content: str, **kwargs: Any) -> bool:
        user_id = kwargs.get("user_id", "")
        if not user_id:
            return False
        return self.commands.update(
            memory_id=memory_id,
            user_id=user_id,
            new_content=new_content,
            metadata=kwargs.get("metadata"),
        )

    def delete(self, memory_id: str, **kwargs: Any) -> bool:
        user_id = kwargs.get("user_id", "")
        if not user_id:
            return False
        return self.commands.delete(memory_id=memory_id, user_id=user_id)

    def delete_all(self, user_id: str, **_kwargs: Any) -> bool:
        return self.commands.clear(user_id=user_id)
