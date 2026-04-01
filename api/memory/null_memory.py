"""
In-memory fallback memory backend for local tests and degraded service paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.api.memory.base import BaseMemoryManager
from .null_memory_protocol_adapter import NullMemoryProtocolAdapter
from .null_memory_store import NullMemoryStore


class NullMemoryManager(BaseMemoryManager):
    """High-level memory manager facade backed by an in-memory store."""

    def __init__(self, *args, **kwargs) -> None:
        self.store = NullMemoryStore()
        self.protocol = NullMemoryProtocolAdapter(self.store)

    def add(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, str]]]:
        record = self.store.add_record(content=content, user_id=user_id, metadata=metadata)
        return {"results": [{"id": record["id"]}]}

    def search(self, query: str, user_id: str, **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        return {"results": self.store.search_records(query=query, user_id=user_id)}

    def get_all(self, user_id: str, **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        return {"results": self.store.list_records(user_id=user_id)}

    def get(self, memory_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return self.store.get_record(memory_id=memory_id)

    def update(self, memory_id: str, new_content: str, **kwargs: Any) -> bool:
        return self.store.update_record(
            memory_id=memory_id,
            new_content=new_content,
            metadata=kwargs.get("metadata"),
        )

    def delete(self, memory_id: str, **kwargs: Any) -> bool:
        return self.store.delete_record(memory_id=memory_id)

    def delete_all(self, user_id: str, **kwargs: Any) -> bool:
        return self.store.clear_user(user_id=user_id)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> str:
        return self.protocol.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.protocol.search_memories(query=query, user_id=user_id, limit=limit)

    def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.protocol.get_all_memories(user_id=user_id, limit=limit)

    def get_all_memories_scoped(
        self,
        *,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.protocol.get_all_memories_scoped(
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
        )

    def get_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        memory = self.get(memory_id)
        if memory is None:
            return None
        if user_id is not None and memory.get("user_id") != user_id:
            return None
        return memory

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        if user_id is not None:
            memory = self.get_memory(memory_id, user_id=user_id)
            if memory is None:
                return False
        return self.update(memory_id, new_content, metadata=metadata)

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        if user_id is not None:
            memory = self.get_memory(memory_id, user_id=user_id)
            if memory is None:
                return False
        return self.delete(memory_id)

    def clear_memory(self, user_id: str) -> bool:
        return self.delete_all(user_id)

    def get_health_status(self) -> Dict[str, Any]:
        return {"status": "degraded", "provider": self.__class__.__name__}

    @property
    def project(self):
        class NullProject:
            def update(self, **kwargs: Any) -> None:
                return None

        return NullProject()
