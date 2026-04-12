"""
Null Memory Implementation.

Provides a fallback in-memory implementation of the memory manager interface
for development environments or when external services are unavailable.

This is a REAL implementation using in-memory dictionaries, not a stub.
"""

from datetime import datetime, timezone

import threading
from typing import Any

from .base import BaseMemoryManager


class NullMemoryManager(BaseMemoryManager):
    """
    In-memory implementation of Memory Manager.

    Implements the interface required by MemoryServer and HindsightMemoryManager
    with actual in-memory storage using dictionaries.
    """

    def __init__(self, *args, **kwargs):
        # In-memory storage: user_id -> list of memory dicts
        self._memories: dict[str, list[dict[str, Any]]] = {}
        self._memory_counter = 0
        self._lock = threading.Lock()
        self.MAX_MEMORIES_PER_USER = 1000  # Prevent OOM

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        self._memory_counter += 1
        return f"mem-{self._memory_counter}"

    def add(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Add memory (raw client interface)."""
        memory_id = self._generate_id()
        memory = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            if user_id not in self._memories:
                self._memories[user_id] = []

            # Enforce capacity limit
            if len(self._memories[user_id]) >= self.MAX_MEMORIES_PER_USER:
                # Remove oldest (first index)
                self._memories[user_id].pop(0)

            self._memories[user_id].append(memory)

        return {"results": [{"id": memory_id}]}

    def search(self, query: str, user_id: str, **kwargs):
        """Search memories (raw client interface)."""
        with self._lock:
            if user_id not in self._memories:
                return {"results": []}

            # Simple substring search
            query_lower = query.lower()
            results = [
                m
                for m in self._memories[user_id]
                if query_lower in m["content"].lower()
            ]
            return {"results": results}

    def get_all(self, user_id: str, **kwargs):
        """Get all memories for user (raw client interface)."""
        with self._lock:
            return {"results": list(self._memories.get(user_id, []))}

    def get(self, memory_id: str, **kwargs):
        """Get specific memory by ID (raw client interface)."""
        with self._lock:
            for memories in self._memories.values():
                for memory in memories:
                    if memory["id"] == memory_id:
                        return memory
        return None

    def update(self, memory_id: str, new_content: str, **kwargs):
        """Update memory (raw client interface)."""
        with self._lock:
            for memories in self._memories.values():
                for memory in memories:
                    if memory["id"] == memory_id:
                        memory["content"] = new_content
                        memory["updated_at"] = datetime.now(timezone.utc).isoformat()
                        if kwargs.get("metadata") is not None:
                            memory["metadata"].update(kwargs["metadata"])
                        return True
        return False

    def delete(self, memory_id: str, **kwargs):
        """Delete memory (raw client interface)."""
        with self._lock:
            for user_id, memories in self._memories.items():
                for i, memory in enumerate(memories):
                    if memory["id"] == memory_id:
                        del self._memories[user_id][i]
                        return True
        return False

    def delete_all(self, user_id: str, **kwargs):
        """Delete all memories for user (raw client interface)."""
        with self._lock:
            if user_id in self._memories:
                del self._memories[user_id]
                return True
        return False

    # --- High Level Interface matches HindsightMemoryManager ---

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> str:
        """Add memory and return ID."""
        if category and metadata:
            metadata["category"] = category
        elif category:
            metadata = {"category": category}

        result = self.add(content, user_id, metadata)
        return result["results"][0]["id"]

    def search_memories(self, query: str, user_id: str) -> list[dict[str, Any]]:
        """Search memories and return list."""
        result = self.search(query, user_id)
        return result["results"]

    def get_all_memories(self, user_id: str) -> list[dict[str, Any]]:
        """Get all memories for user."""
        result = self.get_all(user_id)
        return result["results"]

    def get_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get specific memory by ID."""
        return self.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Update memory content."""
        return self.update(memory_id, new_content, metadata=metadata)

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        """Delete specific memory."""
        return self.delete(memory_id)

    def clear_memory(self, user_id: str) -> bool:
        """Clear all memories for user."""
        return self.delete_all(user_id)

    @property
    def project(self):
        """Mock project property for compatibility."""

        class NullProject:
            def update(self, **kwargs):
                pass

        return NullProject()
