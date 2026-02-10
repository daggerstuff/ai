"""
Null Memory Implementation.

Provides a fallback/stub implementation of the memory manager interface
for development environments or when external services are unavailable.
"""

from typing import Any, Dict, List, Optional


class NullMemoryManager:
    """
    Null implementation of Memory Manager.

    Implements the interface required by MemoryServer and GeminiMem0Manager
    but performs no actual persistence.
    """

    def __init__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        """Simulate adding memory (raw client interface)."""
        return {"results": [{"id": f"null-{hash(str(args)) % 10000}"}]}

    def search(self, *args, **kwargs):
        """Simulate searching (raw client interface)."""
        return {"results": []}

    def get_all(self, *args, **kwargs):
        """Simulate getting all (raw client interface)."""
        return {"results": []}

    def get(self, *args, **kwargs):
        """Simulate getting one (raw client interface)."""
        return None

    def update(self, *args, **kwargs):
        """Simulate update (raw client interface)."""
        pass

    def delete(self, *args, **kwargs):
        """Simulate delete (raw client interface)."""
        pass

    def delete_all(self, *args, **kwargs):
        """Simulate delete all (raw client interface)."""
        pass

    # --- High Level Interface matches GeminiMem0Manager ---

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> str:
        """Simulate adding memory."""
        return f"null-{hash(content) % 10000}"

    def search_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Simulate searching memories."""
        return []

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Simulate getting all memories."""
        return []

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Simulate getting a memory."""
        return None

    def update_memory(
        self, memory_id: str, new_content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Simulate updating memory."""
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Simulate deleting memory."""
        return True

    def clear_memory(self, user_id: str):
        """Simulate clearing memory."""
        pass

    @property
    def project(self):
        """Mock project property."""

        class NullProject:
            def update(self, **kwargs):
                pass

        return NullProject()
