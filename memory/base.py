from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemoryManager(ABC):
    """
    Abstract Base Class for Memory Managers.
    Enforces a consistent interface across different providers.
    """

    @abstractmethod
    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> str:
        """Add a memory and return its ID."""
        pass

    @abstractmethod
    def search_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        pass

    @abstractmethod
    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user."""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        pass

    @abstractmethod
    def clear_memory(self, user_id: str) -> bool:
        """Clear all memories for a user."""
        pass
