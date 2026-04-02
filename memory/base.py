from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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
        metadata: Optional[Any] = None,
        category: Optional[str] = None,
    ) -> str:
        """Add a memory and return its ID."""

    @abstractmethod
    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories."""

    @abstractmethod
    def get_all_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user."""

    @abstractmethod
    def get_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Update an existing memory."""

    @abstractmethod
    def delete_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a memory by ID."""

    @abstractmethod
    def clear_memory(self, user_id: str) -> bool:
        """Clear all memories for a user."""

    def get_provider_name(self) -> str:
        return self.__class__.__name__

    def close(self) -> None:
        """Release any resources held by this manager.

        Subclasses may override to perform teardown (e.g. closing DB
        connections). The default is a no-op.
        """


@runtime_checkable
class ScopedMemoryManager(Protocol):
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
        offset: int = 0,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return memories visible to the supplied user and scope."""
        ...


@runtime_checkable
class CategoryScopedMemoryManager(ScopedMemoryManager, Protocol):
    def count_memories_by_category_scoped(
        self,
        *,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> Dict[str, int]:
        """Return scoped memory counts grouped by category."""
        ...


@runtime_checkable
class HindsightCompatibleMemoryManager(Protocol):
    def retain_items(
        self, bank_id: str, items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Store Hindsight-compatible memory items."""
        ...

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> Dict[str, Any]:
        """Recall Hindsight-compatible memory items."""
        ...

    def recall_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> Dict[str, Any]:
        """Recall Hindsight-compatible memory items constrained to one user."""
        ...

    def list_documents(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List Hindsight-compatible documents for one user."""
        ...

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a Hindsight-compatible document by ID."""
        ...

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a Hindsight-compatible document by ID."""
        ...

    def can_write_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        """Return whether the user may mutate the target document."""
        ...

    def prepare_retained_items(
        self,
        *,
        bank_id: str,
        items: List[Dict[str, Any]],
        user_id: str,
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare scoped Hindsight items for retention."""
        ...


@runtime_checkable
class HealthReportingMemoryManager(Protocol):
    def get_health_status(self) -> Dict[str, Any]:
        """Return a health payload for the memory manager."""
        ...

