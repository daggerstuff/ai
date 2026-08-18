from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


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
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str | None:
        """Add a memory and return its ID, or None if blocked by gating."""

    @abstractmethod
    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories."""

    @abstractmethod
    def get_all_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve all memories for a user."""

    @abstractmethod
    def get_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID."""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Any | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Update an existing memory."""

    @abstractmethod
    def delete_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
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
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int = 100,
        offset: int = 0,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return memories visible to the supplied user and scope."""
        ...


@runtime_checkable
class CategoryScopedMemoryManager(ScopedMemoryManager, Protocol):
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
    ) -> dict[str, int]:
        """Return scoped memory counts grouped by category."""
        ...


@runtime_checkable
class ForesightCompatibleMemoryManager(Protocol):
    def retain_items(self, bank_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Store Foresight-compatible memory items."""
        ...

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        """Recall Foresight-compatible memory items."""
        ...

    def recall_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        """Recall Foresight-compatible memory items constrained to one user."""
        ...

    def list_documents(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Foresight-compatible documents for one user."""
        ...

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a Foresight-compatible document by ID."""
        ...

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
    ) -> bool:
        """Delete a Foresight-compatible document by ID."""
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
        items: list[dict[str, Any]],
        user_id: str,
        base_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Prepare scoped Foresight items for retention."""
        ...


@runtime_checkable
class HealthReportingMemoryManager(Protocol):
    def get_health_status(self) -> dict[str, Any]:
        """Return a health payload for the memory manager."""
        ...
