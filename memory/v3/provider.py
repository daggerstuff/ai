"""
Memory provider interface and implementations.

Pluggable architecture - swap backends without changing code.
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import httpx

from ai.memory.hindsight_local_adapter import metadata_to_tags, serialize_context
from .local_hindsight_sqlite import SQLiteMemoryStore
from .provider_resilience import SQLiteResilienceController
from .shared_service_transport import SharedMemoryServiceError, SharedMemoryServiceTransport

__all__ = [
    "Memory",
    "MemoryProvider",
    "LocalHindsightProvider",
    "SharedMemoryServiceProvider",
    "MockProvider",
    "create_memory_provider",
    "close_memory_provider",
]

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    created_at: str
    metadata: dict


class MemoryProvider(ABC):
    """
    Abstract interface for memory storage.

    Implementations: LocalHindsightProvider, MockProvider
    """

    @abstractmethod
    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """
        Search for relevant memories.

        Args:
            query: Search query text
            user_id: User identifier
            limit: Maximum number of results

        Returns:
            List of matching memories, ordered by relevance
        """
        ...

    @abstractmethod
    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """
        Store a new memory.

        Args:
            content: Memory content
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            Created memory with generated ID
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush any pending provider work before shutdown."""
        ...


class LocalHindsightProvider(MemoryProvider):
    """
    SQLite-backed memory storage.

    Wraps the existing local_hindsight infrastructure.
    Uses aiosqlite for async-safe database operations.
    """

    def __init__(self, bank_id: str, max_retries: int = 3, retry_delay_ms: int = 1000):
        """
        Initialize the provider.

        Args:
            bank_id: Unique identifier for the memory bank
            max_retries: Maximum number of retries for failed operations
            retry_delay_ms: Delay between retries in milliseconds
        """
        if not bank_id or not bank_id.strip():
            raise ValueError("bank_id cannot be empty")

        # Sanitize bank_id to prevent path traversal
        import re
        safe_bank_id = re.sub(r'[^\w\-]', '_', bank_id.strip())
        if safe_bank_id != bank_id.strip():
            logger.warning(f"bank_id sanitized: '{bank_id}' -> '{safe_bank_id}'")

        self.bank_id = safe_bank_id
        self._db_path = Path.home() / ".hindsight" / f"{self.bank_id}.db"
        self._initialized = False
        self._max_retries = max_retries
        self._retry_delay_ms = retry_delay_ms
        self._store = SQLiteMemoryStore(self._db_path)
        self._resilience = SQLiteResilienceController(
            bank_id=self.bank_id,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
        )

    async def _ensure_db(self):
        """Initialize the local SQLite schema if needed."""
        if self._initialized:
            return

        await self._store.ensure_ready()
        logger.debug(f"Initialized memory database at {self._db_path}")
        self._initialized = True

    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic and circuit breaker."""
        return await self._resilience.execute(lambda: operation(*args, **kwargs))

    def _generate_id(self, content: str, user_id: str) -> str:
        """Generate a stable ID for a memory."""
        hash_input = f"{user_id}:{content}:{self.bank_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """Search memories by content using SQLite FTS5."""
        if not query:
            return []

        await self._ensure_db()

        fts_query = self._store.build_fts_query(query)
        if not fts_query:
            return []

        async def _query():
            rows = await self._store.recall(
                user_id=user_id,
                fts_query=fts_query,
                limit=limit,
            )
            return [
                Memory(
                    id=row.id,
                    content=row.content,
                    created_at=row.created_at,
                    metadata=row.metadata,
                )
                for row in rows
            ]

        try:
            return await self._retry_operation(_query)
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []

    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """Store a new memory (async-safe)."""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        await self._ensure_db()

        memory_id = self._generate_id(content, user_id)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata)

        async def _store():
            await self._store.store(
                memory_id=memory_id,
                user_id=user_id,
                content=content,
                created_at=created_at,
                metadata_json=metadata_json,
            )

        await self._retry_operation(_store)

        logger.debug(f"Stored memory {memory_id} for user {user_id}")
        return Memory(
            id=memory_id,
            content=content,
            created_at=created_at,
            metadata=metadata,
        )

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory by ID (async-safe)."""
        await self._ensure_db()

        async def _delete():
            return await self._store.delete(memory_id=memory_id, user_id=user_id)

        try:
            result = await self._retry_operation(_delete)
            if result:
                logger.debug(f"Deleted memory {memory_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            await self._ensure_db()
            return await self._store.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the database connection and reset state."""
        try:
            await self._store.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        # Reset circuit breaker state
        self._resilience.reset()
        self._initialized = False
        logger.debug(f"Closed database connection for {self.bank_id}")

    async def flush(self) -> None:
        """SQLite operations are already awaited eagerly; no extra flush is required."""
        return None


class SharedMemoryServiceProvider(MemoryProvider):
    """HTTP client for the shared local memory service."""

    def __init__(
        self,
        *,
        base_url: str,
        bank_id: str,
        actor_id: str,
        actor_secret: str,
        timeout_ms: int = 5000,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url cannot be empty")
        if not bank_id.strip():
            raise ValueError("bank_id cannot be empty")
        if not actor_id.strip():
            raise ValueError("actor_id cannot be empty")
        if not actor_secret.strip():
            raise ValueError("actor_secret cannot be empty")
        if timeout_ms < 100:
            raise ValueError("timeout_ms must be at least 100ms")

        self.base_url = normalized_base_url
        self.bank_id = bank_id.strip()
        self.actor_id = actor_id.strip()
        self.actor_secret = actor_secret.strip()
        self.timeout_ms = timeout_ms
        self.transport = SharedMemoryServiceTransport(
            base_url=self.base_url,
            actor_id=self.actor_id,
            actor_secret=self.actor_secret,
            timeout_ms=self.timeout_ms,
            client=client,
        )

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        if not query:
            return []
        payload = await self.transport.request_json(
            method="POST",
            path=f"/v1/default/banks/{self.bank_id}/memories/recall",
            user_id=user_id,
            json_body={"query": query, "limit": limit},
        )
        memories: List[Memory] = []
        for item in payload.get("results", []):
            document_id = item.get("document_id") or item.get("id")
            if not document_id:
                logger.warning(
                    "Skipping shared memory recall item without id: %s",
                    item,
                )
                continue
            content = item.get("text") or item.get("content")
            if not content:
                logger.warning(
                    "Skipping shared memory recall item without content: %s",
                    item,
                )
                continue
            memories.append(
                Memory(
                    id=document_id,
                    content=content,
                    created_at=item.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    metadata={
                        "user_id": user_id,
                        "match_context": item.get("context"),
                    },
                )
            )
        return memories

    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        category = metadata.get("category")
        created_at = datetime.now(timezone.utc).isoformat()
        payload = await self.transport.request_json(
            method="POST",
            path=f"/v1/default/banks/{self.bank_id}/memories",
            user_id=user_id,
            json_body={
                "items": [
                    {
                        "content": content,
                        "context": serialize_context(
                            user_id=user_id,
                            metadata=metadata,
                            category=category if isinstance(category, str) else None,
                        ),
                        "tags": metadata_to_tags(
                            user_id=user_id,
                            metadata=metadata,
                            category=category if isinstance(category, str) else None,
                        ),
                    }
                ]
            },
        )
        result_items = payload.get("results") or []
        if not result_items or not isinstance(result_items[0], dict):
            raise RuntimeError("Shared memory service store response is missing results")
        document_id = result_items[0].get("id")
        if not document_id:
            raise RuntimeError("Shared memory service store response is missing an id")
        return Memory(
            id=document_id,
            content=content,
            created_at=created_at,
            metadata={"user_id": user_id, **metadata},
        )

    async def delete(self, memory_id: str, user_id: str) -> bool:
        try:
            await self.transport.request_json(
                method="DELETE",
                path=f"/v1/default/banks/{self.bank_id}/documents/{memory_id}",
                user_id=user_id,
                expected_status=204,
            )
        except SharedMemoryServiceError as exc:
            if exc.status_code == 404:
                return False
            raise
        return True

    async def health_check(self) -> bool:
        try:
            return await self.transport.health_check()
        except Exception as exc:
            logger.error("Shared memory service health check failed: %s", exc)
            return False

    async def close(self) -> None:
        await self.transport.close()

    async def flush(self) -> None:
        """Shared service requests are synchronous per await; no extra flush is required."""
        return None


class MockProvider(MemoryProvider):
    """
    In-memory provider for testing.

    Memories are lost when the process exits.
    """

    def __init__(self):
        self._memories: List[Memory] = []
        self._id_counter = 0

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """
        Return all memories for user.

        Args:
            query: Ignored for this mock implementation
            user_id: User to filter by
            limit: Maximum results to return

        Returns:
            List of memories for this user
        """
        _ = query  # Simple mock, no semantic search
        user_memories = [
            m for m in self._memories if m.metadata.get("user_id") == user_id
        ]
        return user_memories[:limit]

    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """
        Store in memory.

        Args:
            content: Memory content
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            Created memory

        Raises:
            ValueError: If content is empty
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        self._id_counter += 1

        memory = Memory(
            id=f"mock-{self._id_counter}",
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"user_id": user_id, **metadata},
        )
        self._memories.append(memory)
        logger.debug(f"MockProvider: Stored memory {memory.id}")
        return memory

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """
        Delete from in-memory list.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        for i, m in enumerate(self._memories):
            if m.id == memory_id and m.metadata.get("user_id") == user_id:
                self._memories.pop(i)
                logger.debug(f"MockProvider: Deleted memory {memory_id}")
                return True
        return False

    async def health_check(self) -> bool:
        """Mock provider is always healthy."""
        return True

    async def close(self):
        """Clear all memories."""
        self._memories.clear()
        logger.debug("MockProvider: Cleared all memories")

    async def flush(self) -> None:
        """Mock provider does not buffer writes."""
        return None


def create_memory_provider(config: Any) -> MemoryProvider:
    """Build the configured v3 memory provider from SubconsciousConfig."""
    provider_name = config.memory_provider
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "shared_service":
        return SharedMemoryServiceProvider(
            base_url=config.memory_service_base_url,
            bank_id=config.bank_id,
            actor_id=config.memory_service_actor_id,
            actor_secret=config.memory_service_actor_secret,
            timeout_ms=config.memory_service_timeout_ms,
        )
    return LocalHindsightProvider(
        config.bank_id,
        max_retries=config.max_retries,
        retry_delay_ms=config.retry_delay_ms,
    )


async def close_memory_provider(provider: Optional[MemoryProvider]) -> None:
    """Close a provider if it exposes an async close hook."""
    if provider is not None and hasattr(provider, "close"):
        await provider.close()


async def flush_memory_provider(provider: Optional[MemoryProvider]) -> None:
    """Flush a provider if it exposes an async flush hook."""
    if provider is not None and hasattr(provider, "flush"):
        await provider.flush()
