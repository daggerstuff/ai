"""
Memory provider interface and implementations.

Pluggable architecture - swap backends without changing code.
"""

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

__all__ = ["Memory", "MemoryProvider", "LocalHindsightProvider", "MockProvider"]

logger = logging.getLogger(__name__)

# Constants for connection pooling
DB_POOL_SIZE = 5
DB_TIMEOUT_MS = 30000  # 30 seconds


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
    async def delete(self, memory_id: str) -> bool:
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
        self._connection: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create a database connection with connection pooling."""
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = await aiosqlite.connect(
                self._db_path, timeout=DB_TIMEOUT_MS / 1000
            )
        return self._connection

    async def _ensure_db(self):
        """Create tables if they don't exist (async)."""
        if self._initialized:
            return

        conn = await self._get_connection()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)
        """)
        await conn.commit()

        logger.debug(f"Initialized memory database at {self._db_path}")
        self._initialized = True

    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except (aiosqlite.Error, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_delay_ms / 1000
                    logger.warning(
                        f"DB failed: {attempt + 1}/{self._max_retries + 1}. "
                        f"Retry in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"DB failed after {self._max_retries + 1} attempts: {e}"
                    )
                    raise

        raise (
            last_error
            if last_error
            else RuntimeError("Unexpected error in retry logic")
        )

    def _generate_id(self, content: str, user_id: str) -> str:
        """Generate a stable ID for a memory."""
        hash_input = f"{user_id}:{content}:{self.bank_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape LIKE wildcards to prevent SQL injection."""
        return re.sub(r"([%_\\])", r"\\\1", value)

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """Search memories by content (async-safe with proper escaping)."""
        if not query:
            return []

        await self._ensure_db()

        # Split into search terms and escape each one
        search_terms = query.lower().split()[:5]  # Top 5 words
        escaped_terms = [self._escape_like(term) for term in search_terms]

        async def _query():
            conn = await self._get_connection()
            conn.row_factory = aiosqlite.Row

            # Build LIKE query with escaped terms
            like_clauses = " OR ".join(
                ["content LIKE ? ESCAPE '\'" for _ in escaped_terms]
            )
            params = [f"%{term}%" for term in escaped_terms]
            params.insert(0, user_id)
            params.append(limit)

            cursor = await conn.execute(
                f"""
                SELECT id, content, created_at, metadata
                FROM memories
                WHERE user_id = ? AND ({like_clauses})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            )

            rows = await cursor.fetchall()
            return [
                Memory(
                    id=row["id"],
                    content=row["content"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"] or "{}"),
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
            conn = await self._get_connection()
            await conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, user_id, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, content, created_at, metadata_json),
            )
            await conn.commit()

        await self._retry_operation(_store)

        logger.debug(f"Stored memory {memory_id} for user {user_id}")
        return Memory(
            id=memory_id,
            content=content,
            created_at=created_at,
            metadata=metadata,
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID (async-safe)."""
        await self._ensure_db()

        async def _delete():
            conn = await self._get_connection()
            cursor = await conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            await conn.commit()
            return cursor.rowcount > 0

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
            conn = await self._get_connection()
            cursor = await conn.execute("SELECT 1")
            await cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.debug("Closed database connection")


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

    async def delete(self, memory_id: str) -> bool:
        """
        Delete from in-memory list.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        for i, m in enumerate(self._memories):
            if m.id == memory_id:
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
