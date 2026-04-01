"""
Memory provider interface and implementations.

Pluggable architecture - swap backends without changing code.
"""
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

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
        """Search for relevant memories."""
        ...

    @abstractmethod
    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """Store a new memory."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...


class LocalHindsightProvider(MemoryProvider):
    """
    SQLite-backed memory storage.

    Wraps the existing local_hindsight infrastructure.
    """

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self._db_path = Path.home() / ".hindsight" / f"{bank_id}.db"
        self._ensure_db()

    def _ensure_db(self):
        """Create tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
            """)
            conn.commit()

    def _generate_id(self, content: str, user_id: str) -> str:
        """Generate a stable ID for a memory."""
        import hashlib
        hash_input = f"{user_id}:{content}:{self.bank_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """Search memories by content similarity (simple LIKE for now)."""
        # Simple text search - could be enhanced with embeddings
        search_terms = query.lower().split()[:5]  # Top 5 words

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build LIKE query
            like_clauses = " OR ".join(["content LIKE ?" for _ in search_terms])
            params = [f"%{term}%" for term in search_terms]
            params.append(user_id)

            cursor = conn.execute(
                f"""
                SELECT id, content, created_at, metadata
                FROM memories
                WHERE user_id = ? AND ({like_clauses})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params[:len(search_terms)] + [user_id, limit],
            )

            rows = cursor.fetchall()
            return [
                Memory(
                    id=row["id"],
                    content=row["content"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
                for row in rows
            ]

    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """Store a new memory."""
        from datetime import datetime, timezone

        memory_id = self._generate_id(content, user_id)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (id, user_id, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, content, created_at, metadata_json),
            )
            conn.commit()

        return Memory(
            id=memory_id,
            content=content,
            created_at=created_at,
            metadata=metadata,
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0


class MockProvider(MemoryProvider):
    """
    In-memory provider for testing.

    Memories are lost when the process exits.
    """

    def __init__(self):
        self._memories: List[Memory] = []
        self._id_counter = 0

    async def recall(self, query: str, user_id: str, limit: int) -> List[Memory]:
        """Return all memories for user (ignores query for simplicity)."""
        _ = query  # Simple mock, no semantic search
        user_memories = [m for m in self._memories if m.metadata.get("user_id") == user_id]
        return user_memories[:limit]

    async def store(self, content: str, user_id: str, metadata: dict) -> Memory:
        """Store in memory."""
        self._id_counter += 1
        from datetime import datetime, timezone

        memory = Memory(
            id=f"mock-{self._id_counter}",
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"user_id": user_id, **metadata},
        )
        self._memories.append(memory)
        return memory

    async def delete(self, memory_id: str) -> bool:
        """Delete from in-memory list."""
        for i, m in enumerate(self._memories):
            if m.id == memory_id:
                self._memories.pop(i)
                return True
        return False
