"""SQLite persistence helpers for the subconscious v3 local provider."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .provider_schema import ensure_local_hindsight_schema

logger = logging.getLogger(__name__)

DB_TIMEOUT_MS = 30000


@dataclass
class SQLiteMemoryRecord:
    """Local SQLite row mapped into a memory-like record."""

    id: str
    content: str
    created_at: str
    metadata: dict


class SQLiteMemoryStore:
    """Own low-level SQLite connection, schema, and FTS query behavior."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def ensure_ready(self) -> None:
        """Ensure the SQLite connection and schema are available."""
        if self._initialized:
            return
        conn = await self.get_connection()
        await ensure_local_hindsight_schema(conn)
        self._initialized = True

    async def get_connection(self) -> aiosqlite.Connection:
        """Open or reuse the SQLite connection."""
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = await aiosqlite.connect(
                self._db_path, timeout=DB_TIMEOUT_MS / 1000
            )
        return self._connection

    def build_fts_query(self, query: str) -> Optional[str]:
        """Build a safe FTS5 MATCH query from user text."""
        search_terms = query.lower().split()[:5]
        sanitized_terms = [
            self._sanitize_fts_term(term) for term in search_terms if term.strip()
        ]
        if not sanitized_terms:
            return None
        return " AND ".join(f'"{term}"' for term in sanitized_terms)

    async def recall(
        self,
        *,
        user_id: str,
        fts_query: str,
        limit: int,
    ) -> List[SQLiteMemoryRecord]:
        """Execute an FTS-backed recall query."""
        conn = await self.get_connection()
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """
            SELECT m.id, m.content, m.created_at, m.metadata
            FROM memories AS m
            JOIN memories_fts AS fts ON fts.rowid = m.rowid
            WHERE fts.user_id = ? AND memories_fts MATCH ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, fts_query, limit),
        )
        rows = await cursor.fetchall()
        return [
            SQLiteMemoryRecord(
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    async def store(
        self,
        *,
        memory_id: str,
        user_id: str,
        content: str,
        created_at: str,
        metadata_json: str,
    ) -> None:
        """Persist a memory row."""
        conn = await self.get_connection()
        await conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, user_id, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, user_id, content, created_at, metadata_json),
        )
        await conn.commit()

    async def delete(self, *, memory_id: str, user_id: str) -> bool:
        """Delete a memory row scoped to the owning user."""
        conn = await self.get_connection()
        cursor = await conn.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def health_check(self) -> bool:
        """Check whether SQLite is reachable."""
        await self.ensure_ready()
        conn = await self.get_connection()
        cursor = await conn.execute("SELECT 1")
        await cursor.fetchone()
        return True

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._connection is None:
            return
        try:
            await self._connection.close()
        finally:
            self._connection = None
            self._initialized = False

    def _sanitize_fts_term(self, term: str) -> str:
        """Remove FTS control characters and collapse whitespace."""
        sanitized = re.sub(r'["*:^(){}[\]]', " ", term)
        sanitized = " ".join(sanitized.split())
        return sanitized.strip()
