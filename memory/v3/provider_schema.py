"""SQLite schema management helpers for subconscious v3 local memory."""

from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)


async def ensure_local_hindsight_schema(conn: aiosqlite.Connection) -> None:
    """Create and refresh the local hindsight schema used by v3."""
    await _ensure_base_tables(conn)
    await _ensure_fts_table(conn)
    await _ensure_fts_triggers(conn)
    await _rebuild_fts_if_empty(conn)
    await conn.commit()
    logger.debug("Ensured subconscious v3 local memory schema")


async def _ensure_base_tables(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT
        )
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)
        """
    )


async def _ensure_fts_table(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(id UNINDEXED, user_id UNINDEXED, content)
        """
    )


async def _ensure_fts_triggers(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, id, user_id, content)
            VALUES (new.rowid, new.id, new.user_id, new.content);
        END
        """
    )
    await conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM memories_fts WHERE rowid = old.rowid;
        END
        """
    )
    await conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            DELETE FROM memories_fts WHERE rowid = old.rowid;
            INSERT INTO memories_fts(rowid, id, user_id, content)
            VALUES (new.rowid, new.id, new.user_id, new.content);
        END
        """
    )


async def _rebuild_fts_if_empty(conn: aiosqlite.Connection) -> None:
    cursor = await conn.execute("SELECT count(*) FROM memories_fts")
    existing_fts_rows = await cursor.fetchone()
    needs_rebuild = bool(existing_fts_rows and existing_fts_rows[0] == 0)
    if not needs_rebuild:
        return
    await conn.execute(
        """
        INSERT INTO memories_fts(rowid, id, user_id, content)
        SELECT rowid, id, user_id, content FROM memories
        """
    )
