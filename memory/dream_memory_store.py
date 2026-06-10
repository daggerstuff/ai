"""
Dream Memory Store — Async storage interface for Dream Manager.

Provides two backends:
  - LocalDreamMemoryStore: wraps LocalForesightMemoryManager (SQLite)
  - MongoDBDreamStore: async MongoDB-backed store using Motor

Both implement the same protocol so the Dream Manager can use either
depending on deployment configuration.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from .reflection_types import MemoryCategory, MemoryMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DreamCycleRecord:
    """Persistent record of a dream cycle."""

    dream_id: str
    user_id: str
    start_time: str
    end_time: str
    themes: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    emotional_tone: str | None = None
    insight_count: int = 0
    consolidated_memory_ids: list[str] = field(default_factory=list)
    nrem_completed: bool = False
    rem_completed: bool = False
    consolidation_completed: bool = False
    reflection_triggered: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dreamId": self.dream_id,
            "userId": self.user_id,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "themes": self.themes,
            "patterns": self.patterns,
            "emotionalTone": self.emotional_tone,
            "insightCount": self.insight_count,
            "consolidatedMemoryIds": self.consolidated_memory_ids,
            "nremCompleted": self.nrem_completed,
            "remCompleted": self.rem_completed,
            "consolidationCompleted": self.consolidation_completed,
            "reflectionTriggered": self.reflection_triggered,
            "createdAt": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DreamCycleRecord:
        return cls(
            dream_id=data["dreamId"],
            user_id=data["userId"],
            start_time=data.get("startTime", ""),
            end_time=data.get("endTime", ""),
            themes=list(data.get("themes", [])),
            patterns=list(data.get("patterns", [])),
            emotional_tone=data.get("emotionalTone"),
            insight_count=data.get("insightCount", 0),
            consolidated_memory_ids=list(data.get("consolidatedMemoryIds", [])),
            nrem_completed=data.get("nremCompleted", False),
            rem_completed=data.get("remCompleted", False),
            consolidation_completed=data.get("consolidationCompleted", False),
            reflection_triggered=data.get("reflectionTriggered", False),
            created_at=data.get("createdAt", datetime.now(UTC).isoformat()),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DreamMemoryStore(Protocol):
    """Async interface for dream-cycle storage operations."""

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str | None:
        """Store a memory and return its ID (or None if gated)."""
        ...

    async def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent memories for a user."""
        ...

    async def save_dream_cycle(self, record: DreamCycleRecord) -> None:
        """Persist a completed dream cycle."""
        ...

    async def close(self) -> None:
        """Release underlying resources."""
        ...


# ---------------------------------------------------------------------------
# Local implementation (wraps LocalForesightMemoryManager)
# ---------------------------------------------------------------------------


class LocalDreamMemoryStore:
    """Async adapter over LocalForesightMemoryManager (SQLite).

    Sync calls are bridged via ``asyncio.to_thread`` so the Dream Manager's
    async code can use this without modification.
    """

    def __init__(self, memory_manager: Any | None = None) -> None:
        if memory_manager is not None:
            self._manager = memory_manager
        else:
            from .local_foresight_manager import LocalForesightMemoryManager
            from .local_memory_settings import resolve_local_memory_settings

            settings = resolve_local_memory_settings()
            self._manager = LocalForesightMemoryManager(
                db_path=settings.db_path,
                bank_id=settings.bank_id,
            )

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str | None:
        return await asyncio.to_thread(
            self._manager.add_memory,
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )

    async def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._manager.get_all_memories, user_id=user_id, limit=limit)

    async def save_dream_cycle(self, record: DreamCycleRecord) -> None:
        """Persist dream cycle as a memory document in the local store."""
        content = (
            f"Dream cycle {record.dream_id}: {len(record.themes)} themes, "
            f"{len(record.patterns)} patterns, "
            f"{record.insight_count} insights"
        )
        metadata = MemoryMetadata(
            category=MemoryCategory.THERAPEUTIC_INSIGHT,
            user_id=record.user_id,
            tags=[
                f"dream:{record.dream_id}",
                *[f"theme:{t}" for t in record.themes[:3]],
            ],
        )
        await self.add_memory(
            content=content,
            user_id=record.user_id,
            metadata=metadata,
            category=MemoryCategory.THERAPEUTIC_INSIGHT.value,
        )

    async def close(self) -> None:
        if hasattr(self._manager, "close"):
            await asyncio.to_thread(self._manager.close)


# ---------------------------------------------------------------------------
# MongoDB implementation (async via Motor)
# ---------------------------------------------------------------------------

# Lazy-init singleton for the Motor client so we don't reconnect on every
# DreamManager instantiation.
_motor_client: Any | None = None
_motor_db: Any | None = None


def _get_motor_db(mongodb_uri: str | None = None) -> Any:
    """Return a lazily-initialised Motor database handle."""
    global _motor_client, _motor_db
    if _motor_db is not None:
        return _motor_db

    uri = mongodb_uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError as exc:
        raise RuntimeError(
            "motor package is required for MongoDBDreamStore. "
            "Ensure it is installed (ai/pyproject.toml has motor>=3.6.0)."
        ) from exc

    _motor_client = AsyncIOMotorClient(uri)
    _motor_db = _motor_client["pixelated_ai"]
    logger.info("MongoDBDreamStore connected to %s", uri)
    return _motor_db


class MongoDBDreamStore:
    """Async MongoDB-backed store for dream cycles and memories.

    Writes consolidated memories to the ``unified_memories`` collection
    (consistent with ``ai-services/memory_adapter.py``) and stores dream
    cycle metadata in a ``dream_cycles`` collection.

    Usage::

        store = MongoDBDreamStore()
        await store.save_dream_cycle(record)
        memories = await store.get_all_memories(user_id="user-123")
        await store.close()
    """

    # Same collection name used by UnifiedMemoryAdapter
    UNIFIED_MEMORIES_COLLECTION = "unified_memories"
    DREAM_CYCLES_COLLECTION = "dream_cycles"

    def __init__(self, mongodb_uri: str | None = None) -> None:
        self._db = _get_motor_db(mongodb_uri)
        self._memories: Any = self._db[self.UNIFIED_MEMORIES_COLLECTION]
        self._dreams: Any = self._db[self.DREAM_CYCLES_COLLECTION]

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    async def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent unified memories for the given user, newest first."""
        cursor = self._memories.find({"userId": user_id}).sort("createdAt", -1).limit(limit)
        results: list[dict[str, Any]] = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str | None:
        """Store a memory document in the unified_memories collection."""
        import hashlib
        import time

        memory_id = hashlib.sha256(f"{user_id}:{time.time()}:{content[:64]}".encode()).hexdigest()[:24]

        # Handle MemoryMetadata objects
        if metadata is not None and hasattr(metadata, "to_dict"):
            meta_dict = metadata.to_dict()
        elif isinstance(metadata, dict):
            meta_dict = metadata
        else:
            meta_dict = {}

        tag_list: list[str] = []
        if isinstance(meta_dict.get("tags"), list):
            tag_list = meta_dict["tags"]

        doc: dict[str, Any] = {
            "_id": memory_id,
            "userId": user_id,
            "content": content,
            "category": category or meta_dict.get("category", "general"),
            "tags": tag_list,
            "scope": "session",
            "retention": "short_term",
            "importance": 0.5,
            "version": 1,
            "schemaVersion": "1.0",
            "sourceService": "dream_manager",
            "strengthTrend": "stable",
            "activationCount": 0,
            "retrievalCount": 0,
            "isGhost": False,
            "createdAt": datetime.now(UTC).isoformat(),
            "updatedAt": datetime.now(UTC).isoformat(),
        }

        if meta_dict.get("session_id"):
            doc["sessionId"] = meta_dict["session_id"]

        try:
            await self._memories.insert_one(doc)
            return memory_id
        except Exception as e:
            logger.error("MongoDBDreamStore.add_memory failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Dream cycle operations
    # ------------------------------------------------------------------

    async def save_dream_cycle(self, record: DreamCycleRecord) -> None:
        """Persist a dream cycle record to the dream_cycles collection."""
        doc = record.to_dict()
        doc["_id"] = record.dream_id
        doc["_hipaaCompliant"] = True
        doc["_storedAt"] = datetime.now(UTC).isoformat()

        try:
            await self._dreams.replace_one(
                {"_id": record.dream_id},
                doc,
                upsert=True,
            )
            logger.debug("Dream cycle %s saved", record.dream_id)
        except Exception as e:
            logger.error("Failed to save dream cycle %s: %s", record.dream_id, e)

    async def get_dream_cycle(self, dream_id: str) -> DreamCycleRecord | None:
        """Retrieve a dream cycle record by ID."""
        doc = await self._dreams.find_one({"_id": dream_id})
        if doc is None:
            return None
        return DreamCycleRecord.from_dict(doc)

    async def list_dream_cycles(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[DreamCycleRecord]:
        """List recent dream cycles for a user, newest first."""
        cursor = self._dreams.find({"userId": user_id}).sort("createdAt", -1).limit(limit)
        results: list[DreamCycleRecord] = []
        async for doc in cursor:
            results.append(DreamCycleRecord.from_dict(doc))
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        global _motor_client
        if _motor_client is not None:
            _motor_client.close()
            logger.info("MongoDBDreamStore closed")
