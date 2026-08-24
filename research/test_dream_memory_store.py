"""Tests for Dream Memory Store — DreamCycleRecord, LocalDreamMemoryStore, MongoDBDreamStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.research.dream_memory_store import (
    DreamCycleRecord,
    DreamMemoryStore,
    LocalDreamMemoryStore,
    MongoDBDreamStore,
)
from ai.research.manager_factory import create_dream_manager

# ---------------------------------------------------------------------------
# DreamCycleRecord — pure data logic
# ---------------------------------------------------------------------------


class TestDreamCycleRecord:
    def test_roundtrip_to_dict_and_back(self) -> None:
        record = DreamCycleRecord(
            dream_id="dream-abc",
            user_id="user-1",
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T01:00:00",
            themes=["anxiety", "coping"],
            patterns=["recurring_anxiety"],
            emotional_tone="anxious",
            insight_count=3,
            consolidated_memory_ids=["mem-1", "mem-2"],
            nrem_completed=True,
            rem_completed=True,
            consolidation_completed=True,
            reflection_triggered=True,
        )
        d = record.to_dict()
        restored = DreamCycleRecord.from_dict(d)

        assert restored.dream_id == "dream-abc"
        assert restored.user_id == "user-1"
        assert restored.themes == ["anxiety", "coping"]
        assert restored.patterns == ["recurring_anxiety"]
        assert restored.emotional_tone == "anxious"
        assert restored.insight_count == 3
        assert restored.consolidated_memory_ids == ["mem-1", "mem-2"]
        assert restored.nrem_completed is True
        assert restored.rem_completed is True
        assert restored.consolidation_completed is True
        assert restored.reflection_triggered is True

    def test_defaults(self) -> None:
        record = DreamCycleRecord(
            dream_id="d1",
            user_id="u1",
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T01:00:00",
        )
        assert record.themes == []
        assert record.patterns == []
        assert record.emotional_tone is None
        assert record.insight_count == 0
        assert record.consolidated_memory_ids == []
        assert record.nrem_completed is False
        assert record.created_at is not None

    def test_to_dict_camel_case_keys(self) -> None:
        record = DreamCycleRecord(
            dream_id="d1",
            user_id="u1",
            start_time="s",
            end_time="e",
            themes=["t1"],
            patterns=["p1"],
        )
        d = record.to_dict()
        assert "dreamId" in d
        assert "userId" in d
        assert "consolidatedMemoryIds" in d
        assert "emotionalTone" in d
        assert "insightCount" in d
        assert "nremCompleted" in d

    def test_from_dict_missing_optional_fields(self) -> None:
        d = {
            "dreamId": "d1",
            "userId": "u1",
            "startTime": "s",
            "endTime": "e",
        }
        record = DreamCycleRecord.from_dict(d)
        assert record.themes == []
        assert record.patterns == []
        assert record.emotional_tone is None
        assert record.insight_count == 0

    def test_roundtrip_preserves_created_at(self) -> None:
        record = DreamCycleRecord(
            dream_id="d1",
            user_id="u1",
            start_time="s",
            end_time="e",
        )
        d = record.to_dict()
        restored = DreamCycleRecord.from_dict(d)
        assert restored.created_at == record.created_at


# ---------------------------------------------------------------------------
# LocalDreamMemoryStore — async adapter over LocalForesightMemoryManager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_local_manager() -> MagicMock:
    """Return a MagicMock that stands in for LocalForesightMemoryManager."""
    mgr = MagicMock()
    mgr.add_memory.return_value = "local-mem-id"
    mgr.get_all_memories.return_value = [
        {"_id": "m1", "content": "test", "category": "general", "userId": "u1"},
    ]
    mgr.close.return_value = None
    return mgr


@pytest.fixture
def local_store(mock_local_manager: MagicMock) -> LocalDreamMemoryStore:
    return LocalDreamMemoryStore(memory_manager=mock_local_manager)


class TestLocalDreamMemoryStore:
    def test_protocol_compliance(self, local_store: LocalDreamMemoryStore) -> None:
        assert isinstance(local_store, DreamMemoryStore)

    @pytest.mark.asyncio
    async def test_add_memory(self, local_store: LocalDreamMemoryStore, mock_local_manager: MagicMock) -> None:
        result = await local_store.add_memory(
            content="hello",
            user_id="u1",
            metadata={"tags": ["test"]},
            category="general",
        )
        assert result == "local-mem-id"
        mock_local_manager.add_memory.assert_called_once_with(
            content="hello",
            user_id="u1",
            metadata={"tags": ["test"]},
            category="general",
        )

    @pytest.mark.asyncio
    async def test_get_all_memories(self, local_store: LocalDreamMemoryStore, mock_local_manager: MagicMock) -> None:
        memories = await local_store.get_all_memories(user_id="u1", limit=10)
        assert len(memories) == 1
        assert memories[0]["_id"] == "m1"
        mock_local_manager.get_all_memories.assert_called_once_with(user_id="u1", limit=10)

    @pytest.mark.asyncio
    async def test_save_dream_cycle(self, local_store: LocalDreamMemoryStore, mock_local_manager: MagicMock) -> None:
        record = DreamCycleRecord(
            dream_id="dream-1",
            user_id="u1",
            start_time="s",
            end_time="e",
            themes=["anxiety"],
        )
        await local_store.save_dream_cycle(record)
        # save_dream_cycle calls add_memory internally
        mock_local_manager.add_memory.assert_called_once()
        call_kwargs = mock_local_manager.add_memory.call_args.kwargs
        assert call_kwargs["user_id"] == "u1"
        assert "Dream cycle dream-1" in call_kwargs["content"]
        assert call_kwargs["category"] == "therapeutic_insight"

    @pytest.mark.asyncio
    async def test_close(self, local_store: LocalDreamMemoryStore, mock_local_manager: MagicMock) -> None:
        await local_store.close()
        mock_local_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_returns_none_on_empty_content(
        self, local_store: LocalDreamMemoryStore, mock_local_manager: MagicMock
    ) -> None:
        mock_local_manager.add_memory.return_value = None
        result = await local_store.add_memory(content="", user_id="u1")
        assert result is None


# ---------------------------------------------------------------------------
# MongoDBDreamStore — async Motor-backed store
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mongo_collections() -> tuple[MagicMock, MagicMock]:
    """Return (mock_memories_collection, mock_dreams_collection).

    Uses MagicMock (not AsyncMock) so that chained cursor methods like
    ``.find().sort().limit()`` return plain mocks instead of coroutines.
    Only methods that are actually awaited (insert_one, find_one,
    replace_one, delete_one) use AsyncMock.
    """
    memories = MagicMock(name="memories")
    dreams = MagicMock(name="dreams")
    memories.insert_one = AsyncMock()
    memories.replace_one = AsyncMock()
    memories.find_one = AsyncMock()
    memories.delete_one = AsyncMock()
    memories.count_documents = AsyncMock()
    dreams.insert_one = AsyncMock()
    dreams.replace_one = AsyncMock()
    dreams.find_one = AsyncMock()
    dreams.delete_one = AsyncMock()
    return memories, dreams


@pytest.fixture
def mongodb_store(
    mock_mongo_collections: tuple[AsyncMock, AsyncMock],
) -> MongoDBDreamStore:
    memories_mock, dreams_mock = mock_mongo_collections
    store = MongoDBDreamStore.__new__(MongoDBDreamStore)
    store._db = MagicMock()
    store._memories = memories_mock
    store._dreams = dreams_mock
    return store


class TestMongoDBDreamStore:
    def test_protocol_compliance(self, mongodb_store: MongoDBDreamStore) -> None:
        assert isinstance(mongodb_store, DreamMemoryStore)

    @pytest.mark.asyncio
    async def test_get_all_memories(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        memories_mock, _ = mock_mongo_collections
        # Simulate async cursor
        cursor = AsyncMock()
        cursor.__aiter__.return_value = [
            {"_id": "abc123", "userId": "u1", "content": "memory content"},
        ]
        sort_result = MagicMock()
        sort_result.limit.return_value = cursor
        memories_mock.find.return_value.sort.return_value = sort_result

        results = await mongodb_store.get_all_memories(user_id="u1", limit=10)
        assert len(results) == 1
        assert results[0]["_id"] == "abc123"
        assert results[0]["content"] == "memory content"
        memories_mock.find.assert_called_once_with({"userId": "u1"})

    @pytest.mark.asyncio
    async def test_get_all_memories_empty(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        memories_mock, _ = mock_mongo_collections
        cursor = AsyncMock()
        cursor.__aiter__.return_value = []
        sort_result = MagicMock()
        sort_result.limit.return_value = cursor
        memories_mock.find.return_value.sort.return_value = sort_result

        results = await mongodb_store.get_all_memories(user_id="u1", limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_memory(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        memories_mock, _ = mock_mongo_collections
        memories_mock.insert_one = AsyncMock()
        memories_mock.insert_one.return_value = None

        result = await mongodb_store.add_memory(
            content="test memory",
            user_id="u1",
            category="general",
        )
        assert result is not None
        assert isinstance(result, str)
        memories_mock.insert_one.assert_called_once()
        doc = memories_mock.insert_one.call_args[0][0]
        assert doc["userId"] == "u1"
        assert doc["content"] == "test memory"
        assert doc["category"] == "general"
        assert doc["sourceService"] == "dream_manager"

    @pytest.mark.asyncio
    async def test_add_memory_with_metadata_object(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        memories_mock, _ = mock_mongo_collections
        memories_mock.insert_one = AsyncMock()

        from ai.research.reflection_types import MemoryCategory, MemoryMetadata

        metadata = MemoryMetadata(
            category=MemoryCategory.THERAPEUTIC_INSIGHT,
            tags=["dream:abc", "theme:anxiety"],
            session_id="session-1",
        )
        result = await mongodb_store.add_memory(
            content="insight content",
            user_id="u1",
            metadata=metadata,
            category="therapeutic_insight",
        )
        assert result is not None
        doc = memories_mock.insert_one.call_args[0][0]
        assert doc["tags"] == ["dream:abc", "theme:anxiety"]
        assert doc["sessionId"] == "session-1"

    @pytest.mark.asyncio
    async def test_add_memory_failure_returns_none(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        memories_mock, _ = mock_mongo_collections
        memories_mock.insert_one = AsyncMock(side_effect=Exception("DB error"))

        result = await mongodb_store.add_memory(content="x", user_id="u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_dream_cycle(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        _, dreams_mock = mock_mongo_collections
        dreams_mock.replace_one = AsyncMock()

        record = DreamCycleRecord(
            dream_id="dream-1",
            user_id="u1",
            start_time="s",
            end_time="e",
            themes=["anxiety"],
        )
        await mongodb_store.save_dream_cycle(record)

        dreams_mock.replace_one.assert_called_once()
        call_args = dreams_mock.replace_one.call_args
        assert call_args[0][0] == {"_id": "dream-1"}  # filter
        doc = call_args[0][1]
        assert doc["dreamId"] == "dream-1"
        assert doc["userId"] == "u1"
        assert doc["_hipaaCompliant"] is True

    @pytest.mark.asyncio
    async def test_get_dream_cycle(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        _, dreams_mock = mock_mongo_collections
        dreams_mock.find_one = AsyncMock(
            return_value={
                "_id": "dream-1",
                "dreamId": "dream-1",
                "userId": "u1",
                "startTime": "s",
                "endTime": "e",
                "themes": ["anxiety"],
            }
        )

        record = await mongodb_store.get_dream_cycle("dream-1")
        assert record is not None
        assert record.dream_id == "dream-1"
        assert record.themes == ["anxiety"]

    @pytest.mark.asyncio
    async def test_get_dream_cycle_not_found(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        _, dreams_mock = mock_mongo_collections
        dreams_mock.find_one = AsyncMock(return_value=None)

        record = await mongodb_store.get_dream_cycle("nonexistent")
        assert record is None

    @pytest.mark.asyncio
    async def test_list_dream_cycles(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        _, dreams_mock = mock_mongo_collections
        cursor = AsyncMock()
        cursor.__aiter__.return_value = [
            {"_id": "d1", "dreamId": "d1", "userId": "u1", "startTime": "s", "endTime": "e"},
        ]
        sort_result = MagicMock()
        sort_result.limit.return_value = cursor
        dreams_mock.find.return_value.sort.return_value = sort_result

        records = await mongodb_store.list_dream_cycles(user_id="u1", limit=5)
        assert len(records) == 1
        assert records[0].dream_id == "d1"

    @pytest.mark.asyncio
    async def test_close(
        self, mongodb_store: MongoDBDreamStore, mock_mongo_collections: tuple[AsyncMock, AsyncMock]
    ) -> None:
        # Need to patch the global _motor_client
        with patch("ai.research.dream_memory_store._motor_client") as mock_client:
            await mongodb_store.close()
            mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Factory — create_dream_manager
# ---------------------------------------------------------------------------


class TestCreateDreamManager:
    @pytest.mark.asyncio
    async def test_local_fallback_when_no_mongodb_uri(self) -> None:
        with (
            patch.dict("os.environ", {"MONGODB_URI": ""}, clear=False),
            patch("ai.research.local_foresight_manager.LocalForesightMemoryManager") as mock_cls,
            patch("ai.research.local_memory_settings.resolve_local_memory_settings") as mock_settings,
        ):
            mock_cls.return_value = MagicMock()
            mock_settings.return_value = MagicMock(db_path="/tmp/test.db", bank_id="test")
            dm = create_dream_manager()
            from ai.research.dream_memory_store import LocalDreamMemoryStore

            assert isinstance(dm.memory_store, LocalDreamMemoryStore)
            await dm.close()

    @pytest.mark.asyncio
    async def test_mongodb_when_uri_provided(self) -> None:
        with patch("ai.research.dream_memory_store._get_motor_db") as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            dm = create_dream_manager(mongodb_uri="mongodb://localhost:27017")
            from ai.research.dream_memory_store import MongoDBDreamStore

            assert isinstance(dm.memory_store, MongoDBDreamStore)
            await dm.close()
