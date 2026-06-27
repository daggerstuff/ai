"""Tests for MemoryManager — including dream-cycle integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.api.memory.memory_manager import MemoryManager, MessageRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client():
    """A minimal memory client stub that satisfies MemoryManager."""
    client = MagicMock()
    client.add.return_value = {"results": [{"id": "mem-1"}]}
    client.get_all.return_value = {"results": []}
    client.search.return_value = {"results": []}
    return client


@pytest.fixture
def memory_manager(fake_client):
    """MemoryManager instance with a fake client and no MongoDB."""
    return MemoryManager(memory_client=fake_client, mongodb_uri=None)


# ---------------------------------------------------------------------------
# MemoryManager basics
# ---------------------------------------------------------------------------


class TestMemoryManagerBasics:
    def test_init_requires_client(self):
        with pytest.raises(ValueError, match="memory_client is required"):
            MemoryManager(memory_client=None)

    def test_add_message_success(self, memory_manager: MemoryManager):
        result = memory_manager.add_message(
            user_id="user-1",
            session_id="session-1",
            content="Hello",
            role=MessageRole.USER,
        )
        assert result is True
        assert memory_manager.client.add.called

    def test_add_message_fails_when_client_lacks_add(self, fake_client):
        del fake_client.add
        mm = MemoryManager(memory_client=fake_client)
        assert mm.add_message(user_id="u", session_id="s", content="c", role="user") is False

    def test_get_conversation_history_empty(self, memory_manager: MemoryManager):
        history = memory_manager.get_conversation_history(
            user_id="user-1",
            session_id="session-1",
        )
        assert history == []

    def test_memory_stats(self, memory_manager: MemoryManager):
        stats = memory_manager.get_memory_stats(session_id="s-1")
        assert stats["session_id"] == "s-1"
        assert "provider" in stats


# ---------------------------------------------------------------------------
# Dream cycle integration
# ---------------------------------------------------------------------------


class TestDreamCycleIntegration:
    """Tests for the DreamManager wiring inside MemoryManager."""

    @pytest.mark.asyncio
    async def test_dream_manager_is_lazily_created(self, memory_manager: MemoryManager):
        """DreamManager should not be created until first access."""
        assert memory_manager._dream_manager is None
        _ = memory_manager.dream_manager
        assert memory_manager._dream_manager is not None

    @pytest.mark.asyncio
    async def test_dream_manager_uses_local_store_when_no_mongodb(self, memory_manager: MemoryManager):
        """Without MONGODB_URI, DreamManager should use LocalDreamMemoryStore."""
        # Clear MONGODB_URI and set required local store env vars so LocalDreamMemoryStore is used
        with patch.dict(
            "os.environ",
            {
                "FORESIGHT_LOCAL_DB_PATH": "/tmp/pixelated-test-memory.db",
                "HINDSIGHT_LOCAL_DB_PATH": "/tmp/pixelated-test-memory.db",
            },
            clear=True,
        ):
            mm = MemoryManager(memory_client=memory_manager.client, mongodb_uri=None)
            dm = mm.dream_manager
            from ai.memory.dream_memory_store import LocalDreamMemoryStore

            assert isinstance(dm.memory_store, LocalDreamMemoryStore)

    @pytest.mark.asyncio
    async def test_trigger_dream_cycle_with_few_memories_skips_processing(self, memory_manager: MemoryManager):
        """When there are fewer than min_memories_for_dream, the cycle should short-circuit."""
        with patch.dict(
            "os.environ",
            {
                "FORESIGHT_LOCAL_DB_PATH": "/tmp/pixelated-test-memory.db",
                "HINDSIGHT_LOCAL_DB_PATH": "/tmp/pixelated-test-memory.db",
            },
            clear=True,
        ):
            mm = MemoryManager(memory_client=memory_manager.client, mongodb_uri=None)
            result = await mm.trigger_dream_cycle(user_id="test-user")

        assert "dream_id" in result
        assert result["user_id"] == "test-user"
        # Phases are nested under "phases" in the DreamCycleResult.to_dict() output
        assert result.get("phases", {}).get("nrem_completed") is False
        assert result.get("phases", {}).get("rem_completed") is False

    @pytest.mark.asyncio
    async def test_get_dream_status_returns_none_for_unknown(self, memory_manager: MemoryManager):
        status = await memory_manager.get_dream_status("nonexistent-dream")
        assert status is None

    @pytest.mark.asyncio
    async def test_close_dream_manager_releases_resources(self, memory_manager: MemoryManager):
        # Access dream_manager to initialize it
        _ = memory_manager.dream_manager
        assert memory_manager._dream_manager is not None

        await memory_manager.close_dream_manager()
        assert memory_manager._dream_manager is None

    @pytest.mark.asyncio
    async def test_close_releases_dream_manager(self, memory_manager: MemoryManager):
        _ = memory_manager.dream_manager
        assert memory_manager._dream_manager is not None

        await memory_manager.close()
        assert memory_manager._dream_manager is None


# ---------------------------------------------------------------------------
# DreamManager with MongoDB (mocked)
# ---------------------------------------------------------------------------


class TestDreamCycleWithMongoDB:
    @pytest.mark.asyncio
    async def test_dream_manager_uses_mongodb_store_when_uri_provided(self, fake_client):
        """When MONGODB_URI is provided, DreamManager should use MongoDBDreamStore."""
        with patch.dict("os.environ", {"MONGODB_URI": "mongodb://localhost:27017/test"}):
            mm = MemoryManager(memory_client=fake_client, mongodb_uri="mongodb://localhost:27017/test")
            dm = mm.dream_manager
            from ai.memory.dream_memory_store import MongoDBDreamStore

            assert isinstance(dm.memory_store, MongoDBDreamStore)

    @pytest.mark.asyncio
    async def test_dream_cycle_with_mongodb_mocked_store(self, fake_client):
        """Simulate a dream cycle with a mocked MongoDB store."""
        with patch.dict("os.environ", {}, clear=True):
            mm = MemoryManager(memory_client=fake_client, mongodb_uri="mongodb://localhost:27017/test")

            # Replace the real store with a mock
            mock_store = AsyncMock()
            mock_store.add_memory.return_value = "mock-mem-id"
            mock_store.get_all_memories.return_value = []
            mock_store.save_dream_cycle.return_value = None
            mock_store.close.return_value = None

            mm.dream_manager.memory_store = mock_store

            result = await mm.trigger_dream_cycle(user_id="mongo-user")
            assert "dream_id" in result
            assert result["user_id"] == "mongo-user"
            # Few or no memories → early return, phases not completed
            assert result.get("phases", {}).get("nrem_completed") is False
