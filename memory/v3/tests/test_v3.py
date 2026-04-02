"""
Unit tests for Claude Subconscious v3.

Tests core functionality without requiring external dependencies.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from ai.api.mcp_server import memory_auth
from ai.api.mcp_server.memory_server import create_memory_server
from ai.memory.local_hindsight_manager import LocalHindsightMemoryManager
from ai.memory.v3.config import SubconsciousConfig
from ai.memory.v3.context import (
    SubconsciousState,
    get_subconscious,
    reset_subconscious,
    set_subconscious,
)
from ai.memory.v3.provider import (
    LocalHindsightProvider,
    MockProvider,
    SharedMemoryServiceProvider,
    create_memory_provider,
)
from ai.memory.v3.reflection import extract_learnings_from_llm


def _configure_memory_auth(monkeypatch) -> None:
    memory_auth.configured_actor_tokens.cache_clear()
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_TOKENS_JSON",
        '{"subconscious-client":"secret-token"}',
    )
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_POLICIES_JSON",
        '{"subconscious-client":{"allowed_user_prefixes":["alice","bob"]}}',
    )


class TestConfig:
    """Test configuration."""

    def test_config_from_env(self):
        """Config can be created from environment."""
        config = SubconsciousConfig.from_env()
        assert config.enabled in (True, False)
        assert config.max_memories == 5

    def test_config_reads_shared_service_provider_from_env(self, monkeypatch):
        """Provider selection can be driven from environment."""
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_PROVIDER", "shared_service")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_BASE_URL", "http://memory.internal:5003")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_ACTOR_ID", "subconscious-client")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_ACTOR_SECRET", "secret-token")

        config = SubconsciousConfig.from_env()

        assert config.memory_provider == "shared_service"
        assert config.memory_service_base_url == "http://memory.internal:5003"
        assert config.memory_service_actor_id == "subconscious-client"

    def test_config_immutable(self):
        """Config is frozen."""
        config = SubconsciousConfig()
        with pytest.raises(Exception):
            config.enabled = False  # type: ignore

    def test_user_config_binding(self):
        """Config can be bound to user."""
        config = SubconsciousConfig()
        user_config = config.with_user("alice")
        assert user_config.user_id == "alice"
        assert user_config.base == config


class TestMockProvider:
    """Test mock memory provider."""

    @pytest.mark.asyncio
    async def test_store_and_recall(self):
        """Can store and recall memories."""
        provider = MockProvider()

        await provider.store("User prefers TypeScript", "alice", {})
        await provider.store("Project uses pnpm", "alice", {})

        memories = await provider.recall("preferences", "alice", limit=10)
        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_user_isolation(self):
        """Memories are isolated by user."""
        provider = MockProvider()

        await provider.store("Alice's preference", "alice", {})
        await provider.store("Bob's preference", "bob", {})

        alice_memories = await provider.recall("", "alice", limit=10)
        bob_memories = await provider.recall("", "bob", limit=10)

        assert len(alice_memories) == 1
        assert len(bob_memories) == 1
        assert "Alice" in alice_memories[0].content
        assert "Bob" in bob_memories[0].content

    @pytest.mark.asyncio
    async def test_delete(self):
        """Can delete memories."""
        provider = MockProvider()

        memory = await provider.store("test content", "alice", {})
        deleted = await provider.delete(memory.id, "alice")

        assert deleted is True

        memories = await provider.recall("", "alice", limit=10)
        assert len(memories) == 0


class TestReflectionHelpers:
    """Test v3 reflection parsing helpers."""

    @pytest.mark.asyncio
    async def test_extract_learnings_uses_generic_response_extractor(self):
        class _Message:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Response:
            def __init__(self, text: str) -> None:
                self.content = [_Message(text)]

        class _MessagesApi:
            async def create(self, **kwargs):
                _ = kwargs
                return _Response('["Keep answers concise"]')

        class _Client:
            messages = _MessagesApi()

        learnings = await extract_learnings_from_llm(
            api_key="test-key",
            base_url="http://example.test",
            model="test-model",
            conversation="user: hello",
            focus_prompt="extract",
            llm_client=_Client(),
        )

        assert learnings == ["Keep answers concise"]


class TestLocalHindsightProvider:
    """Test local hindsight provider helpers."""

    def test_build_fts_query_sanitizes_control_characters(self):
        provider = LocalHindsightProvider("pixelated")

        query = provider._store.build_fts_query('exact "phrase" OR tag* (weird)')

        assert query is not None
        assert '"' in query
        assert "*" not in query
        assert "(" not in query
        assert ")" not in query


class TestSharedMemoryServiceProvider:
    """Test shared HTTP memory provider."""

    @pytest.mark.asyncio
    async def test_store_recall_delete_against_memory_server(self, monkeypatch, tmp_path):
        _configure_memory_auth(monkeypatch)
        app = create_memory_server()
        app.state.memory_manager = LocalHindsightMemoryManager(
            db_path=str(tmp_path / "hindsight.db")
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            provider = SharedMemoryServiceProvider(
                base_url="http://testserver",
                bank_id="pixelated",
                actor_id="subconscious-client",
                actor_secret="secret-token",
                client=client,
            )

            stored = await provider.store(
                "Alice prefers direct summaries",
                "alice",
                {"project_id": "pixelated", "category": "preference"},
            )
            recalled = await provider.recall("direct summaries", "alice", limit=5)

            assert recalled[0].id == stored.id
            assert recalled[0].content == "Alice prefers direct summaries"

            deleted = await provider.delete(stored.id, "alice")
            assert deleted is True
            assert await provider.recall("direct summaries", "alice", limit=5) == []

    def test_create_memory_provider_builds_shared_service_provider(self, monkeypatch):
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_PROVIDER", "shared_service")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_BASE_URL", "http://memory.internal:5003")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_ACTOR_ID", "subconscious-client")
        monkeypatch.setenv("SUBCONSCIOUS_MEMORY_ACTOR_SECRET", "secret-token")

        provider = create_memory_provider(SubconsciousConfig.from_env())

        assert isinstance(provider, SharedMemoryServiceProvider)
        assert provider.base_url == "http://memory.internal:5003"


class TestSubconsciousState:
    """Test subconscious state."""

    @pytest.mark.asyncio
    async def test_enrich_disabled(self):
        """Returns original message when disabled."""
        config = SubconsciousConfig(enabled=False).with_user("alice")
        state = SubconsciousState(config)
        state._provider = MockProvider()

        result = await state.enrich("Hello world")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_enrich_no_memories(self):
        """Returns original message when no memories found."""
        config = SubconsciousConfig().with_user("alice")
        state = SubconsciousState(config)
        state._provider = MockProvider()

        result = await state.enrich("Hello world")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_enrich_with_memories(self):
        """Prepends memories when found."""
        config = SubconsciousConfig().with_user("alice")
        state = SubconsciousState(config)
        provider = MockProvider()
        await provider.store("User prefers dark mode", "alice", {})
        state._provider = provider

        result = await state.enrich("How do I configure the theme?")
        assert "<subconscious_context>" in result
        assert "dark mode" in result
        assert "How do I configure" in result

    @pytest.mark.asyncio
    async def test_record_and_reflect(self):
        """Records conversation and reflects."""
        config = SubconsciousConfig(
            api_key="",  # No API key = skip LLM
            reflect_on_close=False,
        ).with_user("alice")

        state = SubconsciousState(config)
        state._provider = MockProvider()

        state.record("user", "Hello")
        state.record("assistant", "Hi there")

        assert len(state._conversation_manager.conversation) == 2

    @pytest.mark.asyncio
    async def test_record_trims_conversation_history(self):
        """Conversation history is kept to a bounded sliding window."""
        config = SubconsciousConfig(
            api_key="",
            reflect_on_close=False,
        ).with_user("alice")

        state = SubconsciousState(config)
        state._provider = MockProvider()

        for index in range(120):
            state.record("user", f"message-{index}")

        assert len(state._conversation_manager.conversation) == 100
        assert state._conversation_manager.conversation[0]["content"] == "message-20"


class TestContextVars:
    """Test contextvars API."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Can set and get context."""
        config = SubconsciousConfig(enabled=False)
        token = set_subconscious(config, "alice")

        state = get_subconscious()
        assert state is not None
        assert state.config.user_id == "alice"

        await reset_subconscious(token)

        state = get_subconscious()
        assert state is None

    @pytest.mark.asyncio
    async def test_nested_contexts(self):
        """Nested contexts work correctly."""
        config1 = SubconsciousConfig(enabled=False)
        config2 = SubconsciousConfig(enabled=True)

        token1 = set_subconscious(config1, "alice")
        assert get_subconscious().config.user_id == "alice"

        token2 = set_subconscious(config2, "bob")
        assert get_subconscious().config.user_id == "bob"

        await reset_subconscious(token2)
        assert get_subconscious().config.user_id == "alice"

        await reset_subconscious(token1)
        assert get_subconscious() is None


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_flow(self):
        """Complete flow: set context, enrich, close."""
        config = SubconsciousConfig(
            api_key="",
            reflect_on_close=False,
        )

        token = set_subconscious(config, "alice")
        state = get_subconscious()

        # Should work
        result = await state.enrich("Test message")
        assert result == "Test message"

        state.record("user", "Test message")
        state.record("assistant", "Test response")

        assert len(state._conversation_manager.conversation) == 2

        await reset_subconscious(token)

        assert state._closed is True
        assert get_subconscious() is None
