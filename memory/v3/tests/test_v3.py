"""
Unit tests for Claude Subconscious v3.

Tests core functionality without requiring external dependencies.
"""

import pytest

from ai.memory.v3.config import SubconsciousConfig
from ai.memory.v3.context import (
    SubconsciousState,
    get_subconscious,
    reset_subconscious,
    set_subconscious,
)
from ai.memory.v3.provider import MockProvider


class TestConfig:
    """Test configuration."""

    def test_config_from_env(self):
        """Config can be created from environment."""
        config = SubconsciousConfig.from_env()
        assert config.enabled in (True, False)
        assert config.max_memories == 5

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
        deleted = await provider.delete(memory.id)

        assert deleted is True

        memories = await provider.recall("", "alice", limit=10)
        assert len(memories) == 0


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

        assert len(state._conversation) == 2


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

        assert len(state._conversation) == 2

        await reset_subconscious(token)

        assert state._closed is True
        assert get_subconscious() is None
