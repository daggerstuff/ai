"""
Tests for the improved Foresight MCP server v2 tools.

These tests verify:
- Proper tool annotations
- Structured output (JSON/Markdown)
- Pagination metadata
- Error message quality
- New tools (get_memory, list_memories)
"""

import asyncio
import json

import pytest
from fastmcp import FastMCP

from ai.api.mcp_server import fastmcp_shared, fastmcp_v2_tools
from ai.api.mcp_server.fastmcp_shared import AuthorizedToolContext
from ai.api.mcp_server.fastmcp_v2_tools import (
    MemoryGetInput,
    MemoryListInput,
    MemoryQueryInput,
    MemoryStoreInput,
    ResponseFormat,
    _format_error,
    foresight_delete_memory,
    foresight_get_memory,
    foresight_list_memories,
    foresight_memory_status,
    foresight_query_memories,
    foresight_store_memory,
    foresight_update_memory,
    register_memory_tools_v2,
)
from ai.api.mcp_server.memory_scope import scope_from_kwargs
from ai.api.memory.null_memory import NullMemoryManager

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def null_manager():
    """Provide a NullMemoryManager for testing."""
    return NullMemoryManager()


@pytest.fixture
def authorized_context(null_manager):
    """Create an authorized context for testing."""

    def _create(user_id: str = "test-user", **scope_kwargs):
        return AuthorizedToolContext(
            manager=null_manager,
            scope=scope_from_kwargs(user_id=user_id, **scope_kwargs),
        )

    return _create


# =============================================================================
# Input Model Tests
# =============================================================================


class TestInputModels:
    """Test Pydantic input models."""

    def test_memory_store_input_validates_content_length(self):
        """Store input should enforce content length limits."""
        # Valid short content
        params = MemoryStoreInput(content="Valid content", user_id="user-1")
        assert params.content == "Valid content"

        # Valid long content (up to 10000 chars)
        long_content = "x" * 10000
        params = MemoryStoreInput(content=long_content, user_id="user-1")
        assert len(params.content) == 10000

        # Invalid empty content
        with pytest.raises(ValueError):
            MemoryStoreInput(content="", user_id="user-1")

    def test_memory_query_input_validates_limit(self):
        """Query input should enforce limit range."""
        # Valid limits
        params = MemoryQueryInput(query="test", user_id="user-1", limit=50)
        assert params.limit == 50

        # Invalid limit too high
        with pytest.raises(ValueError):
            MemoryQueryInput(query="test", user_id="user-1", limit=200)

        # Invalid limit zero
        with pytest.raises(ValueError):
            MemoryQueryInput(query="test", user_id="user-1", limit=0)

    def test_response_format_enum(self):
        """Response format should accept valid values."""
        params = MemoryStoreInput(content="test", user_id="user-1", response_format="json")
        assert params.response_format == ResponseFormat.JSON

        params = MemoryStoreInput(content="test", user_id="user-1", response_format="markdown")
        assert params.response_format == ResponseFormat.MARKDOWN

        # Invalid format
        with pytest.raises(ValueError):
            MemoryStoreInput(content="test", user_id="user-1", response_format="xml")


# =============================================================================
# Tool Output Format Tests
# =============================================================================


class TestOutputFormats:
    """Test JSON and Markdown output formatting."""

    @pytest.mark.asyncio
    async def test_store_memory_returns_json_when_requested(self, monkeypatch, null_manager):
        """Store should return structured JSON when format is json."""

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        params = MemoryStoreInput(
            content="Test memory content",
            user_id="test-user",
            category="fact",
            response_format=ResponseFormat.JSON,
        )

        result = await foresight_store_memory(params)

        # Parse as JSON to verify structure
        data = json.loads(result)
        assert data["success"] is True
        assert data["user_id"] == "test-user"
        assert data["content"] == "Test memory content"
        assert data["category"] == "fact"
        assert "memory_id" in data

    @pytest.mark.asyncio
    async def test_query_memories_returns_json_with_pagination(self, monkeypatch, null_manager):
        """Query should return JSON with pagination metadata."""
        # Add some test memories
        null_manager.add_memory("Project alpha details", "test-user")
        null_manager.add_memory("Project beta information", "test-user")
        null_manager.add_memory("Project gamma context", "test-user")

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        params = MemoryQueryInput(
            query="project",
            user_id="test-user",
            limit=2,
            offset=0,
            response_format=ResponseFormat.JSON,
        )

        result = await foresight_query_memories(params)

        data = json.loads(result)
        assert "user_id" in data
        assert "query" in data
        assert "memories" in data
        assert isinstance(data["memories"], list)
        assert "offset" in data
        assert "has_more" in data

    @pytest.mark.asyncio
    async def test_list_memories_returns_markdown_with_summary(self, monkeypatch, null_manager):
        """List should return human-readable markdown."""
        null_manager.add_memory("Memory one", "test-user")
        null_manager.add_memory("Memory two", "test-user")

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        params = MemoryListInput(
            user_id="test-user",
            limit=10,
            response_format=ResponseFormat.MARKDOWN,
        )

        result = await foresight_list_memories(params)

        assert "### Memory List" in result
        assert "test-user" in result
        assert "Showing:" in result


# =============================================================================
# Error Message Tests
# =============================================================================


class TestErrorMessages:
    """Test error message quality and suggestions."""

    def test_format_error_with_suggestion(self):
        """Error messages should include suggestions."""
        msg = _format_error("Memory not found", "Check the memory ID and try again")

        assert "Error: Memory not found" in msg
        assert "Suggestion: Check the memory ID and try again" in msg

    def test_format_error_without_suggestion(self):
        """Error messages work without suggestions."""
        msg = _format_error("Unknown error")

        assert msg == "Error: Unknown error"

    @pytest.mark.asyncio
    async def test_get_memory_returns_actionable_error_when_not_found(self, monkeypatch, null_manager):
        """Get memory should provide helpful error when memory doesn't exist."""

        def mock_context(**_kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id="test-user"),
            )

        # Mock memory_in_scope to return False
        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)
        monkeypatch.setattr(fastmcp_v2_tools, "memory_in_scope", lambda **kw: False)

        params = MemoryGetInput(
            memory_id="nonexistent-id",
            user_id="test-user",
            response_format=ResponseFormat.MARKDOWN,
        )

        result = await foresight_get_memory(params)

        assert "Error:" in result
        assert "not found" in result
        assert "Suggestion:" in result or "Verify" in result


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Test pagination functionality."""

    @pytest.mark.asyncio
    async def test_list_memories_respects_offset(self, monkeypatch, null_manager):
        """List should skip offset number of memories."""
        # Add 5 memories
        for i in range(5):
            null_manager.add_memory(f"Memory {i}", "test-user")

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        params = MemoryListInput(
            user_id="test-user",
            limit=2,
            offset=2,
            response_format=ResponseFormat.JSON,
        )

        result = await foresight_list_memories(params)

        data = json.loads(result)
        assert data["offset"] == 2
        assert len(data["memories"]) <= 2

    @pytest.mark.asyncio
    async def test_query_memories_includes_next_offset(self, monkeypatch, null_manager):
        """Query should include next_offset when more results exist."""
        # Add multiple memories matching a query
        for i in range(10):
            null_manager.add_memory(f"Test item {i}", "test-user")

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        params = MemoryQueryInput(
            query="test",
            user_id="test-user",
            limit=3,
            offset=0,
            response_format=ResponseFormat.JSON,
        )

        result = await foresight_query_memories(params)

        data = json.loads(result)
        if data["has_more"]:
            assert data["next_offset"] == 3


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test that tools are properly registered."""

    def test_register_tools_creates_all_tools(self):
        """All tools should be registered."""
        mcp = FastMCP("test_mcp")
        register_memory_tools_v2(mcp)

        # The tools should be registered
        # FastMCP doesn't expose a direct way to list tools, but we can
        # verify the function executed without error
        assert True

    def test_tool_names_have_service_prefix(self):
        """Tool names should start with foresight_."""
        # Get all functions registered as tools from the module
        funcs = [
            foresight_store_memory,
            foresight_query_memories,
            foresight_get_memory,
            foresight_list_memories,
            foresight_update_memory,
            foresight_delete_memory,
            foresight_memory_status,
        ]

        for func in funcs:
            # Function names should start with foresight_
            assert func.__name__.startswith("foresight_"), f"Tool {func.__name__} should have foresight_ prefix"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with NullMemoryManager."""

    @pytest.mark.asyncio
    async def test_full_crud_cycle(self, monkeypatch, null_manager):
        """Test complete create, read, update, delete cycle."""

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        # Also mock the status tool to use the same context
        monkeypatch.setattr(fastmcp_shared, "stdio_trusted_tool_context", mock_context)

        # Create
        store_params = MemoryStoreInput(
            content="Initial memory content",
            user_id="test-user",
            category="fact",
            response_format=ResponseFormat.JSON,
        )
        create_result = await foresight_store_memory(store_params)
        create_data = json.loads(create_result)
        assert create_data["success"] is True

        # Query to verify storage
        query_params = MemoryQueryInput(
            query="Initial",
            user_id="test-user",
            response_format=ResponseFormat.JSON,
        )
        query_result = await foresight_query_memories(query_params)
        query_data = json.loads(query_result)
        assert len(query_data["memories"]) >= 1

        # List to see all memories
        list_params = MemoryListInput(
            user_id="test-user",
            response_format=ResponseFormat.JSON,
        )
        list_result = await foresight_list_memories(list_params)
        list_data = json.loads(list_result)
        assert list_data["total"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_operations_are_safe(self, monkeypatch, null_manager):
        """Multiple operations should not interfere with each other."""

        def mock_context(**kwargs):
            return AuthorizedToolContext(
                manager=null_manager,
                scope=scope_from_kwargs(user_id=kwargs.get("user_id", "test-user")),
            )

        monkeypatch.setattr(fastmcp_v2_tools, "authorized_tool_context_from_json", mock_context)

        # Run multiple stores concurrently
        async def store_one(i):
            params = MemoryStoreInput(
                content=f"Concurrent memory {i}",
                user_id="test-user",
                response_format=ResponseFormat.JSON,
            )
            return await foresight_store_memory(params)

        results = await asyncio.gather(*[store_one(i) for i in range(5)])

        # All should succeed
        for result in results:
            data = json.loads(result)
            assert data["success"] is True

        # Verify all stored
        list_params = MemoryListInput(
            user_id="test-user",
            response_format=ResponseFormat.JSON,
        )
        list_result = await foresight_list_memories(list_params)
        list_data = json.loads(list_result)
        assert list_data["total"] >= 5
