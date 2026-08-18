from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.memory.mem0_nvidia.manager import NvidiaMem0Config, NvidiaMem0Manager


@pytest.fixture
def mock_openai():
    with patch("ai.memory.mem0_nvidia.manager.OpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_async_openai():
    with patch("ai.memory.mem0_nvidia.manager.AsyncOpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_mem0_client():
    with patch("ai.memory.mem0_nvidia.manager.MemoryClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


def test_nvidia_manager_initialization(mock_openai, mock_async_openai, _mock_mem0_client):
    config = NvidiaMem0Config(nvidia_api_key="test-nv", mem0_api_key="test-mem0")
    # Patch TherapeuticProcessor to avoid real sub-calls if needed
    with patch("ai.memory.therapeutic_processor.TherapeuticProcessor"):
        manager = NvidiaMem0Manager(config)
        assert manager.processor is not None
        assert mock_openai.called, "OpenAI client was not initialized"
        assert mock_async_openai.called, "AsyncOpenAI client was not initialized"


@pytest.mark.asyncio
async def test_nvidia_manager_search(_mock_openai, _mock_async_openai, mock_mem0_client):
    config = NvidiaMem0Config(nvidia_api_key="test-nv", mem0_api_key="test-mem0")
    manager = NvidiaMem0Manager(config, memory_provider=mock_mem0_client)

    # Mock search result
    mock_mem0_client.search.return_value = {"results": [{"memory": "Fact 1", "id": "m1"}]}

    results = manager.search_memories("query", user_id="u1")
    assert len(results) == 1
    assert results[0]["memory"] == "Fact 1"
    mock_mem0_client.search.assert_called_with("query", user_id="u1", limit=10)


@pytest.mark.asyncio
async def test_nvidia_manager_add(_mock_openai, _mock_async_openai, mock_mem0_client):
    config = NvidiaMem0Config(nvidia_api_key="test-nv", mem0_api_key="test-mem0")
    manager = NvidiaMem0Manager(config, memory_provider=mock_mem0_client)

    # Mock add result
    mock_mem0_client.add.return_value = {"results": [{"id": "m2"}]}

    memory_id = manager.add_memory("I feel happy", user_id="u1", category="EMOTIONAL_STATE")
    assert memory_id == "m2"

    # Verify add was called with metadata
    args, kwargs = mock_mem0_client.add.call_args
    assert "I feel happy" in args[0]
    assert kwargs["user_id"] == "u1"
    assert kwargs["metadata"]["category"] == "EMOTIONAL_STATE"


@pytest.mark.asyncio
async def test_nvidia_manager_get_response(_mock_openai, mock_async_openai, mock_mem0_client):
    config = NvidiaMem0Config(nvidia_api_key="test-nv", mem0_api_key="test-mem0")
    manager = NvidiaMem0Manager(config, memory_provider=mock_mem0_client)

    # Mock search
    mock_mem0_client.search.return_value = {"results": []}

    # Mock NVIDIA Chat Completion - properly mocked for AsyncOpenAI
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I hear you."

    # Correct way to mock AsyncOpenAI's chat.completions.create
    mock_async_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

    # We need to re-initialize manager or re-inject client because the previous
    # mock_async_openai.called check might have used a different instance if
    # we weren't careful.
    # Actually, NvidiaMem0Manager creates self.async_client = AsyncOpenAI(...)
    # So it calls the mock class.

    response = await manager.get_response(user_id="u1", message="Hello")
    assert response == "I hear you."
    assert mock_async_openai.return_value.chat.completions.create.called
