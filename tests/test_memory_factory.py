import os
from unittest.mock import patch

from ai.api.memory.base import BaseMemoryManager
from ai.memory.hindsight_manager import HindsightMemoryManager
from ai.memory.manager_factory import MemoryManagerFactory
from ai.memory.mem0_gemini.manager import GeminiMem0Manager
from ai.memory.mem0_nvidia.manager import NvidiaMem0Manager


@patch("ai.memory.manager_factory.HindsightMemoryManager")
@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_creates_gemini_manager(mock_nvidia, mock_gemini, mock_hindsight):
    """Test that the factory creates a GeminiMem0Manager when configured."""
    with patch.dict(
        os.environ,
        {"MEM0_PROVIDER": "gemini", "GOOGLE_API_KEY": "test-key"},
        clear=True,
    ):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_gemini.return_value


@patch("ai.memory.manager_factory.HindsightMemoryManager")
@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_creates_nvidia_manager(mock_nvidia, mock_gemini, mock_hindsight):
    """Test that the factory creates an NvidiaMem0Manager when configured."""
    with patch.dict(
        os.environ,
        {
            "MEM0_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_USE_ENHANCED": "false",
        },
        clear=True,
    ):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_nvidia.return_value


@patch("ai.memory.manager_factory.HindsightMemoryManager")
@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_default_provider(mock_nvidia, mock_gemini, mock_hindsight):
    """Test that the factory defaults to gemini if no provider is specified."""
    # Ensure clear environment for testing defaults
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_gemini.return_value


@patch("ai.memory.manager_factory.HindsightMemoryManager")
@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_creates_hindsight_manager(mock_nvidia, mock_gemini, mock_hindsight):
    with patch.dict(
        os.environ,
        {"MEMORY_PROVIDER": "hindsight", "HINDSIGHT_API_KEY": "test-key"},
        clear=True,
    ):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_hindsight.return_value
        mock_hindsight.assert_called_once()


def test_factory_enhanced_nvidia_path_returns_memory_manager_compatible_object(
    monkeypatch,
):
    """Enhanced NVIDIA mode must still satisfy the BaseMemoryManager contract."""

    class FakeEnhancedManager:
        def __init__(self, config):
            self.config = config
            self.sync_client = object()
            self.client = object()

        async def generate(self, prompt: str, **kwargs):
            return f"analysis:{prompt}"

    class FakeMemoryManager(BaseMemoryManager):
        def add_memory(self, content, user_id, metadata=None, category=None):
            return "mem-1"

        def search_memories(self, query, user_id):
            return [{"memory": "stored memory", "score": 1.0}]

        def get_all_memories(self, user_id):
            return [{"memory": "stored memory", "metadata": {"category": "fact"}}]

        def get_memory(self, memory_id):
            return {"id": memory_id}

        def update_memory(self, memory_id, new_content, metadata=None):
            return True

        def delete_memory(self, memory_id):
            return True

        def clear_memory(self, user_id):
            return True

    monkeypatch.setattr(
        "ai.memory.manager_factory.EnhancedNvidiaNimManager", FakeEnhancedManager
    )
    monkeypatch.setattr(
        "ai.memory.manager_factory.NvidiaMem0Manager",
        lambda config: FakeMemoryManager(),
    )

    with patch.dict(
        os.environ,
        {
            "MEM0_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_USE_ENHANCED": "true",
        },
        clear=True,
    ):
        manager = MemoryManagerFactory.create_manager()

    assert isinstance(manager, BaseMemoryManager)
    assert manager.search_memories("stored", "vivi")[0]["memory"] == "stored memory"
    assert hasattr(manager, "generate")
