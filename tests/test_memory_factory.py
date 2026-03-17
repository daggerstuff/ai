import os
from unittest.mock import patch

from ai.memory.manager_factory import MemoryManagerFactory
from ai.memory.mem0_gemini.manager import GeminiMem0Manager
from ai.memory.mem0_nvidia.manager import NvidiaMem0Manager


@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_creates_gemini_manager(mock_nvidia, mock_gemini):
    """Test that the factory creates a GeminiMem0Manager when configured."""
    with patch.dict(
        os.environ, {"MEM0_PROVIDER": "gemini", "GOOGLE_API_KEY": "test-key"}
    ):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_gemini.return_value


@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_creates_nvidia_manager(mock_nvidia, mock_gemini):
    """Test that the factory creates an NvidiaMem0Manager when configured."""
    with patch.dict(
        os.environ, {"MEM0_PROVIDER": "nvidia", "NVIDIA_API_KEY": "test-key"}
    ):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_nvidia.return_value


@patch("ai.memory.manager_factory.GeminiMem0Manager")
@patch("ai.memory.manager_factory.NvidiaMem0Manager")
def test_factory_default_provider(mock_nvidia, mock_gemini):
    """Test that the factory defaults to gemini if no provider is specified."""
    # Ensure clear environment for testing defaults
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
        manager = MemoryManagerFactory.create_manager()
        assert manager == mock_gemini.return_value
