import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from rag.nemotron_rag import NemotronRAGConfig


def test_nemotron_rag_config_env_var():
    # Setup env var
    os.environ["NVIDIA_API_KEY"] = "env-api-key"

    # Initialize without args
    config = NemotronRAGConfig()

    # Assert
    assert config.api_key == "env-api-key", "API key not loaded from environment correctly"

    # Clean up
    del os.environ["NVIDIA_API_KEY"]
