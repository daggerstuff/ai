import logging
import os
from typing import Optional

from ai.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager
from ai.memory.mem0_nvidia.manager import NvidiaMem0Config, NvidiaMem0Manager
from ai.api.memory.base import BaseMemoryManager

logger = logging.getLogger("MemoryManagerFactory")


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Supports NVIDIA NIM and Google Gemini providers.
    """

    @staticmethod
    def create_manager() -> BaseMemoryManager:
        """
        Creates a memory manager based on environment configuration.
        
        Logic:
        1. Check MEM0_PROVIDER env var ('nvidia' or 'gemini').
        2. If not set, check for NVIDIA_API_KEY (defaults to nvidia).
        3. If not set, check for GEMINI_API_KEY (fallback to gemini).
        4. If nothing found, return NullMemoryManager.
        """
        provider = os.environ.get("MEM0_PROVIDER", "").lower()

        # 1. Force Provider if specified
        if provider == "nvidia":
            return MemoryManagerFactory._create_nvidia_manager()
        elif provider == "gemini":
            return MemoryManagerFactory._create_gemini_manager()

        # 2. Autodetect
        if os.environ.get("NVIDIA_API_KEY"):
            return MemoryManagerFactory._create_nvidia_manager()
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return MemoryManagerFactory._create_gemini_manager()

        # 3. Fallback to Null
        logger.warning("No memory provider configured. Using NullMemoryManager.")
        from ai.api.memory.null_memory import NullMemoryManager
        return NullMemoryManager()

    @staticmethod
    def _create_nvidia_manager() -> BaseMemoryManager:
        """Helper to create NVIDIA manager."""
        try:
            config = NvidiaMem0Config(
                nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
                mem0_api_key=os.environ.get("MEM0_API_KEY"),
                model_name=os.environ.get(
                    "NVIDIA_MODEL_NAME", "meta/llama-3.1-405b-instruct"
                ),
            )
            return NvidiaMem0Manager(config)
        except Exception as e:
            logger.error(f"Failed to initialize NvidiaMem0Manager: {e}")
            raise

    @staticmethod
    def _create_gemini_manager() -> BaseMemoryManager:
        """Helper to create Gemini manager."""
        try:
            config = GeminiMem0Config(
                gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                mem0_api_key=os.environ.get("MEM0_API_KEY"),
                model_name=os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro"),
            )
            return GeminiMem0Manager(config)
        except Exception as e:
            logger.error(f"Failed to initialize GeminiMem0Manager: {e}")
            raise


def get_memory_manager() -> BaseMemoryManager:
    """Legacy wrapper for create_manager."""
    return MemoryManagerFactory.create_manager()
