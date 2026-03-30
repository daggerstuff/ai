import logging
import os
from typing import Any, Dict, Optional

from .base import BaseMemoryManager
from .null_memory import NullMemoryManager

logger = logging.getLogger("MemoryManagerFactory")


def _get_hindsight_manager_class():
    """Lazy import to avoid circular dependency."""
    from .hindsight_manager import HindsightMemoryManager
    return HindsightMemoryManager


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Supports Hindsight as the primary memory backend.
    """

    @staticmethod
    def create_manager() -> BaseMemoryManager:
        """
        Creates a memory manager based on environment configuration.

        Logic:
        1. Check MEMORY_PROVIDER env var ('hindsight').
        2. If not set, check for HINDSIGHT_API_KEY (defaults to hindsight).
        3. If nothing found, return NullMemoryManager.
        """
        provider = (
            os.environ.get("MEMORY_PROVIDER") or ""
        ).lower()

        # 1. Force Provider if specified
        if provider == "hindsight" or provider == "":
            return MemoryManagerFactory._create_hindsight_manager()

        # 2. Autodetect
        if os.environ.get("HINDSIGHT_API_KEY"):
            return MemoryManagerFactory._create_hindsight_manager()

        # 3. Fallback to Null
        logger.warning("No memory provider configured. Using NullMemoryManager.")
        return NullMemoryManager()

    @staticmethod
    def _create_hindsight_manager() -> BaseMemoryManager:
        """Helper to create Hindsight manager."""
        try:
            logger.info("Using HindsightMemoryManager")
            HindsightMemoryManagerClass = _get_hindsight_manager_class()
            return HindsightMemoryManagerClass()
        except Exception as e:
            logger.error(f"Failed to initialize HindsightMemoryManager: {e}")
            raise


def get_memory_manager() -> BaseMemoryManager:
    """Legacy wrapper for create_manager."""
    return MemoryManagerFactory.create_manager()
