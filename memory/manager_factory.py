import logging
import os
from typing import Optional, Type

from .base import BaseMemoryManager
from .local_hindsight_manager import LocalHindsightMemoryManager
from .local_memory_settings import resolve_local_memory_settings

logger = logging.getLogger("MemoryManagerFactory")


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Local Hindsight is the only supported backend in this repo path.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        local_manager_class: Type[BaseMemoryManager] = LocalHindsightMemoryManager,
    ) -> None:
        self.provider = provider
        self.local_manager_class = local_manager_class

    def create_manager(self, provider: Optional[str] = None) -> BaseMemoryManager:
        """
        Creates a memory manager based on environment configuration.

        Logic:
        1. Use the explicit provider argument when present.
        2. Treat local and hindsight provider values as aliases for the shared local service.
        3. If no supported provider is configured, fail closed with a configuration error.
        """
        provider_name = (
            provider or self.provider or os.environ.get("MEMORY_PROVIDER") or ""
        ).lower().strip()

        if provider_name in {"local_hindsight", "local-hindsight", "local", "hindsight"}:
            return self._create_local_hindsight_manager()

        raise RuntimeError(
            "No supported memory provider configured. "
            "Set MEMORY_PROVIDER=local_hindsight to run the shared local memory service."
        )

    def _create_local_hindsight_manager(self) -> BaseMemoryManager:
        """Helper to create local persistent Hindsight-compatible manager."""
        try:
            settings = resolve_local_memory_settings(
                db_path=os.environ.get("HINDSIGHT_LOCAL_DB_PATH"),
                bank_id=os.environ.get("HINDSIGHT_BANK_ID"),
            )
            logger.info("Using LocalHindsightMemoryManager")
            return self.local_manager_class(
                db_path=settings.db_path,
                bank_id=settings.bank_id,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LocalHindsightMemoryManager: {e}")
            raise
def get_required_memory_manager() -> BaseMemoryManager:
    """Return the configured shared memory manager or fail closed."""
    return MemoryManagerFactory().create_manager()
