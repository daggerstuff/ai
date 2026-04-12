import logging

from .base import BaseMemoryManager
from .local_hindsight_manager import LocalHindsightMemoryManager
from .local_memory_settings import resolve_local_memory_settings, resolve_memory_provider

logger = logging.getLogger("MemoryManagerFactory")


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Local Hindsight is the only supported backend in this repo path.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        local_manager_class: type[BaseMemoryManager] = LocalHindsightMemoryManager,
    ) -> None:
        self.provider = provider
        self.local_manager_class = local_manager_class

    def create_manager(self, provider: str | None = None) -> BaseMemoryManager:
        """
        Creates a memory manager based on environment configuration.

        Logic:
        1. Use the explicit provider argument when present.
        2. Enforce one canonical provider name for the shared local service.
        3. If no supported provider is configured, fail closed with a configuration error.
        """
        resolve_memory_provider(provider or self.provider)
        return self._create_local_hindsight_manager()

    def _create_local_hindsight_manager(self) -> BaseMemoryManager:
        """Helper to create local persistent Hindsight-compatible manager."""
        try:
            settings = resolve_local_memory_settings()
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
