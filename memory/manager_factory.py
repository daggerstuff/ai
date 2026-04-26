import logging

from .base import BaseMemoryManager
from .local_memory_settings import resolve_local_memory_settings, resolve_memory_provider

logger = logging.getLogger("MemoryManagerFactory")


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Local Foresight is the only supported backend in this repo path.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        local_manager_class: type[BaseMemoryManager] | None = None,
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
        return self._create_local_foresight_manager()

    def _create_local_foresight_manager(self) -> BaseMemoryManager:
        """Helper to create local persistent Foresight-compatible manager."""
        try:
            settings = resolve_local_memory_settings()
            logger.info("Using LocalForesightMemoryManager")
            
            # For now, we use the local_manager_class if provided, 
            # otherwise we'll need to import the new Foresight manager.
            # Since we are removing Foresight, we should transition to the Foresight implementation.
            if self.local_manager_class:
                return self.local_manager_class(
                    db_path=settings.db_path,
                    bank_id=settings.bank_id,
                )
            
            # Lazy import to avoid circular dependency and handle migration
            from .local_foresight_manager import LocalForesightMemoryManager
            return LocalForesightMemoryManager(
                db_path=settings.db_path,
                bank_id=settings.bank_id,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LocalForesightMemoryManager: {e}")
            raise


def get_required_memory_manager() -> BaseMemoryManager:
    """Return the configured shared memory manager or fail closed."""
    return MemoryManagerFactory().create_manager()
