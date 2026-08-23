from __future__ import annotations

from .null_memory_repository import NullMemoryRepository


class NullMemoryHealthService:
    """Health reporting for the null memory backend."""

    def __init__(self, store: NullMemoryRepository) -> None:
        self.store = store

    def status(self, *, user_id: str | None = None) -> dict[str, str]:
        if user_id is None:
            status = "degraded" if self.store.has_capacity_pressure() else "healthy"
        else:
            status = "degraded" if self.store.has_capacity_pressure_for_user(user_id=user_id) else "healthy"
        return {"status": status, "provider": "NullMemoryManager"}
