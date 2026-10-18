from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryHealthProvider(Protocol):
    def get_health_status(self) -> dict[str, object]: ...


def resolve_memory_readiness(manager) -> str | None:
    if not isinstance(manager, MemoryHealthProvider):
        return None
    status = manager.get_health_status()
    if not isinstance(status, dict):
        return None
    raw = status.get("status")
    return str(raw) if raw is not None else None


def resolve_memory_health(*, readiness: str | None, _memory_count: int) -> str:
    normalized_readiness = (readiness or "unknown").lower()

    if normalized_readiness in {"degraded", "unhealthy", "error"}:
        return "Degraded"
    if normalized_readiness in {"healthy", "ok", "ready"}:
        return "Healthy"
    if normalized_readiness in {"unknown", "initializing", "starting"}:
        return "Initializing"
    return "Initializing"
