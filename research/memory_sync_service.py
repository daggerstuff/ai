"""
Compatibility memory sync service for legacy unified-memory tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SyncDirection(StrEnum):
    FORESIGHT_TO_LETTA = "foresight_to_letta"
    LETTA_TO_FORESIGHT = "letta_to_foresight"


@dataclass
class SyncResult:
    """Result payload for a sync operation."""

    foresight_to_foresight: int = 0
    foresight_to_letta: int = 0
    letta_to_foresight: int = 0
    conflicts_resolved: int = 0
    errors: int = 0


class MemorySyncService:
    """Minimal compatibility sync service."""

    def __init__(self, foresight: object, letta: object, config: dict | None = None):
        self.foresight = foresight
        self.letta = letta
        self.sync_interval = int((config or {}).get("sync_interval", 300))
        self.auto_sync = bool((config or {}).get("auto_sync", False))

    async def sync_now(self, direction: SyncDirection) -> SyncResult:
        if direction == SyncDirection.FORESIGHT_TO_LETTA:
            return SyncResult(
                foresight_to_foresight=0, foresight_to_letta=0, letta_to_foresight=0, conflicts_resolved=0, errors=0
            )
        if direction == SyncDirection.LETTA_TO_FORESIGHT:
            return SyncResult(
                foresight_to_foresight=0, foresight_to_letta=0, letta_to_foresight=0, conflicts_resolved=0, errors=0
            )
        return SyncResult(
            foresight_to_foresight=0,
            foresight_to_letta=0,
            letta_to_foresight=0,
            conflicts_resolved=0,
            errors=0,
        )


__all__ = ["MemorySyncService", "SyncDirection", "SyncResult"]
