"""Compatibility shim for older code paths that still import HindsightMemoryManager.

The shared memory architecture is now local-only. This class preserves the old
import path while delegating directly to the local Hindsight-compatible manager.
"""
from __future__ import annotations

from .local_hindsight_manager import LocalHindsightMemoryManager


class HindsightMemoryManager(LocalHindsightMemoryManager):
    """Backward-compatible alias for the local shared memory manager."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        bank_id: str | None = None,
        timeout: float = 30.0,
        session: object | None = None,
        db_path: str | None = None,
    ) -> None:
        del api_key, api_url, timeout, session
        super().__init__(db_path=db_path, bank_id=bank_id)
