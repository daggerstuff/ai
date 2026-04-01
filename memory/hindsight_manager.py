"""
Compatibility shim for older code paths that still import HindsightMemoryManager.

The shared memory architecture is now local-only. This class preserves the old
import path while delegating directly to the local Hindsight-compatible manager.
"""

from __future__ import annotations

from typing import Optional

from .local_hindsight_manager import LocalHindsightMemoryManager


class HindsightMemoryManager(LocalHindsightMemoryManager):
    """Backward-compatible alias for the local shared memory manager."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        bank_id: Optional[str] = None,
        timeout: float = 30.0,
        session: object | None = None,
        db_path: Optional[str] = None,
    ) -> None:
        del api_key, api_url, timeout, session
        super().__init__(db_path=db_path, bank_id=bank_id)
