from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LocalMemorySettings:
    db_path: str
    bank_id: str


def resolve_local_memory_settings(
    *,
    db_path: Optional[str] = None,
    bank_id: Optional[str] = None,
) -> LocalMemorySettings:
    resolved_db_path = db_path or os.environ.get("HINDSIGHT_LOCAL_DB_PATH")
    if not resolved_db_path:
        raise RuntimeError("HINDSIGHT_LOCAL_DB_PATH or db_path is required")
    return LocalMemorySettings(
        db_path=resolved_db_path,
        bank_id=bank_id or os.environ.get("HINDSIGHT_BANK_ID") or "pixelated",
    )
