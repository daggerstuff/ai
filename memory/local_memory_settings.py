from __future__ import annotations

import os
from dataclasses import dataclass

CANONICAL_MEMORY_PROVIDER = "local_foresight"


@dataclass(frozen=True)
class LocalMemorySettings:
    db_path: str
    bank_id: str


def resolve_memory_provider(provider: str | None = None) -> str:
    """Resolve the configured memory backend and enforce one canonical provider name."""
    value = (provider or os.environ.get("MEMORY_PROVIDER") or "").strip().lower()
    if value != CANONICAL_MEMORY_PROVIDER:
        raise RuntimeError(
            "No supported memory provider configured. "
            f"Set MEMORY_PROVIDER={CANONICAL_MEMORY_PROVIDER} to run the shared local memory service."
        )
    return CANONICAL_MEMORY_PROVIDER


def resolve_local_memory_settings(
    *,
    db_path: str | None = None,
    bank_id: str | None = None,
) -> LocalMemorySettings:
    resolved_db_path = db_path or os.environ.get("FORESIGHT_LOCAL_DB_PATH") or os.environ.get("HINDSIGHT_LOCAL_DB_PATH")
    if not resolved_db_path:
        raise RuntimeError("FORESIGHT_LOCAL_DB_PATH or db_path is required")
    return LocalMemorySettings(
        db_path=resolved_db_path,
        bank_id=bank_id or os.environ.get("FORESIGHT_BANK_ID") or os.environ.get("HINDSIGHT_BANK_ID") or "pixelated",
    )
