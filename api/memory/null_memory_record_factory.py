from __future__ import annotations

import re
from collections.abc import Callable

from .null_memory_record import NullMemoryRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class NullMemoryRecordFactory:
    """Creates null-memory records without leaking schema rules into the facade."""

    def __init__(
        self,
        *,
        generate_id: Callable[[], str],
        now: Callable[[], str],
    ) -> None:
        self._generate_id = generate_id
        self._now = now

    def create(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> NullMemoryRecord:
        timestamp = self._now()
        return NullMemoryRecord.create(
            memory_id=memory_id or self._generate_id(),
            content=content,
            user_id=user_id,
            metadata=dict(metadata or {}),
            created_at=timestamp,
        )

    def create_replacement(
        self,
        *,
        existing_record: NullMemoryRecord,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> NullMemoryRecord:
        return NullMemoryRecord.create(
            memory_id=memory_id or existing_record.id,
            content=content,
            user_id=user_id,
            metadata=dict(metadata or {}),
            created_at=existing_record.created_at,
            updated_at=self._now(),
        )

    def now(self) -> str:
        return self._now()
