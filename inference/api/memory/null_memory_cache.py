from __future__ import annotations

from collections import OrderedDict
from typing import Any


class NullMemoryCategoryCountCache:
    """Small cache for scoped null-memory category counts."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], dict[str, int]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> dict[str, int] | None:
        cached = self._entries.get(key)
        if cached is None:
            return None
        self._entries.move_to_end(key)
        return dict(cached)

    def put(self, key: tuple[Any, ...], *, value: dict[str, int]) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = dict(value)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
