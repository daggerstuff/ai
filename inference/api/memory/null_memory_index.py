from __future__ import annotations

import threading


class NullMemoryIndex:
    """Thread-safe record ownership and ID registry for the null backend."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def generate_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"mem-{self._counter}"

    def owner_for(self, memory_id: str) -> str | None:
        with self._lock:
            return self._owners.get(memory_id)

    def set_owner(self, *, memory_id: str, user_id: str) -> None:
        with self._lock:
            self._owners[memory_id] = user_id

    def remove_owner(self, *, memory_id: str) -> None:
        with self._lock:
            self._owners.pop(memory_id, None)
