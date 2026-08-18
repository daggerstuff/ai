from __future__ import annotations

import threading


class NullMemoryCapacityTracker:
    """Maintains O(1) capacity-pressure state for the null backend."""

    def __init__(self) -> None:
        self._at_capacity_users: set[str] = set()
        self._lock = threading.Lock()

    def has_pressure(self) -> bool:
        with self._lock:
            return bool(self._at_capacity_users)

    def user_has_pressure(self, *, user_id: str) -> bool:
        with self._lock:
            return user_id in self._at_capacity_users

    def set_user_count(self, *, user_id: str, count: int, max_memories: int) -> None:
        with self._lock:
            if count >= max_memories:
                self._at_capacity_users.add(user_id)
            else:
                self._at_capacity_users.discard(user_id)

    def clear_user(self, *, user_id: str) -> None:
        with self._lock:
            self._at_capacity_users.discard(user_id)
