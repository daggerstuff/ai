from __future__ import annotations

import threading


class NullMemoryCategoryTracker:
    """Maintains per-user category tallies for the null backend."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def get_counts(self, *, user_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self._counts.get(user_id, {}))

    def apply(
        self,
        *,
        user_id: str,
        previous_category: str | None,
        current_category: str | None,
    ) -> None:
        with self._lock:
            counts = self._counts.setdefault(user_id, {})
            if previous_category is not None:
                remaining = counts.get(previous_category, 0) - 1
                if remaining > 0:
                    counts[previous_category] = remaining
                else:
                    counts.pop(previous_category, None)
            if current_category is not None:
                counts[current_category] = counts.get(current_category, 0) + 1
            if not counts:
                self._counts.pop(user_id, None)

    def clear_user(self, *, user_id: str) -> None:
        with self._lock:
            self._counts.pop(user_id, None)
