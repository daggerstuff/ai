from __future__ import annotations

import threading


class NullMemoryCoordination:
    """Owns per-user locks and revision signals for the null backend."""

    def __init__(self) -> None:
        self._revision = 0
        self._user_revisions: dict[str, int] = {}
        self._index_lock = threading.Lock()
        self._user_locks: dict[str, threading.Lock] = {}

    def user_lock(self, user_id: str) -> threading.Lock:
        with self._index_lock:
            lock = self._user_locks.get(user_id)
            if lock is None:
                lock = threading.Lock()
                self._user_locks[user_id] = lock
            return lock

    @property
    def revision(self) -> int:
        with self._index_lock:
            return self._revision

    def user_revision(self, user_id: str) -> int:
        with self._index_lock:
            return self._user_revisions.get(user_id, 0)

    def touch(self, *, user_id: str) -> None:
        with self._index_lock:
            self._revision += 1
            self._user_revisions[user_id] = self._user_revisions.get(user_id, 0) + 1

    def drop_user_lock(self, user_id: str) -> None:
        with self._index_lock:
            self._user_locks.pop(user_id, None)
