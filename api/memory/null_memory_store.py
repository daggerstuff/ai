from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.memory.hindsight_local_adapter import normalize_tags


class NullMemoryStore:
    """Thread-safe in-memory record store used by the null memory backend."""

    def __init__(self) -> None:
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_index: Dict[str, str] = {}
        self._memory_counter = 0
        self._revision = 0
        self._user_revisions: Dict[str, int] = {}
        self._index_lock = threading.Lock()
        self._user_locks: Dict[str, threading.Lock] = {}
        self.max_memories_per_user = 1000

    def _generate_id(self) -> str:
        self._memory_counter += 1
        return f"mem-{self._memory_counter}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _user_lock(self, user_id: str) -> threading.Lock:
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

    def add_record(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "id": memory_id or self._generate_id(),
            "content": content,
            "user_id": user_id,
            "metadata": dict(metadata or {}),
            "created_at": self._now(),
        }
        user_lock = self._user_lock(user_id)
        with user_lock:
            user_memories = self._memories.setdefault(user_id, [])
            existing_index = self._existing_record_index(
                user_memories=user_memories,
                memory_id=record["id"],
            )
            if existing_index is not None:
                self._replace_existing_record(
                    user_memories=user_memories,
                    existing_index=existing_index,
                    record=record,
                )
                return dict(record)
            self._ensure_capacity(user_memories=user_memories)
            user_memories.append(record)
        self._index_record(user_id=user_id, memory_id=record["id"])
        return dict(record)

    @staticmethod
    def _existing_record_index(
        *,
        user_memories: List[Dict[str, Any]],
        memory_id: str,
    ) -> Optional[int]:
        return next(
            (index for index, memory in enumerate(user_memories) if memory["id"] == memory_id),
            None,
        )

    def _replace_existing_record(
        self,
        *,
        user_memories: List[Dict[str, Any]],
        existing_index: int,
        record: Dict[str, Any],
    ) -> None:
        record["created_at"] = user_memories[existing_index]["created_at"]
        user_memories[existing_index] = record

    def _ensure_capacity(self, *, user_memories: List[Dict[str, Any]]) -> None:
        if len(user_memories) < self.max_memories_per_user:
            return
        evicted = user_memories.pop(0)
        with self._index_lock:
            self._memory_index.pop(evicted["id"], None)

    def _index_record(self, *, user_id: str, memory_id: str) -> None:
        with self._index_lock:
            self._memory_index[memory_id] = user_id
            self._revision += 1
            self._user_revisions[user_id] = self._user_revisions.get(user_id, 0) + 1

    def search_records(self, *, query: str, user_id: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        with self._user_lock(user_id):
            return [
                dict(memory)
                for memory in self._memories.get(user_id, [])
                if query_lower in memory["content"].lower()
            ]

    def recall_records(
        self,
        *,
        query: str,
        user_id: str,
        tags: Optional[List[str]],
        tags_match: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        requested_tags = normalize_tags(tags)
        matches: List[Dict[str, Any]] = []
        for memory in self.search_records(query=query, user_id=user_id):
            memory_tags = self._memory_tags(memory)
            if not self._matches_requested_tags(
                memory_tags=memory_tags,
                requested_tags=requested_tags,
                tags_match=tags_match,
            ):
                continue
            matches.append(self._protocol_record(memory=memory, memory_tags=memory_tags))
            if len(matches) >= limit:
                break
        return matches

    @staticmethod
    def _memory_tags(memory: Dict[str, Any]) -> List[str]:
        return normalize_tags((memory.get("metadata") or {}).get("tags", []))

    @staticmethod
    def _matches_requested_tags(
        *,
        memory_tags: List[str],
        requested_tags: List[str],
        tags_match: str,
    ) -> bool:
        if not requested_tags:
            return True
        if tags_match == "all":
            return all(tag in memory_tags for tag in requested_tags)
        return any(tag in memory_tags for tag in requested_tags)

    @staticmethod
    def _protocol_record(
        *,
        memory: Dict[str, Any],
        memory_tags: List[str],
    ) -> Dict[str, Any]:
        return {
            "document_id": memory["id"],
            "text": memory["content"],
            "tags": memory_tags,
        }

    def list_records(self, *, user_id: str) -> List[Dict[str, Any]]:
        with self._user_lock(user_id):
            return [dict(memory) for memory in self._memories.get(user_id, [])]

    def list_all_records(self) -> List[Dict[str, Any]]:
        with self._index_lock:
            user_ids = list(self._memories.keys())
        records: List[Dict[str, Any]] = []
        for user_id in user_ids:
            records.extend(self.list_records(user_id=user_id))
        return records

    def get_record(self, *, memory_id: str) -> Optional[Dict[str, Any]]:
        with self._index_lock:
            user_id = self._memory_index.get(memory_id)
        if user_id is None:
            return None
        with self._user_lock(user_id):
            for memory in self._memories.get(user_id, []):
                if memory["id"] == memory_id:
                    return dict(memory)
        return None

    def update_record(
        self,
        *,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._index_lock:
            user_id = self._memory_index.get(memory_id)
        if user_id is None:
            return False
        with self._user_lock(user_id):
            for memory in self._memories.get(user_id, []):
                if memory["id"] != memory_id:
                    continue
                memory["content"] = new_content
                memory["updated_at"] = self._now()
                if metadata is not None:
                    merged = dict(memory.get("metadata", {}))
                    merged.update(metadata)
                    memory["metadata"] = merged
                with self._index_lock:
                    self._revision += 1
                    self._user_revisions[user_id] = self._user_revisions.get(user_id, 0) + 1
                return True
        return False

    def delete_record(self, *, memory_id: str) -> bool:
        with self._index_lock:
            user_id = self._memory_index.get(memory_id)
        if user_id is None:
            return False
        with self._user_lock(user_id):
            memories = self._memories.get(user_id, [])
            for index, memory in enumerate(memories):
                if memory["id"] != memory_id:
                    continue
                del memories[index]
                with self._index_lock:
                    self._memory_index.pop(memory_id, None)
                    self._revision += 1
                    self._user_revisions[user_id] = self._user_revisions.get(user_id, 0) + 1
                return True
        return False

    def clear_user(self, *, user_id: str) -> bool:
        with self._user_lock(user_id):
            if user_id not in self._memories:
                return False
            memory_ids = [memory["id"] for memory in self._memories[user_id]]
            del self._memories[user_id]
        with self._index_lock:
            for memory_id in memory_ids:
                self._memory_index.pop(memory_id, None)
            self._revision += 1
            self._user_revisions[user_id] = self._user_revisions.get(user_id, 0) + 1
            self._user_locks.pop(user_id, None)
        return True
