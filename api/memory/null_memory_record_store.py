from __future__ import annotations

import threading
from collections import OrderedDict

from .null_memory_index import NullMemoryIndex
from .null_memory_record import NullMemoryRecord
from .null_memory_record_factory import NullMemoryRecordFactory
from .null_memory_search_index import NullMemorySearchIndex


class InMemoryRecordStore:
    """Low-level record persistence and ownership bookkeeping."""

    def __init__(self, *, index: NullMemoryIndex, record_factory: NullMemoryRecordFactory) -> None:
        self._records: dict[str, OrderedDict[str, NullMemoryRecord]] = {}
        self._id_source = index
        self._record_factory = record_factory
        self._index_lock = threading.RLock()
        self._user_locks: dict[str, threading.RLock] = {}
        self._search_index = NullMemorySearchIndex()

    def _user_lock(self, user_id: str) -> threading.RLock:
        with self._index_lock:
            return self._user_locks.setdefault(user_id, threading.RLock())

    def generate_id(self) -> str:
        return self._id_source.generate_id()

    def now(self) -> str:
        return self._record_factory.now()

    def create_record(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> NullMemoryRecord:
        return self._record_factory.create(
            content=content,
            user_id=user_id,
            metadata=metadata,
            memory_id=memory_id,
        )

    def create_replacement_record(
        self,
        *,
        existing_record: NullMemoryRecord,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> NullMemoryRecord:
        return self._record_factory.create_replacement(
            existing_record=existing_record,
            content=content,
            user_id=user_id,
            metadata=metadata,
            memory_id=memory_id,
        )

    def upsert_content(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> tuple[NullMemoryRecord, NullMemoryRecord | None]:
        with self._user_lock(user_id):
            resolved_memory_id = memory_id or self.generate_id()
            existing_record = self._records.get(user_id, {}).get(resolved_memory_id)
            record = (
                self.create_replacement_record(
                    existing_record=existing_record,
                    content=content,
                    user_id=user_id,
                    metadata=metadata,
                    memory_id=resolved_memory_id,
                )
                if existing_record is not None
                else self.create_record(
                    content=content,
                    user_id=user_id,
                    metadata=metadata,
                    memory_id=resolved_memory_id,
                )
            )
            previous = self.upsert(user_id=user_id, record=record)
            return record, previous

    def upsert(self, *, user_id: str, record: NullMemoryRecord) -> NullMemoryRecord | None:
        with self._user_lock(user_id):
            user_records = self._records.setdefault(user_id, OrderedDict())
            previous = user_records.get(record.id)
            if previous is not None:
                self._search_index.remove(user_id=user_id, record=previous)
            user_records[record.id] = record
            self._search_index.add(user_id=user_id, record=record)
            return previous

    def evict_oldest(self, *, user_id: str) -> NullMemoryRecord | None:
        with self._user_lock(user_id):
            user_records = self._records.get(user_id)
            if user_records is None or len(user_records) == 0:
                return None
            _record_id, record = user_records.popitem(last=False)
            if record is None:
                return None
            self._search_index.remove(user_id=user_id, record=record)
            if not user_records:
                self._records.pop(user_id, None)
                self._search_index.clear_user(user_id=user_id)
            return record

    def list_for_user(self, *, user_id: str) -> tuple[NullMemoryRecord, ...]:
        with self._user_lock(user_id):
            return tuple(self._records.get(user_id, {}).values())

    def snapshot_for_user(self, *, user_id: str) -> tuple[NullMemoryRecord, ...]:
        with self._user_lock(user_id):
            return tuple(self._records.get(user_id, {}).values())

    def search_for_user(
        self,
        *,
        query_lower: str,
        user_id: str,
        tags: tuple[str, ...] = (),
        tags_match: str = "any",
    ) -> tuple[NullMemoryRecord, ...]:
        with self._user_lock(user_id):
            records = self._records.get(user_id)
            if not records:
                return ()
            candidate_ids = self._search_index.search_ids(
                user_id=user_id,
                records=records,
                query_lower=query_lower,
                tags=tags,
                tags_match=tags_match,
                query_match="all",
            )
            return tuple(records[memory_id] for memory_id in candidate_ids if memory_id in records)

    def get(self, *, memory_id: str, user_id: str) -> NullMemoryRecord | None:
        with self._user_lock(user_id):
            return self._records.get(user_id, {}).get(memory_id)

    def replace(
        self,
        *,
        memory_id: str,
        user_id: str,
        record: NullMemoryRecord,
    ) -> NullMemoryRecord | None:
        with self._user_lock(user_id):
            current = self.get(memory_id=memory_id, user_id=user_id)
            if current is None:
                return None
            user_records = self._records.get(user_id)
            if user_records is None:
                return None
            user_records[memory_id] = record
            self._search_index.remove(user_id=user_id, record=current)
            self._search_index.add(user_id=user_id, record=record)
            return current

    def delete(self, *, memory_id: str, user_id: str) -> NullMemoryRecord | None:
        with self._user_lock(user_id):
            current = self.get(memory_id=memory_id, user_id=user_id)
            if current is None:
                return None
            user_records = self._records.get(user_id)
            if user_records is None:
                return None
            user_records.pop(memory_id, None)
            self._search_index.remove(user_id=user_id, record=current)
            if not user_records:
                self._records.pop(user_id, None)
                self._search_index.clear_user(user_id=user_id)
            return current

    def clear_user(self, *, user_id: str) -> tuple[str, ...]:
        with self._user_lock(user_id):
            user_records = self._records.pop(user_id, None)
            if not user_records:
                return ()
            self._search_index.clear_user(user_id=user_id)
            removed_ids = tuple(user_records.keys())
            with self._index_lock:
                self._user_locks.pop(user_id, None)
            return removed_ids

    def user_count(self, *, user_id: str) -> int:
        with self._user_lock(user_id):
            return len(self._records.get(user_id, {}))


NullMemoryRecordStore = InMemoryRecordStore
