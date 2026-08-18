from __future__ import annotations

from collections import defaultdict

from .null_memory_record import NullMemoryRecord

_MAX_INDEXED_TOKENS_PER_RECORD = 128
_MAX_INDEXED_TAGS_PER_RECORD = 32


class NullMemorySearchIndex:
    """Per-user token and tag index for null-memory record lookup."""

    def __init__(self) -> None:
        self._tokens: dict[str, dict[str, set[str]]] = defaultdict(dict)
        self._tags: dict[str, dict[str, set[str]]] = defaultdict(dict)

    def add(self, *, user_id: str, record: NullMemoryRecord) -> None:
        for token in record.content_tokens[:_MAX_INDEXED_TOKENS_PER_RECORD]:
            self._tokens[user_id].setdefault(token, set()).add(record.id)
        for tag in record.normalized_tags[:_MAX_INDEXED_TAGS_PER_RECORD]:
            self._tags[user_id].setdefault(tag, set()).add(record.id)

    def rebuild_user(self, *, user_id: str, records: tuple[NullMemoryRecord, ...]) -> None:
        self.clear_user(user_id=user_id)
        for record in records:
            self.add(user_id=user_id, record=record)

    def remove(self, *, user_id: str, record: NullMemoryRecord) -> None:
        token_index = self._tokens.get(user_id, {})
        for token in record.content_tokens[:_MAX_INDEXED_TOKENS_PER_RECORD]:
            ids = token_index.get(token)
            if not ids:
                continue
            ids.discard(record.id)
            if not ids:
                token_index.pop(token, None)
        if not token_index:
            self._tokens.pop(user_id, None)

        tag_index = self._tags.get(user_id, {})
        for tag in record.normalized_tags[:_MAX_INDEXED_TAGS_PER_RECORD]:
            ids = tag_index.get(tag)
            if not ids:
                continue
            ids.discard(record.id)
            if not ids:
                tag_index.pop(tag, None)
        if not tag_index:
            self._tags.pop(user_id, None)

    def clear_user(self, *, user_id: str) -> None:
        self._tokens.pop(user_id, None)
        self._tags.pop(user_id, None)

    def compact_user(self, *, user_id: str, allowed_record_ids: set[str]) -> None:
        if not allowed_record_ids:
            self.clear_user(user_id=user_id)
            return
        for index in (self._tokens.get(user_id, {}), self._tags.get(user_id, {})):
            stale_terms = []
            for term, ids in index.items():
                ids.intersection_update(allowed_record_ids)
                if not ids:
                    stale_terms.append(term)
            for term in stale_terms:
                index.pop(term, None)
        if not self._tokens.get(user_id):
            self._tokens.pop(user_id, None)
        if not self._tags.get(user_id):
            self._tags.pop(user_id, None)

    def candidate_ids(
        self,
        *,
        user_id: str,
        record_ids: set[str],
        query_lower: str,
        tags: tuple[str, ...] = (),
        tags_match: str = "any",
        query_match: str = "all",
    ) -> tuple[str, ...]:
        candidates = set(record_ids)

        query_tokens = tuple(token for token in query_lower.split() if token)
        if query_tokens:
            token_ids = self._candidate_ids_for_terms(
                index=self._tokens.get(user_id, {}),
                terms=query_tokens,
                match_mode=query_match,
            )
            candidates &= token_ids

        if tags:
            tag_ids = self._candidate_ids_for_terms(
                index=self._tags.get(user_id, {}),
                terms=tags,
                match_mode=tags_match,
            )
            candidates &= tag_ids

        return tuple(sorted(candidates))

    def search_ids(
        self,
        *,
        user_id: str,
        records: dict[str, NullMemoryRecord],
        query_lower: str,
        tags: tuple[str, ...] = (),
        tags_match: str = "any",
        query_match: str = "all",
    ) -> tuple[str, ...]:
        candidate_ids = self.candidate_ids(
            user_id=user_id,
            record_ids=set(records.keys()),
            query_lower=query_lower,
            tags=tags,
            tags_match=tags_match,
            query_match=query_match,
        )
        return tuple(memory_id for memory_id in candidate_ids if memory_id in records)

    @staticmethod
    def _candidate_ids_for_terms(
        *,
        index: dict[str, set[str]],
        terms: tuple[str, ...],
        match_mode: str,
    ) -> set[str]:
        ordered_terms = sorted(terms, key=lambda term: len(index.get(term, ())))
        term_iter = iter(ordered_terms)
        first_term = next(term_iter, None)
        if first_term is None:
            return set()
        result = set(index.get(first_term, set()))
        if match_mode == "all":
            for term in term_iter:
                result &= index.get(term, set())
                if not result:
                    break
            return result
        return set().union(result, *(index.get(term, set()) for term in term_iter))
