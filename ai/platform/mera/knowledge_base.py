"""Knowledge base (KB) for Mera — in-memory condition profile store.

Matches the slimmed prototype contract: pure Python, CPU-resolvable,
no external DB or model dependency.  Profiles mirror
:class:`~platform.mera.types.TherapeuticCondition` and can be queried by
condition id or descriptor overlap for the Memorize stage.
"""

from __future__ import annotations

from .types import TherapeuticCondition


class KnowledgeBase:
    """In-memory KB of clinical conditions for the Memorize stage."""

    def __init__(self) -> None:
        self._profiles: dict[str, TherapeuticCondition] = {}

    def add(self, condition: TherapeuticCondition) -> None:
        self._profiles[condition.condition_id] = condition

    def get(self, condition_id: str) -> TherapeuticCondition | None:
        return self._profiles.get(condition_id)

    def search_by_descriptor(self, descriptor: str, top_k: int = 5) -> list[str]:
        descriptor_lower = descriptor.lower()
        results = [
            cid
            for cid, profile in self._profiles.items()
            if descriptor_lower in (d.lower() for d in profile.typical_symptoms)
            or descriptor_lower in profile.name.lower()
        ]
        return results[:top_k]

    def all_condition_ids(self) -> list[str]:
        return list(self._profiles.keys())
