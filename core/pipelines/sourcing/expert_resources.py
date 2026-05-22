"""Expert resource metadata accessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Expert:
    name: str
    specialties: list[str]
    contact: str | None = None


class ExpertResources:
    """Lookup static and user-defined expert resources."""

    def __init__(self, catalog: list[Expert] | None = None) -> None:
        self.catalog = catalog or []

    def register(self, expert: Expert) -> None:
        self.catalog.append(expert)

    def find_by_specialty(self, specialty: str) -> list[Expert]:
        query = specialty.lower()
        return [e for e in self.catalog if any(query == s.lower() for s in e.specialties)]

    def to_records(self) -> list[dict[str, Any]]:
        return [e.__dict__ for e in self.catalog]


__all__ = ["Expert", "ExpertResources"]
