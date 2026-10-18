"""Collect and aggregate psychometric instrument metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InstrumentResult:
    name: str
    version: str
    valid: bool


class InstrumentsCollector:
    """Collect instrument definitions and usage metadata."""

    def __init__(self) -> None:
        self._instruments: list[InstrumentResult] = []

    def collect(self, items: list[dict[str, Any]]) -> list[InstrumentResult]:
        results = []
        for item in items:
            name = str(item.get("name", "unknown"))
            version = str(item.get("version", "1.0"))
            result = InstrumentResult(name=name, version=version, valid=bool(item.get("valid", True)))
            results.append(result)
            self._instruments.append(result)
        return results

    def summary(self) -> dict[str, Any]:
        return {"total": len(self._instruments), "valid": sum(1 for i in self._instruments if i.valid)}


__all__ = ["InstrumentResult", "InstrumentsCollector"]
