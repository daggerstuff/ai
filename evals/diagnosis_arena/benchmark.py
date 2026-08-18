"""
DiagnosisArena benchmark adapter.

Loads structured clinical cases from a JSONL file or from an in-memory
fixture list. Cases follow the schema defined in ``types.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .types import ClinicalCase, Difficulty


def case_from_dict(payload: dict) -> ClinicalCase:
    """Construct a ClinicalCase from a JSON-compatible dict."""
    required = {"case_id", "difficulty", "presentation"}
    missing = required - payload.keys()
    if missing:
        msg = f"case payload missing fields: {sorted(missing)}"
        raise ValueError(msg)

    return ClinicalCase(
        case_id=str(payload["case_id"]),
        difficulty=Difficulty(payload["difficulty"]),
        presentation=str(payload["presentation"]),
        history=str(payload.get("history", "")),
        exam=str(payload.get("exam", "")),
        labs=str(payload.get("labs", "")),
        imaging=str(payload.get("imaging", "")),
        progression=str(payload.get("progression", "")),
        mcq_options=tuple(payload.get("mcq_options", [])),
        final_diagnosis=str(payload.get("final_diagnosis", "")),
        differential_diagnoses=tuple(payload.get("differential_diagnoses", [])),
        supporting_evidence=tuple(payload.get("supporting_evidence", [])),
        key_differentiators=tuple(payload.get("key_differentiators", [])),
    )


class DiagnosisArenaBenchmark:
    """In-memory or file-backed benchmark of ClinicalCases."""

    def __init__(self, cases: Iterable[ClinicalCase] = ()):
        self._cases: list[ClinicalCase] = list(cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)

    def by_difficulty(self, difficulty: Difficulty) -> list[ClinicalCase]:
        """Return cases filtered by difficulty tier."""
        return [c for c in self._cases if c.difficulty is difficulty]

    def get(self, case_id: str) -> ClinicalCase:
        """Look up a case by id; raises KeyError if absent."""
        for case in self._cases:
            if case.case_id == case_id:
                return case
        msg = f"unknown case_id: {case_id}"
        raise KeyError(msg)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> DiagnosisArenaBenchmark:
        """Load cases from a JSONL file (one case per line)."""
        text = Path(path).read_text(encoding="utf-8")
        cases: list[ClinicalCase] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cases.append(case_from_dict(json.loads(stripped)))
        return cls(cases)

    @classmethod
    def from_json(cls, path: str | Path) -> DiagnosisArenaBenchmark:
        """Load cases from a JSON array file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            msg = "expected top-level JSON array of case objects"
            raise ValueError(msg)
        return cls(case_from_dict(item) for item in data)

    def add(self, case: ClinicalCase) -> None:
        """Append a single case to the benchmark."""
        self._cases.append(case)

    def extend(self, cases: Iterable[ClinicalCase]) -> None:
        """Append multiple cases to the benchmark."""
        self._cases.extend(cases)

    def to_jsonl(self, path: str | Path) -> None:
        """Persist the benchmark to a JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for case in self._cases:
                payload = {
                    "case_id": case.case_id,
                    "difficulty": case.difficulty.value,
                    "presentation": case.presentation,
                    "history": case.history,
                    "exam": case.exam,
                    "labs": case.labs,
                    "imaging": case.imaging,
                    "progression": case.progression,
                    "mcq_options": list(case.mcq_options),
                    "final_diagnosis": case.final_diagnosis,
                    "differential_diagnoses": list(case.differential_diagnoses),
                    "supporting_evidence": list(case.supporting_evidence),
                    "key_differentiators": list(case.key_differentiators),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
