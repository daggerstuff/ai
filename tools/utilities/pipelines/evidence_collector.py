"""Evidence Collector for PIX-4747.

Extracts the three evidence streams required by the Workstream E spec:

* performance gaps (system metrics falling short of targets)
* clinical safety alerts
* therapist interaction frequencies

and converts them into ``TicketSignal`` records consumable by
``MultiCriteriaScorer``.

Ingestion sources
-----------------
* ``ingest_dicts`` / ``ingest_json``: offline, deterministic, fully tested.
  Each row is a dict with the schema described in the dataclass docstrings.
* ``PostgresEvidenceCollector``: thin adapter that pulls the same three
  streams from PostgreSQL via psycopg2, following the repository's lazy
  import pattern (psycopg2 is not required unless the adapter is used).

No PHI is stored by this module: interaction data is aggregated to
frequencies, and alert rows carry only category, severity, and counts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.tools.utilities.pipelines.scoring import TicketSignal

logger = logging.getLogger(__name__)

DEFAULT_TABLES: dict[str, str] = {
    "performance_gaps": "performance_gaps",
    "safety_alerts": "clinical_safety_alerts",
    "interactions": "therapist_interaction_counts",
}


@dataclass
class EvidenceRow:
    """A single extracted evidence row (any of the three streams)."""

    item_id: str
    title: str
    source: str  # one of: performance_gaps, safety_alerts, interactions
    severity: str = "medium"
    frequency_score: float = 0.0
    performance_gap: float = 0.0
    user_facing_impact: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class EvidenceCollector:
    """Builds ticket signals from raw evidence rows.

    Aggregation: rows are grouped by ``item_id``; per ticket we keep the
    worst severity (max weight), the max frequency/gap/impact across rows,
    and the earliest title.
    """

    _SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def ingest_dicts(self, data: dict[str, Any] | Sequence[dict[str, Any]]) -> list[EvidenceRow]:
        """Ingest rows from a dict payload or a list of row dicts.

        Accepts either a list of row dicts, or a dict with any of the keys
        ``performance_gaps``, ``safety_alerts``, ``interactions`` mapping to
        lists of row dicts (unknown keys are ignored with a warning).
        """
        if isinstance(data, dict):
            rows: list[dict[str, Any]] = []
            for key in ("performance_gaps", "safety_alerts", "interactions"):
                rows.extend(data.get(key, []))
            for key in data:
                if key not in ("performance_gaps", "safety_alerts", "interactions"):
                    logger.warning("EvidenceCollector: ignoring unknown key %r", key)
        else:
            rows = list(data)
        return [self._row_from_dict(r) for r in rows]

    def ingest_json(self, path: str | Path) -> list[EvidenceRow]:
        with open(path) as f:
            return self.ingest_dicts(json.load(f))

    def to_signals(self, rows: Sequence[EvidenceRow]) -> list[TicketSignal]:
        """Aggregate evidence rows into one TicketSignal per item id."""
        by_item: dict[str, list[EvidenceRow]] = {}
        for row in rows:
            by_item.setdefault(row.item_id, []).append(row)
        signals: list[TicketSignal] = []
        for item_id, item_rows in by_item.items():
            signals.append(self._aggregate(item_id, item_rows))
        return signals

    def _aggregate(self, item_id: str, rows: list[EvidenceRow]) -> TicketSignal:
        worst = max(rows, key=lambda r: self._SEVERITY_RANK.get(r.severity, 0))
        return TicketSignal(
            item_id=item_id,
            title=rows[0].title,
            severity=worst.severity,
            frequency_score=max(r.frequency_score for r in rows),
            performance_gap=max(r.performance_gap for r in rows),
            user_facing_impact=max(r.user_facing_impact for r in rows),
        )

    @staticmethod
    def _row_from_dict(raw: dict[str, Any]) -> EvidenceRow:
        return EvidenceRow(
            item_id=str(raw.get("item_id", "")),
            title=str(raw.get("title", "")),
            source=str(raw.get("source", "")),
            severity=str(raw.get("severity", "medium")),
            frequency_score=float(raw.get("frequency_score", 0.0)),
            performance_gap=float(raw.get("performance_gap", 0.0)),
            user_facing_impact=float(raw.get("user_facing_impact", 0.0)),
        )


class PostgresEvidenceCollector(EvidenceCollector):
    """Thin PostgreSQL adapter for the three evidence streams.

    Mirrors the repository's psycopg2 lazy-import pattern (see
    ``configs/flag_store.py``): the driver is imported only when the adapter
    is constructed, so the rest of the pipeline remains importable without
    it.
    """

    def __init__(self, database_url: str, tables: dict[str, str] | None = None) -> None:
        try:
            import importlib

            psycopg2 = importlib.import_module("psycopg2")
        except ImportError:
            raise RuntimeError(
                "psycopg2 not installed — PostgreSQL evidence collection unavailable. "
                "Install with: pip install psycopg2-binary"
            ) from None
        self._psycopg2 = psycopg2
        self._database_url = database_url
        self._tables = dict(DEFAULT_TABLES)
        if tables:
            self._tables.update(tables)

    def collect(self) -> list[EvidenceRow]:
        """Fetch all three streams in one connection."""
        conn = self._psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                rows: list[EvidenceRow] = []
                rows.extend(self._fetch_performance_gaps(cur))
                rows.extend(self._fetch_safety_alerts(cur))
                rows.extend(self._fetch_interaction_frequencies(cur))
                return rows
        finally:
            conn.close()

    def _fetch_performance_gaps(self, cur: Any) -> list[EvidenceRow]:
        query = (
            "SELECT item_id, title, severity, performance_gap, user_facing_impact "
            f"FROM {self._tables['performance_gaps']}"
        )
        cur.execute(query)
        return [
            EvidenceRow(
                item_id=str(r[0]),
                title=str(r[1]),
                source="performance_gaps",
                severity=str(r[2]),
                performance_gap=float(r[3] or 0.0),
                user_facing_impact=float(r[4] or 0.0),
            )
            for r in cur.fetchall()
        ]

    def _fetch_safety_alerts(self, cur: Any) -> list[EvidenceRow]:
        query = f"SELECT item_id, title, severity, alert_count, user_facing_impact FROM {self._tables['safety_alerts']}"
        cur.execute(query)
        return [
            EvidenceRow(
                item_id=str(r[0]),
                title=str(r[1]),
                source="safety_alerts",
                severity=str(r[2]),
                # Normalise alert count into a [0, 1] frequency proxy.
                frequency_score=min(1.0, float(r[3] or 0.0) / 100.0),
                user_facing_impact=float(r[4] or 0.0),
                extra={"alert_count": r[3]},
            )
            for r in cur.fetchall()
        ]

    def _fetch_interaction_frequencies(self, cur: Any) -> list[EvidenceRow]:
        query = f"SELECT item_id, title, interaction_count, total_sessions FROM {self._tables['interactions']}"
        cur.execute(query)
        rows: list[EvidenceRow] = []
        for r in cur.fetchall():
            count = float(r[2] or 0.0)
            total = float(r[3] or 0.0)
            frequency = min(1.0, count / total) if total > 0 else 0.0
            rows.append(
                EvidenceRow(
                    item_id=str(r[0]),
                    title=str(r[1]),
                    source="interactions",
                    frequency_score=frequency,
                    extra={"interaction_count": r[2], "total_sessions": r[3]},
                )
            )
        return rows


__all__ = [
    "DEFAULT_TABLES",
    "EvidenceCollector",
    "EvidenceRow",
    "PostgresEvidenceCollector",
]
