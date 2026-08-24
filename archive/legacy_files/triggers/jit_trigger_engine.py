"""JITTriggerEngine: per-clinician rolling-window flag aggregation.

Consumes CaseFlag events from the AnalysisOrchestrator's EventBus emission,
groups them by ``clinician_id``, evaluates a 7-day rolling window per
clinician, and fires a ``TriggerDecision`` when the flag count reaches
the threshold (default 3).

State is persisted via FlagStore (SQLite or PostgreSQL) so flags survive
service restarts. On init the engine backfills from the store.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from triggers.case_flag import CaseFlag
from triggers.flag_store import FlagStore
from triggers.jit_scenario_injector import TriggerDecision

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
DEFAULT_THRESHOLD = 3


class JITTriggerEngine:
    """Aggregate CaseFlag entries per clinician and emit trigger decisions.

    Args:
        store: Optional FlagStore for persistence. If None, flags are
            kept in-memory only (not recommended for production).
        window_days: Size of the rolling window in days.
        threshold: Number of flags within the window to trigger.
    """

    def __init__(
        self,
        store: FlagStore | None = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
        threshold: int = DEFAULT_THRESHOLD,
    ):
        self._store = store
        self._window_days = window_days
        self._threshold = threshold
        self._flags_by_clinician: dict[str, list[CaseFlag]] = defaultdict(list)

        # Backfill from persistent store on init
        if self._store is not None:
            for flag in self._store.load_all():
                self._flags_by_clinician[flag.clinician_id].append(flag)
            logger.info(
                "JITTriggerEngine backfilled %d flags from store",
                sum(len(v) for v in self._flags_by_clinician.values()),
            )

    def add_flag(self, flag: CaseFlag) -> TriggerDecision | None:
        """Record a new flag and evaluate whether the clinician should trigger.

        Returns a TriggerDecision if the clinician now meets the threshold,
        otherwise None.
        """
        self._flags_by_clinician[flag.clinician_id].append(flag)
        if self._store is not None:
            self._store.save(flag)

        return self._evaluate(flag.clinician_id)

    def rolling_window(
        self, clinician_id: str, *, now: datetime | None = None
    ) -> list[CaseFlag]:
        """Return flags for a clinician within the rolling window.

        Args:
            clinician_id: The clinician whose flags to query.
            now: Override for "current time" (useful for testing).
        """
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=self._window_days)
        return [
            f for f in self._flags_by_clinician.get(clinician_id, [])
            if f.timestamp >= cutoff
        ]

    def _evaluate(self, clinician_id: str) -> TriggerDecision | None:
        """Check whether a clinician has enough recent flags to trigger."""
        recent = self.rolling_window(clinician_id)
        if len(recent) >= self._threshold:
            matching_flags = len(recent)
            logger.info(
                "JIT trigger fired for clinician %s (%d flags in %dd window)",
                clinician_id,
                matching_flags,
                self._window_days,
            )
            return TriggerDecision(
                should_trigger=True,
                matching_flags=matching_flags,
                clinician_id=clinician_id,
            )
        return None

    def evaluate_clinician(self, clinician_id: str) -> TriggerDecision:
        """Force-evaluate a clinician regardless of new flag arrival.

        Returns a TriggerDecision with should_trigger=True if threshold met.
        """
        recent = self.rolling_window(clinician_id)
        should = len(recent) >= self._threshold
        return TriggerDecision(
            should_trigger=should,
            matching_flags=len(recent),
            clinician_id=clinician_id,
        )

    def purge_old_flags(self, *, now: datetime | None = None) -> int:
        """Remove flags older than the rolling window from memory and store.

        Returns the number of flags purged.
        """
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=self._window_days)
        purged = 0
        for clinician_id in list(self._flags_by_clinician.keys()):
            before = len(self._flags_by_clinician[clinician_id])
            self._flags_by_clinician[clinician_id] = [
                f for f in self._flags_by_clinician[clinician_id]
                if f.timestamp >= cutoff
            ]
            purged += before - len(self._flags_by_clinician[clinician_id])

        if self._store is not None:
            self._store.purge_before(cutoff)

        return purged

    def get_clinician_ids(self) -> list[str]:
        """Return all clinician IDs that have flags on record."""
        return list(self._flags_by_clinician.keys())
