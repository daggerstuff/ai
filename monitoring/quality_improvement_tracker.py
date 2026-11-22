"""Tracking of quality improvement interventions.

An intervention is a named change (a training run, a prompt revision, a dataset
fix) aimed at one quality component. This module owns their lifecycle —
created, started, measured, completed — and measures the effect by comparing the
target component before and after the intervention window against the main
quality database.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality_metrics_store import load_quality_frame

INTERVENTION_STATUSES = ("created", "in_progress", "completed", "abandoned")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interventions (
    intervention_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    intervention_type TEXT NOT NULL,
    target_component TEXT NOT NULL,
    expected_improvement REAL NOT NULL,
    created_by TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS intervention_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id TEXT NOT NULL,
    notes TEXT,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (intervention_id) REFERENCES interventions (intervention_id)
);
"""


@dataclass
class Intervention:
    """One tracked intervention and its lifecycle timestamps."""

    intervention_id: str
    name: str
    description: str
    intervention_type: str
    target_component: str
    expected_improvement: float
    created_by: str
    created_at: str
    status: str = "created"
    started_at: str | None = None
    completed_at: str | None = None
    measurements: list[str] = field(default_factory=list)


@dataclass
class InterventionImpact:
    """Measured effect of one intervention on its target component."""

    intervention_id: str
    target_component: str
    baseline_mean: float
    current_mean: float
    absolute_change: float
    relative_change: float
    expected_improvement: float
    target_progress: float
    verdict: str
    improvement_metrics: dict[str, Any] = field(default_factory=dict)
    statistical_tests: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sample_baseline: int = 0
    sample_current: int = 0


class QualityImprovementTracker:
    """Persist interventions and measure what they changed."""

    def __init__(self, db_path: str | Path | None = None, interventions_db: str | Path | None = None) -> None:
        if db_path is None:
            handle, temp_path = tempfile.mkstemp(prefix="quality_metrics_", suffix=".db")
            os.close(handle)
            Path(temp_path).unlink(missing_ok=True)
            db_path = temp_path
        self.db_path = db_path if isinstance(db_path, str) else str(db_path)

        if interventions_db is not None:
            self.interventions_db = Path(interventions_db)
        else:
            source = Path(self.db_path)
            self.interventions_db = source.with_name(f"{source.stem}_interventions.db")

        self.interventions_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.interventions_db) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.interventions_db)
        connection.row_factory = sqlite3.Row
        return connection

    def create_intervention(
        self,
        name: str,
        description: str,
        intervention_type: str,
        target_component: str,
        expected_improvement: float,
        created_by: str = "",
    ) -> str:
        """Register a new intervention and return its identifier."""
        created_at = datetime.now(UTC)
        intervention_id = f"INT_{created_at.strftime('%Y%m%d%H%M%S')}_{created_at.microsecond:06d}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interventions (
                    intervention_id, name, description, intervention_type, target_component,
                    expected_improvement, created_by, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'created')
                """,
                (
                    intervention_id,
                    name,
                    description,
                    intervention_type,
                    target_component,
                    float(expected_improvement),
                    created_by,
                    created_at.isoformat(),
                ),
            )
        return intervention_id

    def _transition(self, intervention_id: str, from_status: str, to_status: str, column: str) -> bool:
        stamp = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE interventions SET status = ?, {column} = ? WHERE intervention_id = ? AND status = ?",
                (to_status, stamp, intervention_id, from_status),
            )
            return cursor.rowcount > 0

    def start_intervention(self, intervention_id: str) -> bool:
        """Move an intervention from created to in_progress."""
        return self._transition(intervention_id, "created", "in_progress", "started_at")

    def complete_intervention(self, intervention_id: str) -> bool:
        """Move an intervention from in_progress to completed."""
        return self._transition(intervention_id, "in_progress", "completed", "completed_at")

    def abandon_intervention(self, intervention_id: str) -> bool:
        """Close an intervention without claiming an effect."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE interventions SET status = 'abandoned' WHERE intervention_id = ? AND status != 'completed'",
                (intervention_id,),
            )
            return cursor.rowcount > 0

    def record_progress_measurement(self, intervention_id: str, notes: str) -> bool:
        """Attach a free-text progress note to a running intervention."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM interventions WHERE intervention_id = ?", (intervention_id,)
            ).fetchone()
            if row is None or row["status"] not in {"in_progress", "created"}:
                return False
            conn.execute(
                "INSERT INTO intervention_measurements (intervention_id, notes, recorded_at) VALUES (?, ?, ?)",
                (intervention_id, notes, datetime.now(UTC).isoformat()),
            )
            return True

    def get_intervention(self, intervention_id: str) -> Intervention | None:
        """Load one intervention with its measurements."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM interventions WHERE intervention_id = ?", (intervention_id,)).fetchone()
            if row is None:
                return None
            measurements = [
                record["notes"]
                for record in conn.execute(
                    "SELECT notes FROM intervention_measurements WHERE intervention_id = ? ORDER BY id",
                    (intervention_id,),
                )
            ]
        return Intervention(
            intervention_id=row["intervention_id"],
            name=row["name"],
            description=row["description"] or "",
            intervention_type=row["intervention_type"],
            target_component=row["target_component"],
            expected_improvement=float(row["expected_improvement"]),
            created_by=row["created_by"] or "",
            created_at=row["created_at"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            measurements=measurements,
        )

    def list_interventions(self, status: str | None = None) -> list[Intervention]:
        """List interventions, optionally filtered by status."""
        query = "SELECT intervention_id FROM interventions"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at"
        with self._connect() as conn:
            identifiers = [row["intervention_id"] for row in conn.execute(query, parameters)]
        return [intervention for identifier in identifiers if (intervention := self.get_intervention(identifier))]

    def analyze_intervention_impact(self, intervention_id: str) -> InterventionImpact:
        """Compare the target component before and after the intervention."""
        intervention = self.get_intervention(intervention_id)
        if intervention is None:
            raise KeyError(f"unknown intervention: {intervention_id}")

        started_at = pd.Timestamp(intervention.started_at or intervention.created_at)
        completed_at = pd.Timestamp(intervention.completed_at or datetime.now(UTC))

        frame = load_quality_frame(self.db_path, days_back=None)
        component = intervention.target_component
        if frame.empty or component not in frame.columns:
            return InterventionImpact(
                intervention_id=intervention_id,
                target_component=component,
                baseline_mean=0.0,
                current_mean=0.0,
                absolute_change=0.0,
                relative_change=0.0,
                expected_improvement=intervention.expected_improvement,
                target_progress=0.0,
                verdict="no_data",
                improvement_metrics={
                    "component": component,
                    "available": False,
                    "absolute_improvement": 0.0,
                    "target_achievement": 0.0,
                },
            )

        timestamps = pd.to_datetime(frame["created_at"], utc=True)
        baseline_window = frame[timestamps < started_at][component].dropna()
        current_window = frame[timestamps >= completed_at][component].dropna()
        if baseline_window.empty or current_window.empty:
            baseline_window = frame[component].dropna()
            current_window = baseline_window

        baseline_mean = float(baseline_window.mean())
        current_mean = float(current_window.mean())
        absolute_change = current_mean - baseline_mean
        relative_change = absolute_change / baseline_mean if baseline_mean else 0.0
        target_progress = (
            absolute_change / intervention.expected_improvement if intervention.expected_improvement else 0.0
        )
        verdict = (
            "exceeded"
            if absolute_change >= intervention.expected_improvement
            else "met"
            if absolute_change > 0
            else "no_change"
            if absolute_change == 0
            else "regressed"
        )

        statistical_tests: list[dict[str, Any]] = []
        if baseline_window.size >= 2 and current_window.size >= 2:
            welch = scipy_stats.ttest_ind(
                baseline_window.to_numpy(dtype=float),
                current_window.to_numpy(dtype=float),
                equal_var=False,
                nan_policy="omit",
            )
            statistical_tests.append(
                {
                    "test": "welch_t",
                    "statistic": float(welch.statistic),
                    "p_value": float(welch.pvalue) if np.isfinite(welch.pvalue) else 1.0,
                    "significant": bool(np.isfinite(welch.pvalue) and welch.pvalue < 0.05),
                    "sample_baseline": int(baseline_window.size),
                    "sample_current": int(current_window.size),
                }
            )

        recommendations: list[str] = []
        if verdict == "regressed":
            recommendations.append(
                f"Roll back or revise {intervention.name}: {component} fell by {abs(absolute_change):.4f}."
            )
        elif verdict == "exceeded":
            recommendations.append(
                f"Keep {intervention.name}: {component} gained {absolute_change:.4f}, above the "
                f"{intervention.expected_improvement:.4f} target."
            )
        elif verdict == "met":
            recommendations.append(
                f"{intervention.name} moved {component} in the right direction but short of target; "
                "extend the intervention window before deciding."
            )
        else:
            recommendations.append(
                f"{intervention.name} left {component} unchanged; check whether the intervention reached "
                "the conversations it targeted."
            )
        if statistical_tests and not statistical_tests[0]["significant"]:
            recommendations.append(
                "The before/after difference is not statistically significant at p < 0.05; "
                "collect more conversations before claiming an effect."
            )

        return InterventionImpact(
            intervention_id=intervention_id,
            target_component=component,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            absolute_change=absolute_change,
            relative_change=relative_change,
            expected_improvement=intervention.expected_improvement,
            target_progress=float(target_progress),
            verdict=verdict,
            improvement_metrics={
                "component": component,
                "available": True,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "absolute_change": absolute_change,
                "absolute_improvement": absolute_change,
                "relative_change": relative_change,
                "expected_improvement": intervention.expected_improvement,
                "target_progress": float(target_progress),
                "target_achievement": float(min(target_progress, 1.0)),
                "standard_error": float(
                    np.sqrt(
                        baseline_window.var(ddof=1) / max(baseline_window.size, 1)
                        + current_window.var(ddof=1) / max(current_window.size, 1)
                    )
                )
                if baseline_window.size > 1 and current_window.size > 1
                else 0.0,
            },
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            sample_baseline=int(baseline_window.size),
            sample_current=int(current_window.size),
        )
