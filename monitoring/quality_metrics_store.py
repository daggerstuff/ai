"""Shared data access for the quality analytics modules.

Every analyzer in this package reads the same two tables: ``conversations``
carries the tier and dataset a conversation belongs to, ``quality_metrics``
carries its validated component scores. They differ only in what they do with
the resulting frame, so the join and the date window live here once.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

COMPONENT_COLUMNS: tuple[str, ...] = (
    "therapeutic_accuracy",
    "conversation_coherence",
    "emotional_authenticity",
    "clinical_compliance",
    "personality_consistency",
    "language_quality",
    "safety_score",
)

SCORE_COLUMNS: tuple[str, ...] = (*COMPONENT_COLUMNS, "overall_quality")

_LATEST_QUALITY_ROW = """
    JOIN quality_metrics q ON q.id = (
        SELECT q2.id
        FROM quality_metrics q2
        WHERE q2.conversation_id = c.id
        ORDER BY q2.validated_at DESC, q2.id DESC
        LIMIT 1
    )
"""


def load_quality_frame(
    db_path: str | Path,
    days_back: int | None = 30,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return one row per conversation joined to its latest quality metrics.

    Args:
        db_path: SQLite database holding the ``conversations`` and
            ``quality_metrics`` tables.
        days_back: Only include conversations created within this many days of
            ``now``. ``None`` disables the window.
        now: Reference point for the window, defaulting to the current UTC time.

    Returns:
        A frame with ``conversation_id``, ``tier``, ``dataset_name``,
        ``created_at``, ``conversation_length``, ``validated_at`` and every
        column in :data:`SCORE_COLUMNS`. Conversations whose metrics are
        missing are dropped, and the score columns are numeric.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"quality database not found: {path}")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT c.id AS conversation_id,
                   c.tier,
                   c.dataset_name,
                   c.created_at,
                   c.conversation_length,
                   q.validated_at,
                   {", ".join(f"q.{column}" for column in SCORE_COLUMNS)}
            FROM conversations c
            {_LATEST_QUALITY_ROW}
            """,
            conn,
        )

    if frame.empty:
        return frame

    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce").dt.tz_localize(None)
    frame["validated_at"] = pd.to_datetime(frame["validated_at"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if days_back is not None:
        reference = now or datetime.now(UTC)
        if reference.tzinfo is not None:
            reference = reference.astimezone(UTC).replace(tzinfo=None)
        cutoff = reference - timedelta(days=days_back)
        frame = frame[frame["created_at"] >= cutoff]

    return frame.dropna(subset=["overall_quality"]).reset_index(drop=True)


def describe_scores(frame: pd.DataFrame, columns: tuple[str, ...] = SCORE_COLUMNS) -> dict[str, dict[str, Any]]:
    """Summarise the score columns as count, mean, std, min, max and quartiles."""
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        summary[column] = {
            "count": int(series.size),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
            "min": float(series.min()),
            "q1": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q3": float(series.quantile(0.75)),
            "max": float(series.max()),
        }
    return summary
