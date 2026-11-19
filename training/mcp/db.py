"""SQLite job registry — the source of truth for training-infra jobs.

Backend IDs (ovhai UUIDs, W&B run IDs) are stored as references on the row;
the registry's own ``job_id`` is what agents hold. Rows are upserted by
``sync.py`` from live backends and by future submit paths.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    backend_id TEXT,
    backend_state TEXT,
    spec_json TEXT,
    artifacts_json TEXT,
    cost_class TEXT,
    requested_by TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind);
CREATE INDEX IF NOT EXISTS idx_jobs_backend ON jobs(source, backend_id);
"""

_local = threading.local()


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != str(db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        _local.conn = conn
        _local.path = str(db_path)
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_job_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def upsert_job(db_path: Path, job: dict[str, Any]) -> None:
    """Insert or refresh a job row. ``job_id`` is required."""
    now = _now()
    conn = _conn(db_path)
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, kind, source, name, status, backend_id, backend_state,
            spec_json, artifacts_json, cost_class, requested_by, error,
            created_at, updated_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            kind=excluded.kind,
            source=excluded.source,
            name=COALESCE(excluded.name, jobs.name),
            status=excluded.status,
            backend_id=COALESCE(excluded.backend_id, jobs.backend_id),
            backend_state=COALESCE(excluded.backend_state, jobs.backend_state),
            spec_json=COALESCE(excluded.spec_json, jobs.spec_json),
            artifacts_json=COALESCE(excluded.artifacts_json, jobs.artifacts_json),
            cost_class=COALESCE(excluded.cost_class, jobs.cost_class),
            requested_by=COALESCE(excluded.requested_by, jobs.requested_by),
            error=excluded.error,
            updated_at=excluded.updated_at,
            finished_at=COALESCE(excluded.finished_at, jobs.finished_at)
        """,
        (
            job["job_id"],
            job["kind"],
            job["source"],
            job.get("name"),
            job["status"],
            job.get("backend_id"),
            job.get("backend_state"),
            json.dumps(job["spec"]) if job.get("spec") is not None else None,
            json.dumps(job["artifacts"]) if job.get("artifacts") is not None else None,
            job.get("cost_class"),
            job.get("requested_by"),
            job.get("error"),
            job.get("created_at") or now,
            now,
            job.get("finished_at"),
        ),
    )
    conn.commit()


def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    row = _conn(db_path).execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(
    db_path: Path,
    *,
    status: str | None = None,
    kind: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("status", status), ("kind", kind), ("source", source)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(limit, 500)))
    rows = _conn(db_path).execute(f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?", params).fetchall()
    return [_row_to_dict(row) for row in rows]


def find_by_backend(db_path: Path, source: str, backend_id: str) -> dict[str, Any] | None:
    row = (
        _conn(db_path)
        .execute("SELECT * FROM jobs WHERE source = ? AND backend_id = ?", (source, backend_id))
        .fetchone()
    )
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("spec_json", "artifacts_json"):
        raw = data.pop(key)
        data[key.removesuffix("_json")] = json.loads(raw) if raw else None
    return data
