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
    stage TEXT,
    idempotency_key TEXT,
    backend_id TEXT,
    backend_state TEXT,
    backend_ids TEXT,
    spec_json TEXT,
    spec_hash TEXT,
    plan_hash TEXT,
    artifacts_json TEXT,
    artifact_refs TEXT,
    cost_class TEXT,
    cost_estimate TEXT,
    requested_by TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind);
CREATE INDEX IF NOT EXISTS idx_jobs_backend ON jobs(source, backend_id);
CREATE TABLE IF NOT EXISTS plan_cache (
    plan_hash TEXT PRIMARY KEY,
    recipe_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT,
    action TEXT NOT NULL,
    job_id TEXT,
    recipe_id TEXT,
    spec_hash TEXT,
    plan_hash TEXT,
    detail_json TEXT,
    remote_addr TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_job ON audit_log(job_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""

_local = threading.local()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    additions = {
        "stage": "TEXT",
        "idempotency_key": "TEXT",
        "backend_ids": "TEXT",
        "spec_hash": "TEXT",
        "plan_hash": "TEXT",
        "artifact_refs": "TEXT",
        "cost_estimate": "TEXT",
    }
    for column, decl in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key)")
    conn.commit()


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != str(db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        _migrate(conn)
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
            job_id, kind, source, name, status, stage, idempotency_key, backend_id,
            backend_state, backend_ids, spec_json, spec_hash, plan_hash,
            artifacts_json, artifact_refs, cost_class, cost_estimate, requested_by,
            error, created_at, updated_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            kind=excluded.kind,
            source=excluded.source,
            name=COALESCE(excluded.name, jobs.name),
            status=excluded.status,
            stage=COALESCE(excluded.stage, jobs.stage),
            idempotency_key=COALESCE(excluded.idempotency_key, jobs.idempotency_key),
            backend_id=COALESCE(excluded.backend_id, jobs.backend_id),
            backend_state=COALESCE(excluded.backend_state, jobs.backend_state),
            backend_ids=COALESCE(excluded.backend_ids, jobs.backend_ids),
            spec_json=COALESCE(excluded.spec_json, jobs.spec_json),
            spec_hash=COALESCE(excluded.spec_hash, jobs.spec_hash),
            plan_hash=COALESCE(excluded.plan_hash, jobs.plan_hash),
            artifacts_json=COALESCE(excluded.artifacts_json, jobs.artifacts_json),
            artifact_refs=COALESCE(excluded.artifact_refs, jobs.artifact_refs),
            cost_class=COALESCE(excluded.cost_class, jobs.cost_class),
            cost_estimate=COALESCE(excluded.cost_estimate, jobs.cost_estimate),
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
            job.get("stage"),
            job.get("idempotency_key"),
            job.get("backend_id"),
            job.get("backend_state"),
            json.dumps(job["backend_ids"]) if job.get("backend_ids") is not None else None,
            json.dumps(job["spec"]) if job.get("spec") is not None else None,
            job.get("spec_hash"),
            job.get("plan_hash"),
            json.dumps(job["artifacts"]) if job.get("artifacts") is not None else None,
            json.dumps(job["artifact_refs"]) if job.get("artifact_refs") is not None else None,
            job.get("cost_class"),
            job.get("cost_estimate"),
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


def find_by_idempotency(db_path: Path, key: str) -> dict[str, Any] | None:
    if not key:
        return None
    row = _conn(db_path).execute("SELECT * FROM jobs WHERE idempotency_key = ?", (key,)).fetchone()
    return _row_to_dict(row) if row else None


def cache_plan(db_path: Path, plan_hash: str, recipe_id: str, spec: dict[str, Any]) -> None:
    conn = _conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO plan_cache (plan_hash, recipe_id, spec_json, created_at) VALUES (?, ?, ?, ?)",
        (plan_hash, recipe_id, json.dumps(spec), _now()),
    )
    conn.commit()


def get_cached_plan(db_path: Path, plan_hash: str) -> dict[str, Any] | None:
    row = _conn(db_path).execute("SELECT * FROM plan_cache WHERE plan_hash = ?", (plan_hash,)).fetchone()
    if not row:
        return None
    return {
        "plan_hash": row["plan_hash"],
        "recipe_id": row["recipe_id"],
        "spec": json.loads(row["spec_json"]),
        "created_at": row["created_at"],
    }


def write_audit(
    db_path: Path,
    *,
    action: str,
    actor: str | None = None,
    job_id: str | None = None,
    recipe_id: str | None = None,
    spec_hash: str | None = None,
    plan_hash: str | None = None,
    detail: dict[str, Any] | None = None,
    remote_addr: str | None = None,
) -> int:
    """Append one immutable audit row. Returns the new row id.

    Every mutating control-plane action (plan, submit, cancel) is recorded
    with its actor, the recipe/hashes it acted on, and a JSON detail blob so
    a run trace can be reconstructed from the registry alone.
    """
    conn = _conn(db_path)
    cursor = conn.execute(
        """
        INSERT INTO audit_log (
            ts, actor, action, job_id, recipe_id, spec_hash, plan_hash,
            detail_json, remote_addr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now(),
            actor,
            action,
            job_id,
            recipe_id,
            spec_hash,
            plan_hash,
            json.dumps(detail) if detail is not None else None,
            remote_addr,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def list_audit(
    db_path: Path,
    *,
    job_id: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("job_id", job_id), ("action", action)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(limit, 500)))
    rows = _conn(db_path).execute(f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?", params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        raw = data.pop("detail_json", None)
        data["detail"] = json.loads(raw) if raw else None
        out.append(data)
    return out


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("spec_json", "artifacts_json", "backend_ids", "artifact_refs"):
        raw = data.pop(key, None)
        out_key = key.removesuffix("_json") if key.endswith("_json") else key
        data[out_key] = json.loads(raw) if raw else None
    return data
