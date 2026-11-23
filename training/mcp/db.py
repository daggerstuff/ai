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


#: Bumped whenever ``_migrate`` gains a step. Recorded in ``PRAGMA user_version``
#: so an operator can tell which schema a registry file is at, and so a future
#: migration can branch on the version instead of probing columns forever.
SCHEMA_VERSION = 1


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
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


#: Registry files are owner-only. The registry holds no credentials today, but
#: it does hold job specs, actor names and the audit trail; leaving it 644 made
#: all of that readable by every local user. SQLite recreates -wal/-shm on each
#: connection, so the mode has to be re-applied rather than set once.
_REGISTRY_MODE = 0o600
_REGISTRY_DIR_MODE = 0o700


def _harden(path: Path) -> None:
    """Best-effort chmod of a registry file/dir; never fail the caller."""
    try:
        if path.exists():
            path.chmod(_REGISTRY_MODE if path.is_file() else _REGISTRY_DIR_MODE)
    except OSError:
        pass


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
        _harden(db_path)
        _harden(db_path.parent)
        for sidecar in (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
            _harden(sidecar)
    return conn


def now() -> str:
    """UTC timestamp in the registry's canonical format (second resolution)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# Backwards-compatible private alias used inside this module.
_now = now


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


#: Statuses that end a job's life. Nothing may overwrite a row in one of
#: these states: the operator's cancellation and the monitor's process exit
#: race, and whichever lands first is the outcome the job really had.
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def transition_job(
    db_path: Path,
    job_id: str,
    *,
    status: str,
    stage: str | None = None,
    error: str | None = None,
    finished_at: str | None = None,
) -> bool:
    """Move a job to ``status`` unless an operator already cancelled it.

    The guard is part of the UPDATE rather than a read-then-write: ``cancel()``
    runs on the MCP request thread while the subprocess monitor writes from its
    own thread, so a monitor that read ``running`` before the cancel landed must
    not overwrite the terminal ``cancelled`` row with a process-exit status.
    Returns False when the row is already terminal (nothing was written): a
    monitor that read an active row before an operator cancelled, or before
    another terminal write landed, must not revive or rewrite that outcome.
    """
    conn = _conn(db_path)
    cursor = conn.execute(
        """
        UPDATE jobs SET
            status = ?,
            stage = COALESCE(?, stage),
            error = ?,
            finished_at = COALESCE(?, finished_at),
            updated_at = ?
        WHERE job_id = ? AND status NOT IN ("succeeded", "failed", "cancelled")
        """,
        (status, stage, error, finished_at, _now(), job_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_job_cancelled(db_path: Path, job_id: str) -> bool:
    """Cancel a job only if it has not already reached a terminal state.

    The guard lives in the UPDATE so a cancellation cannot overwrite a result the
    monitor committed while the request thread was reading the row. Returns False
    when the job was already over; the caller must not re-record the outcome.
    """
    conn = _conn(db_path)
    now = _now()
    cursor = conn.execute(
        """
        UPDATE jobs SET
            status = 'cancelled',
            stage = 'cancelled',
            error = 'cancelled by operator',
            finished_at = ?,
            updated_at = ?
        WHERE job_id = ? AND status NOT IN ('succeeded', 'failed', 'cancelled')
        """,
        (now, now, job_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def fail_orphaned_local_jobs(db_path: Path) -> int:
    """Fail local jobs left active by a previous server process.

    A local job is a subprocess of this server, so any local row still active when
    a new process starts belongs to a process that is gone. Left alone it stays
    `running` forever and consumes the concurrency slot it no longer holds.
    """
    conn = _conn(db_path)
    now = _now()
    cursor = conn.execute(
        """
        UPDATE jobs SET
            status = 'failed',
            stage = 'failed',
            error = 'orphaned: server restarted while the job was active',
            finished_at = ?,
            updated_at = ?
        WHERE source = 'local' AND status IN ('queued', 'submitted', 'running', 'collecting')
        """,
        (now, now),
    )
    conn.commit()
    return cursor.rowcount


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


def cap_rows(db_path: Path, *, since: str | None = None) -> list[dict[str, Any]]:
    """Every job row created at or after ``since``, unpaged.

    ``list_jobs`` clamps to 500 rows, which is right for a UI listing but wrong
    for a budget check: silently dropping rows would let the daily/monthly caps
    be exceeded once the registry grows past one page.

    ``since`` is compared lexicographically, which is correct for the
    registry's ISO-8601 ``created_at`` format — a bare date (``2026-11-20``)
    matches that whole day, and a full timestamp matches from that instant.
    """
    if since:
        rows = _conn(db_path).execute("SELECT * FROM jobs WHERE created_at >= ?", (since,)).fetchall()
    else:
        rows = _conn(db_path).execute("SELECT * FROM jobs").fetchall()
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


def get_cached_plan(db_path: Path, plan_hash: str, *, max_age_seconds: int | None = None) -> dict[str, Any] | None:
    """Return the cached plan, or None when absent or older than ``max_age``.

    The rejection message has always said "not found or expired", but nothing
    implemented expiry: the cache grew forever and an approval from weeks ago
    still bound. Plans are cheap to recompute, so an age bound is safe.
    """
    row = _conn(db_path).execute("SELECT * FROM plan_cache WHERE plan_hash = ?", (plan_hash,)).fetchone()
    if not row:
        return None
    created_at = row["created_at"]
    if max_age_seconds is not None and _age_seconds(created_at) > max_age_seconds:
        return None
    return {
        "plan_hash": row["plan_hash"],
        "recipe_id": row["recipe_id"],
        "spec": json.loads(row["spec_json"]),
        "created_at": created_at,
    }


def _age_seconds(created_at: str | None) -> float:
    """Seconds since an ISO-8601 ``created_at``; ``inf`` when unparseable.

    Fail closed: a row whose timestamp cannot be read is treated as ancient, so
    a corrupt cache entry cannot resurrect an unbounded approval window.
    """
    if not created_at:
        return float("inf")
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - parsed).total_seconds())


def prune_plans(db_path: Path, *, max_age_seconds: int) -> int:
    """Delete plan_cache rows older than ``max_age_seconds``; return the count."""
    conn = _conn(db_path)
    rows = conn.execute("SELECT plan_hash, created_at FROM plan_cache").fetchall()
    stale = [row["plan_hash"] for row in rows if _age_seconds(row["created_at"]) > max_age_seconds]
    if stale:
        conn.executemany("DELETE FROM plan_cache WHERE plan_hash = ?", [(h,) for h in stale])
        conn.commit()
    return len(stale)


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
