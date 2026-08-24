"""SQLite/PostgreSQL persistence for CaseFlag entries.

Stores flags keyed by ``clinician_id`` so the JITTriggerEngine can
recover state across service restarts. Uses SQLite by default (zero
config) and PostgreSQL when ``DATABASE_URL`` is set to a postgres DSN.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from .case_flag import CaseFlag

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS case_flags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    clinician_id TEXT    NOT NULL,
    flag_type    TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    payload     TEXT    NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_case_flags_clinician_ts
    ON case_flags (clinician_id, timestamp);
"""

_CREATE_PG_SQL = """
CREATE TABLE IF NOT EXISTS case_flags (
    id          SERIAL PRIMARY KEY,
    clinician_id TEXT    NOT NULL,
    flag_type    TEXT    NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    payload     JSONB NOT NULL
);
"""

_CREATE_PG_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_case_flags_clinician_ts
    ON case_flags (clinician_id, timestamp);
"""


def _is_postgres_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("postgresql", "postgres")


def _decode_payload(raw: Any) -> dict[str, Any]:
    """Decode a stored payload value into a dict.

    SQLite TEXT columns return JSON strings; PostgreSQL JSONB columns return
    already-parsed dicts via psycopg2.
    """
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


class FlagStore:
    """Persistent storage for CaseFlag entries.

    By default uses a local SQLite file at ``~/.pixelated/flag_store.db``.
    When ``DATABASE_URL`` (or the ``database_url`` constructor arg) points
    to a PostgreSQL DSN, switches to PostgreSQL.
    """

    def __init__(
        self,
        database_url: str | None = None,
        sqlite_path: str | None = None,
    ):
        self._database_url = database_url if database_url is not None else os.environ.get("DATABASE_URL", "")
        self._sqlite_path = sqlite_path or os.path.join(
            os.path.expanduser("~"), ".pixelated", "flag_store.db"
        )
        self._pg = _is_postgres_url(self._database_url) if self._database_url else False
        self._conn: Any = None
        self._connect()

    def _connect(self) -> None:
        if self._pg:
            try:
                psycopg2 = importlib.import_module("psycopg2")  # type: ignore[import-untyped]

                self._conn = psycopg2.connect(self._database_url)
                cur = self._conn.cursor()
                cur.execute(_CREATE_PG_SQL)
                cur.execute(_CREATE_PG_INDEX_SQL)
                self._conn.commit()
                cur.close()
                logger.info("FlagStore connected to PostgreSQL")
            except ImportError:
                logger.warning("psycopg2 not installed; falling back to SQLite")
                self._pg = False
                self._connect_sqlite()
        else:
            self._connect_sqlite()

    def _connect_sqlite(self) -> None:
        os.makedirs(os.path.dirname(self._sqlite_path), exist_ok=True)
        self._conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._conn.execute(_CREATE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()

    def save(self, flag: CaseFlag) -> None:
        """Persist a single CaseFlag."""
        payload = json.dumps(flag.to_dict())
        if self._pg:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO case_flags (clinician_id, flag_type, timestamp, payload) VALUES (%s, %s, %s, %s)",
                (flag.clinician_id, flag.flag_type.value, flag.timestamp.isoformat(), payload),
            )
            self._conn.commit()
            cur.close()
        else:
            self._conn.execute(
                "INSERT INTO case_flags (clinician_id, flag_type, timestamp, payload) VALUES (?, ?, ?, ?)",
                (flag.clinician_id, flag.flag_type.value, flag.timestamp.isoformat(), payload),
            )
            self._conn.commit()

    def load_all(self) -> list[CaseFlag]:
        """Load every persisted CaseFlag (used at engine init for backfill)."""
        cur = self._conn.cursor()
        if self._pg:
            cur.execute("SELECT payload FROM case_flags ORDER BY timestamp")
        else:
            cur.execute("SELECT payload FROM case_flags ORDER BY timestamp")
        rows = cur.fetchall()
        cur.close()
        return [CaseFlag.from_dict(_decode_payload(row[0])) for row in rows]

    def load_for_clinician(self, clinician_id: str) -> list[CaseFlag]:
        """Load all flags for a specific clinician."""
        cur = self._conn.cursor()
        if self._pg:
            cur.execute(
                "SELECT payload FROM case_flags WHERE clinician_id = %s ORDER BY timestamp",
                (clinician_id,),
            )
        else:
            cur.execute(
                "SELECT payload FROM case_flags WHERE clinician_id = ? ORDER BY timestamp",
                (clinician_id,),
            )
        rows = cur.fetchall()
        cur.close()
        return [CaseFlag.from_dict(_decode_payload(row[0])) for row in rows]

    def purge_before(self, cutoff: datetime) -> int:
        """Delete flags older than ``cutoff``. Returns number of rows deleted."""
        cur = self._conn.cursor()
        if self._pg:
            cur.execute(
                "DELETE FROM case_flags WHERE timestamp < %s",
                (cutoff.isoformat(),),
            )
        else:
            cur.execute(
                "DELETE FROM case_flags WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
        deleted = cur.rowcount
        self._conn.commit()
        cur.close()
        return deleted

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
