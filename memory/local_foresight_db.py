from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, LifoQueue

from .local_foresight_schema import LocalForesightSchemaManager


class LocalForesightDatabase:
    """Connection leasing and schema management for the local SQLite store."""

    def __init__(self, db_path: str, pool_size: int = 8) -> None:
        self.db_path = db_path
        self._schema_lock = threading.Lock()
        self._pool = LifoQueue(maxsize=pool_size)

        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def lease(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = self._pool.get_nowait()
        except Empty:
            conn = self._create_connection()
        try:
            with conn:
                yield conn
        finally:
            try:
                self._pool.put_nowait(conn)
            except Exception:
                conn.close()

    def ensure_schema(self) -> None:
        with self._schema_lock, self.lease() as conn:
            LocalForesightSchemaManager.ensure_schema(conn)

    def health_details(self) -> dict[str, object]:
        conn = self._create_connection()
        try:
            quick_check_row = conn.execute("PRAGMA quick_check(1)").fetchone()
            quick_check_ok = bool(quick_check_row and quick_check_row[0] == "ok")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
            return {
                "db_ready": quick_check_ok,
                "db_writable": True,
                "db_quick_check": quick_check_row[0] if quick_check_row else "unknown",
            }
        finally:
            conn.close()

    def close(self) -> None:
        while True:
            try:
                self._pool.get_nowait().close()
            except Empty:
                return
