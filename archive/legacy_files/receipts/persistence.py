"""R5/INT-5: Receipt-ledger persistence + audit export.

The in-memory :class:`~ai.receipts.receipt.Ledger` (R1) grows without bound.
This module adds SQLite (stdlib, zero deps) and PostgreSQL (optional driver)
stores behind a common interface, a :class:`PersistentLedger` that writes
through to a store, and JSONL audit export.

Store choice is driven by ``RECEIPT_LEDGER_DB_PATH`` (SQLite file) or
``RECEIPT_LEDGER_POSTGRES_URL`` (fallback to ``DATABASE_URL``). If neither is
set, :func:`get_persistent_ledger` returns a plain in-memory ledger so
existing callers keep working unchanged.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from ai.receipts.receipt import Ledger, ReceiptEnvelope

logger = logging.getLogger(__name__)

GENESIS_PREV_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    prev_hash           TEXT NOT NULL,
    model_fingerprint   TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,
    output_hash         TEXT NOT NULL,
    fhe_ciphertext_hash TEXT NOT NULL,
    receipt_hash        TEXT PRIMARY KEY,
    appended_at         TEXT NOT NULL
)
"""


@runtime_checkable
class ReceiptStore(Protocol):
    """Common persistence interface for receipt ledgers."""

    def append(self, receipt: ReceiptEnvelope) -> None: ...

    def load_all(self) -> list[ReceiptEnvelope]: ...

    def count(self) -> int: ...


class SQLiteReceiptStore:
    """SQLite-backed receipt store (stdlib only, thread-safe)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = str(path or os.environ.get("RECEIPT_LEDGER_DB_PATH") or "receipt_ledger.db")
        self._lock = threading.Lock()
        parent = Path(self._path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def append(self, receipt: ReceiptEnvelope) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.prev_hash,
                    receipt.model_fingerprint,
                    receipt.prompt_hash,
                    receipt.output_hash,
                    receipt.fhe_ciphertext_hash,
                    receipt.receipt_hash,
                    receipt.receipt_hash,  # appended_at column unused placeholder
                ),
            )

    def load_all(self) -> list[ReceiptEnvelope]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM receipts ORDER BY rowid").fetchall()
        receipts: list[ReceiptEnvelope] = []
        for row in rows:
            receipts.append(
                ReceiptEnvelope.from_dict(
                    {
                        "prev_hash": row["prev_hash"],
                        "model_fingerprint": row["model_fingerprint"],
                        "prompt_hash": row["prompt_hash"],
                        "output_hash": row["output_hash"],
                        "fhe_ciphertext_hash": row["fhe_ciphertext_hash"],
                        "receipt_hash": row["receipt_hash"],
                    }
                )
            )
        return receipts

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0])


class PostgresReceiptStore:
    """PostgreSQL-backed receipt store.

    Requires the optional ``psycopg`` (v3) or ``psycopg2`` driver; a clear
    ``ImportError`` is raised on construction when neither is installed.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or os.environ.get("RECEIPT_LEDGER_POSTGRES_URL") or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError("PostgresReceiptStore requires RECEIPT_LEDGER_POSTGRES_URL or DATABASE_URL")
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError:
            try:
                import psycopg2  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "PostgresReceiptStore requires 'psycopg' or 'psycopg2'; install one or use "
                    "SQLiteReceiptStore (RECEIPT_LEDGER_DB_PATH)"
                ) from exc
            self._driver = "psycopg2"
        else:
            self._driver = "psycopg"
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self):
        if self._driver == "psycopg":
            import psycopg

            return psycopg.connect(self._dsn)
        import psycopg2

        return psycopg2.connect(self._dsn)

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA)
            conn.commit()

    def append(self, receipt: ReceiptEnvelope) -> None:
        with self._lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO receipts VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (receipt_hash) DO NOTHING",
                    (
                        receipt.prev_hash,
                        receipt.model_fingerprint,
                        receipt.prompt_hash,
                        receipt.output_hash,
                        receipt.fhe_ciphertext_hash,
                        receipt.receipt_hash,
                        receipt.receipt_hash,
                    ),
                )
            conn.commit()

    def load_all(self) -> list[ReceiptEnvelope]:
        with self._lock, self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM receipts ORDER BY rowid")
            rows: list[tuple[object, ...]] = list(cur.fetchall())
        receipts: list[ReceiptEnvelope] = []
        for row in rows:
            receipts.append(
                ReceiptEnvelope.from_dict(
                    {
                        "prev_hash": str(row[0]),
                        "model_fingerprint": str(row[1]),
                        "prompt_hash": str(row[2]),
                        "output_hash": str(row[3]),
                        "fhe_ciphertext_hash": str(row[4]),
                        "receipt_hash": str(row[5]),
                    }
                )
            )
        return receipts

    def count(self) -> int:
        with self._lock, self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM receipts")
            row = cur.fetchone()
            if row is None:
                return 0
            return int(row[0])


class PersistentLedger(Ledger):
    """Ledger that writes through to a :class:`ReceiptStore`."""

    def __init__(self, store: ReceiptStore) -> None:
        super().__init__()
        self._store = store

    @property
    def store(self) -> ReceiptStore:
        return self._store

    def append(self, receipt: ReceiptEnvelope) -> None:
        self._store.append(receipt)
        super().append(receipt)

    def restore(self) -> int:
        """Rebuild in-memory state from the store; returns restored count."""
        self._receipts = []
        self._leaves = []
        for receipt in self._store.load_all():
            super().append(receipt)
        return len(self._receipts)

    def export_audit(self, path: str | os.PathLike[str] | None = None) -> str:
        """Export every receipt as JSONL; returns the serialized payload."""
        lines = "\n".join(receipt.to_json() for receipt in self._receipts)
        if lines:
            lines += "\n"
        if path is not None:
            Path(path).write_text(lines, encoding="utf-8")
        return lines


def get_persistent_ledger() -> Ledger:
    """Env-gated ledger factory.

    - ``RECEIPT_LEDGER_POSTGRES_URL`` / ``DATABASE_URL`` set → Postgres store
    - ``RECEIPT_LEDGER_DB_PATH`` set (or writable default) → SQLite store
    - otherwise → plain in-memory :class:`Ledger`
    """
    pg_url = os.environ.get("RECEIPT_LEDGER_POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if pg_url:
        try:
            ledger = PersistentLedger(PostgresReceiptStore(pg_url))
        except (ImportError, ValueError) as exc:
            logger.warning("Postgres receipt store unavailable (%s); falling back", exc)
        else:
            return ledger
    return PersistentLedger(SQLiteReceiptStore())
