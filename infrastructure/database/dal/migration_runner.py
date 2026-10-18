"""Database migration runner for CMS Business Strategy system.

Executes numbered migration files (.sql for PostgreSQL, .js for MongoDB)
in order, tracking applied migrations in a PostgreSQL table.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

MIGRATION_FILE_RE = re.compile(r"^(\d+)_\w+\.(sql|js)$")


class MigrationRunner:
    """Runs database migrations in sequence."""

    def __init__(self, postgres_pool, mongo_uri: str = "", mongo_db: str = ""):
        self._pg_pool = postgres_pool
        self._mongo_uri = mongo_uri
        self._mongo_db = mongo_db

    # ------------------------------------------------------------------
    # BOOTSTRAP
    # ------------------------------------------------------------------

    def _ensure_migration_table(self) -> None:
        """Create the migration tracking table if it doesn't exist."""
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS _migrations (
                        id SERIAL PRIMARY KEY,
                        version INT UNIQUE NOT NULL,
                        filename VARCHAR(255) NOT NULL,
                        applied_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            conn.commit()
        finally:
            self._pg_pool.putconn(conn)

    def _get_applied_versions(self) -> set[int]:
        """Return the set of already-applied migration version numbers."""
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version FROM _migrations ORDER BY version")
                return {row[0] for row in cur.fetchall()}
        finally:
            self._pg_pool.putconn(conn)

    def _record_migration(self, version: int, filename: str) -> None:
        """Record a successfully applied migration."""
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO _migrations (version, filename) VALUES (%s, %s)",
                    [version, filename],
                )
            conn.commit()
        finally:
            self._pg_pool.putconn(conn)

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    def _discover_migrations(self) -> list[tuple[int, str, Path]]:
        """Find all migration files sorted by version number."""
        if not MIGRATIONS_DIR.exists():
            logger.warning("Migrations directory not found: %s", MIGRATIONS_DIR)
            return []

        migrations: list[tuple[int, str, Path]] = []
        for path in sorted(MIGRATIONS_DIR.iterdir()):
            match = MIGRATION_FILE_RE.match(path.name)
            if match:
                version = int(match.group(1))
                migrations.append((version, path.name, path))

        migrations.sort(key=lambda m: m[0])
        return migrations

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    def _run_sql_migration(self, path: Path) -> None:
        """Execute a .sql migration file against PostgreSQL."""
        sql = path.read_text()
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Applied SQL migration: %s", path.name)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pg_pool.putconn(conn)

    def _run_js_migration(self, path: Path) -> None:
        """Execute a .js migration file against MongoDB via mongosh."""
        if not self._mongo_uri:
            raise RuntimeError("MongoDB URI required for JS migrations")

        result = subprocess.run(
            ["mongosh", self._mongo_uri, "--eval", path.read_text()],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MongoDB migration {path.name} failed: {result.stderr}")
        logger.info("Applied JS migration: %s", path.name)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run_pending(self) -> list[str]:
        """Run all pending migrations. Returns list of applied filenames."""
        self._ensure_migration_table()
        applied = self._get_applied_versions()
        discovered = self._discover_migrations()

        pending = [(v, f, p) for v, f, p in discovered if v not in applied]
        if not pending:
            logger.info("No pending migrations")
            return []

        applied_names: list[str] = []
        for version, filename, path in pending:
            logger.info("Running migration %s ...", filename)
            if path.suffix == ".sql":
                self._run_sql_migration(path)
            elif path.suffix == ".js":
                self._run_js_migration(path)
            self._record_migration(version, filename)
            applied_names.append(filename)

        logger.info("Applied %d migrations", len(applied_names))
        return applied_names

    def status(self) -> dict[str, Any]:
        """Return migration status summary."""
        self._ensure_migration_table()
        applied = self._get_applied_versions()
        discovered = self._discover_migrations()

        pending = [(v, f) for v, f, _ in discovered if v not in applied]
        return {
            "applied_count": len(applied),
            "pending_count": len(pending),
            "pending": [f"v{v} {f}" for v, f in pending],
            "latest_applied": max(applied) if applied else None,
        }
