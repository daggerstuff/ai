"""PostgreSQL DAL for CMS Business Strategy relational tables.

Provides typed query builders for users, permissions, audit logs,
approval workflows, notifications, comments, document sharing,
activity tracking, and system settings. Uses the psycopg2 connection
pool from CMSPostgresConnection.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY BUILDER
# ============================================================================


@dataclass
class PgFilter:
    """A single WHERE clause filter."""

    column: str
    operator: str = "="
    value: Any = None

    def to_sql(self, param_index: int) -> tuple[str, list[Any]]:
        """Return (SQL fragment, params) using positional placeholder."""
        if self.operator.upper() in ("IS NULL", "IS NOT NULL"):
            return f"{self.column} {self.operator}", []
        if self.operator.upper() in ("IN", "NOT IN"):
            if not isinstance(self.value, (list, tuple)) or not self.value:
                return "", []
            placeholders = ", ".join(f"${param_index + i}" for i in range(len(self.value)))
            return f"{self.column} {self.operator} ({placeholders})", list(self.value)
        return f"{self.column} {self.operator} ${param_index}", [self.value]


@dataclass
class PgQuery:
    """Builds a parameterized SQL query."""

    table: str
    columns: list[str] = field(default_factory=lambda: ["*"])
    filters: list[PgFilter] = field(default_factory=list)
    order_by: str = ""
    order_desc: bool = False
    limit: int | None = None
    offset: int | None = None

    def where(self, column: str, operator: str = "=", value: Any = None) -> "PgQuery":
        self.filters.append(PgFilter(column, operator, value))
        return self

    def order(self, column: str, desc: bool = False) -> "PgQuery":
        self.order_by = column
        self.order_desc = desc
        return self

    def page(self, limit: int, offset: int = 0) -> "PgQuery":
        self.limit = limit
        self.offset = offset
        return self

    def select_sql(self) -> tuple[str, list[Any]]:
        """Build a SELECT statement."""
        cols = ", ".join(self.columns)
        sql = f"SELECT {cols} FROM {self.table}"
        params: list[Any] = []
        where_clause, where_params = self._build_where()
        if where_clause:
            sql += f" WHERE {where_clause}"
            params.extend(where_params)
        if self.order_by:
            direction = "DESC" if self.order_desc else "ASC"
            sql += f" ORDER BY {self.order_by} {direction}"
        if self.limit is not None:
            sql += f" LIMIT {self.limit}"
        if self.offset is not None:
            sql += f" OFFSET {self.offset}"
        return sql, params

    def count_sql(self) -> tuple[str, list[Any]]:
        """Build a COUNT(*) statement."""
        sql = f"SELECT COUNT(*) FROM {self.table}"
        params: list[Any] = []
        where_clause, where_params = self._build_where()
        if where_clause:
            sql += f" WHERE {where_clause}"
            params.extend(where_params)
        return sql, params

    def delete_sql(self) -> tuple[str, list[Any]]:
        """Build a DELETE statement."""
        if not self.filters:
            raise ValueError("DELETE requires at least one filter to prevent full table wipe")
        sql = f"DELETE FROM {self.table}"
        params: list[Any] = []
        where_clause, where_params = self._build_where()
        sql += f" WHERE {where_clause}"
        params.extend(where_params)
        return sql, params

    def _build_where(self) -> tuple[str, list[Any]]:
        """Build WHERE clause from filters."""
        if not self.filters:
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        param_idx = 1
        for f in self.filters:
            clause, clause_params = f.to_sql(param_idx)
            if not clause:
                continue
            clauses.append(clause)
            params.extend(clause_params)
            param_idx += len(clause_params)
        return " AND ".join(clauses), params


# ============================================================================
# BASE POSTGRES REPOSITORY
# ============================================================================


class PgBaseRepository:
    """Base repository for a PostgreSQL table."""

    table_name: str = ""

    def __init__(self, pool):
        if not self.table_name:
            raise ValueError(f"{type(self).__name__} must define table_name")
        self._pool = pool
        self.logger = logging.getLogger(f"cms.dal.pg.{self.table_name}")

    def _get_conn(self):
        return self._pool.getconn()

    def _put_conn(self, conn) -> None:
        self._pool.putconn(conn)

    def _execute(self, sql: str, params: list[Any] | None = None) -> int:
        """Execute a write statement. Returns affected row count."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                rowcount = cur.rowcount
            conn.commit()
            return rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def _fetch_one(self, sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        """Fetch a single row as a dict."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                row = cur.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return dict(zip(columns, row, strict=False))
        finally:
            self._put_conn(conn)

    def _fetch_all(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Fetch all rows as list of dicts."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                rows = cur.fetchall()
                if not rows or not cur.description:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=False)) for row in rows]
        finally:
            self._put_conn(conn)

    # ------------------------------------------------------------------
    # GENERIC CRUD
    # ------------------------------------------------------------------

    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        query = PgQuery(self.table_name).where("id", "=", record_id)
        sql, params = query.select_sql()
        return self._fetch_one(sql, params)

    def find_many(
        self,
        filters: list[PgFilter] | None = None,
        order_by: str = "",
        order_desc: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = PgQuery(self.table_name)
        for f in filters or []:
            query.filters.append(f)
        if order_by:
            query.order(order_by, order_desc)
        query.page(limit, offset)
        sql, params = query.select_sql()
        return self._fetch_all(sql, params)

    def count(self, filters: list[PgFilter] | None = None) -> int:
        query = PgQuery(self.table_name)
        for f in filters or []:
            query.filters.append(f)
        sql, params = query.count_sql()
        result = self._fetch_one(sql, params)
        return int(result["count"]) if result else 0

    def delete_by_id(self, record_id: str) -> bool:
        query = PgQuery(self.table_name).where("id", "=", record_id)
        sql, params = query.delete_sql()
        return self._execute(sql, params) > 0


# ============================================================================
# CONCRETE REPOSITORIES
# ============================================================================


class UserRepository(PgBaseRepository):
    table_name = "users"

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM users WHERE email = $1",
            [email],
        )

    def find_by_role(self, role: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("role", "=", role)],
            order_by="name",
            limit=limit,
        )

    def find_active(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("status", "=", "active")],
            order_by="name",
            limit=limit,
        )

    def update_last_login(self, user_id: str) -> None:
        self._execute(
            "UPDATE users SET last_login = NOW(), updated_at = NOW() WHERE id = $1",
            [user_id],
        )


class PermissionRepository(PgBaseRepository):
    table_name = "permissions"

    def find_by_user(self, user_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("user_id", "=", user_id)],
            order_by="resource_type",
        )

    def find_by_resource(self, resource_type: str, resource_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[
                PgFilter("resource_type", "=", resource_type),
                PgFilter("resource_id", "=", resource_id),
            ],
        )

    def grant(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permission_level: str,
        granted_by: str,
        expires_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        sql = """
            INSERT INTO permissions (user_id, resource_type, resource_id, permission_level, granted_by, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, resource_type, resource_id)
            DO UPDATE SET permission_level = $4, granted_by = $5, expires_at = $6, granted_at = NOW()
            RETURNING *
        """
        return self._fetch_one(sql, [user_id, resource_type, resource_id, permission_level, granted_by, expires_at])

    def revoke(self, user_id: str, resource_type: str, resource_id: str) -> bool:
        return (
            self._execute(
                "DELETE FROM permissions WHERE user_id = $1 AND resource_type = $2 AND resource_id = $3",
                [user_id, resource_type, resource_id],
            )
            > 0
        )


class AuditLogRepository(PgBaseRepository):
    table_name = "audit_logs"

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        changes: dict[str, Any] | None = None,
        ip_address: str | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO audit_logs
               (user_id, action, resource_type, resource_id, changes, ip_address, status, error_message)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            [
                user_id,
                action,
                resource_type,
                resource_id,
                json.dumps(changes) if changes else None,
                ip_address,
                status,
                error_message,
            ],
        )

    def find_by_user(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("user_id", "=", user_id)],
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )

    def find_by_resource(self, resource_type: str, resource_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[
                PgFilter("resource_type", "=", resource_type),
                PgFilter("resource_id", "=", resource_id),
            ],
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )


class ApprovalWorkflowRepository(PgBaseRepository):
    table_name = "approval_workflows"

    def find_by_resource_type(self, resource_type: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("resource_type", "=", resource_type), PgFilter("status", "=", "active")],
            order_by="name",
        )


class ApprovalRequestRepository(PgBaseRepository):
    table_name = "approval_requests"

    def find_pending_by_requestor(self, requestor_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("requestor_id", "=", requestor_id), PgFilter("status", "=", "pending")],
            order_by="created_at",
            order_desc=True,
        )

    def find_by_resource(self, resource_type: str, resource_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[
                PgFilter("resource_type", "=", resource_type),
                PgFilter("resource_id", "=", resource_id),
            ],
            order_by="created_at",
            order_desc=True,
        )

    def approve_step(self, step_id: str, reviewed_by: str, comments: str = "") -> bool:
        now = datetime.now(UTC)
        return (
            self._execute(
                """UPDATE approval_steps
               SET status = 'approved', reviewed_by = $1, reviewed_at = $2, review_comments = $3
               WHERE id = $4""",
                [reviewed_by, now, comments, step_id],
            )
            > 0
        )

    def reject_step(self, step_id: str, reviewed_by: str, comments: str = "") -> bool:
        now = datetime.now(UTC)
        return (
            self._execute(
                """UPDATE approval_steps
               SET status = 'rejected', reviewed_by = $1, reviewed_at = $2, review_comments = $3
               WHERE id = $4""",
                [reviewed_by, now, comments, step_id],
            )
            > 0
        )


class NotificationRepository(PgBaseRepository):
    table_name = "notifications"

    def find_unread(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("user_id", "=", user_id), PgFilter("read_at", "IS NULL")],
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )

    def mark_read(self, notification_id: str) -> bool:
        return (
            self._execute(
                "UPDATE notifications SET read_at = NOW() WHERE id = $1",
                [notification_id],
            )
            > 0
        )

    def mark_all_read(self, user_id: str) -> int:
        return self._execute(
            "UPDATE notifications SET read_at = NOW() WHERE user_id = $1 AND read_at IS NULL",
            [user_id],
        )

    def create(
        self,
        user_id: str,
        notif_type: str,
        title: str,
        message: str,
        resource_type: str = "",
        resource_id: str = "",
    ) -> dict[str, Any] | None:
        sql = """
            INSERT INTO notifications (user_id, type, title, message, resource_type, resource_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        return self._fetch_one(sql, [user_id, notif_type, title, message, resource_type, resource_id])


class CommentRepository(PgBaseRepository):
    table_name = "comments"

    def find_by_document(self, document_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("document_id", "=", document_id)],
            order_by="created_at",
            limit=limit,
        )

    def resolve(self, comment_id: str, resolved_by: str) -> bool:
        now = datetime.now(UTC)
        return (
            self._execute(
                "UPDATE comments SET resolved = TRUE, resolved_by = $1, resolved_at = $2 WHERE id = $3",
                [resolved_by, now, comment_id],
            )
            > 0
        )


class DocumentShareRepository(PgBaseRepository):
    table_name = "document_shares"

    def find_by_document(self, document_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("document_id", "=", document_id)],
            order_by="shared_at",
            order_desc=True,
        )

    def find_by_recipient(self, user_id: str) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("shared_with", "=", user_id)],
            order_by="shared_at",
            order_desc=True,
        )


class DocumentActivityRepository(PgBaseRepository):
    table_name = "document_activity"

    def log(
        self,
        document_id: str,
        user_id: str,
        activity_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO document_activity (document_id, user_id, activity_type, metadata)
               VALUES ($1, $2, $3, $4)""",
            [document_id, user_id, activity_type, json.dumps(metadata) if metadata else None],
        )

    def find_by_document(self, document_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("document_id", "=", document_id)],
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )


class DocumentVersionRepository(PgBaseRepository):
    table_name = "document_versions"

    def find_by_document(self, document_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.find_many(
            filters=[PgFilter("document_id", "=", document_id)],
            order_by="version_number",
            order_desc=True,
            limit=limit,
        )

    def create(
        self,
        document_id: str,
        version_number: int,
        title: str,
        content: str,
        created_by: str,
        change_summary: str = "",
    ) -> dict[str, Any] | None:
        sql = """
            INSERT INTO document_versions (document_id, version_number, title, content, created_by, change_summary)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        return self._fetch_one(sql, [document_id, version_number, title, content, created_by, change_summary])


class SystemSettingsRepository(PgBaseRepository):
    table_name = "system_settings"

    def get_value(self, key: str) -> Any:
        result = self._fetch_one("SELECT value FROM system_settings WHERE key = $1", [key])
        return result["value"] if result else None

    def set_value(self, key: str, value: Any, updated_by: str | None = None) -> None:
        self._execute(
            """INSERT INTO system_settings (key, value, updated_by, updated_at)
               VALUES ($1, $2, $3, NOW())
               ON CONFLICT (key) DO UPDATE SET value = $2, updated_by = $3, updated_at = NOW()""",
            [key, json.dumps(value), updated_by],
        )
