import contextlib
import logging
import os
import re
import sqlite3
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from pydantic import BaseModel

from configs.api_authentication import (
    AuthenticationSystem,
    PermissionLevel,
    UserRole,
)
from configs.fastapi_auth_middleware import (
    AuthenticationDependencies,
    api_key_header,
)

logger = logging.getLogger(__name__)

# Initialize Authentication System (use a strong secret key in production)
AUTH_SECRET_KEY = os.environ.get("AUTH_SECRET_KEY")
if not AUTH_SECRET_KEY:
    raise ValueError(
        "AUTH_SECRET_KEY environment variable is missing or empty. It must be configured securely in production."
    )
auth_system = AuthenticationSystem(AUTH_SECRET_KEY)
auth_deps = AuthenticationDependencies(auth_system)

# Test API key creation is disabled in production
# To enable for development, set CREATE_TEST_API_KEY=true
if os.getenv("CREATE_TEST_API_KEY", "false").lower() == "true":
    API_KEY_EXPIRY_DAYS = int(os.getenv("API_KEY_EXPIRY_DAYS", "365"))
    TEST_API_KEY, _ = auth_system.create_api_key(
        "test_dataset_api_key",
        [PermissionLevel.READ, PermissionLevel.WRITE],
        expires_in_days=API_KEY_EXPIRY_DAYS,
    )
    logger.warning("Test API key created for development purposes only")

app = FastAPI(title="Dataset Access API", description="API for accessing and querying datasets.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is missing or empty. It must be configured securely in production."
    )


def validate_identifier(identifier: str) -> str:
    """
    Validates that the identifier contains only alphanumeric characters and underscores.
    Returns the identifier if valid, raises HTTPException otherwise.
    This prevents SQL injection by disallowing special characters.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        raise HTTPException(status_code=400, detail=f"Invalid identifier format: {identifier}")
    return identifier


def safe_execute_query(cursor, query_template: str, params=None, **kwargs):
    """
    Executes a query safely. Any dynamic table names or schema-level identifiers
    must be passed as keywords to format the query_template, and they must be
    validated identifiers.
    """
    for key, val in kwargs.items():
        if key == "where":
            # Allow where clause to contain alphanumeric, underscores, spaces, ?, =, and "AND", and quotes
            if not re.match(r"^[a-zA-Z0-9_\s?=\"',]*$", val):
                raise ValueError(f"Invalid where clause: {val}")
        elif not re.match(r"^[a-zA-Z0-9_]+$", val):
            raise ValueError(f"Invalid identifier: {val}")
    query = query_template.format(**kwargs)
    if params is not None:
        return cursor.execute(query, params)
    return cursor.execute(query)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row  # This enables name-based access to columns
    return conn


class DatasetMetadata(BaseModel):
    id: str
    name: str
    description: str | None = None
    row_count: int
    columns: list[dict[str, Any]]  # Changed to list of dicts for more detail
    created_at: str = "N/A"
    updated_at: str = "N/A"


class QueryResult(BaseModel):
    data: list[dict[str, Any]]
    total_rows: int
    page: int
    page_size: int


async def get_api_key_user(api_key: str = Security(api_key_header)) -> dict[str, Any]:
    """Dedicated API key authentication dependency"""
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")

    # Validate API key using the authentication system
    api_key_obj = auth_system.authenticate_api_key(api_key)
    if not api_key_obj:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return {
        "username": api_key_obj.name,
        "scopes": api_key_obj.permissions,
        "auth_type": "api_key",
    }


async def get_current_active_user_or_api_key(request: Request, api_key: str | None = Depends(api_key_header)):
    """Modified authentication function that supports both user tokens and API keys"""
    # First try to get authenticated user from request state (JWT token auth)
    user = getattr(request.state, "authenticated_user", None)
    if user:
        user_role = getattr(user, "role", None)
        user_scopes: list[str] = []
        if user_role:
            # role could be str or UserRole enum
            if isinstance(user_role, str):
                with contextlib.suppress(ValueError):
                    user_role = UserRole(user_role)
            user_scopes = auth_system.role_permissions.get(user_role, [])
        user_scopes = getattr(user, "permissions", user_scopes)
        return {
            "username": user.username,
            "scopes": user_scopes,
            "auth_type": "user_token",
        }

    # If no user token, try API key authentication
    if api_key:
        api_key_obj = auth_system.authenticate_api_key(api_key)
        if api_key_obj:
            return {
                "username": api_key_obj.name,
                "scopes": api_key_obj.permissions,
                "auth_type": "api_key",
            }

    # Check if there's an authenticated API key in request state (from middleware)
    api_key_obj = getattr(request.state, "authenticated_api_key", None)
    if api_key_obj:
        return {
            "username": api_key_obj.name,
            "scopes": api_key_obj.permissions,
            "auth_type": "api_key",
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required - provide either user token or API key",
    )


def require_read_scope(
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key),
) -> Any:
    """Dependency to enforce READ scope on endpoints."""
    if PermissionLevel.READ not in current_auth_entity.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to perform READ operations",
        )
    return current_auth_entity


@app.get("/datasets", response_model=list[DatasetMetadata])
async def list_datasets(
    current_auth_entity: Any = Depends(require_read_scope),
):
    """List all available datasets (tables in the database)."""

    datasets = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table["name"]
            if table_name == "sqlite_sequence":  # Skip internal SQLite table
                continue

            # Validate table name format
            try:
                safe_table_name = validate_identifier(table_name)
            except HTTPException:
                continue

            # Get row count
            safe_execute_query(cursor, 'SELECT COUNT(*) FROM "{table}"', table=safe_table_name)
            row_count = cursor.fetchone()[0]

            # Get columns
            safe_execute_query(cursor, 'PRAGMA table_info("{table}")', table=safe_table_name)
            columns_info = cursor.fetchall()
            columns = []
            for col in columns_info:
                columns.append(
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "notnull": bool(col["notnull"]),
                        "pk": bool(col["pk"]),
                    }
                )

            datasets.append(
                DatasetMetadata(
                    id=table_name,
                    name=table_name.replace("_", " ").title(),
                    description=f"Data from the {table_name} table.",
                    row_count=row_count,
                    columns=columns,
                )
            )
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    finally:
        if conn:
            conn.close()
    return datasets


@app.get("/datasets/{dataset_id}/metadata", response_model=DatasetMetadata)
async def get_dataset_metadata(
    dataset_id: str,
    current_auth_entity: Any = Depends(require_read_scope),
):
    """Get metadata (schema) for a specific dataset (table)."""

    conn = None
    try:
        # Validate input format immediately to prevent any SQL injection attempts
        validate_identifier(dataset_id)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if table exists and get row count
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
            (dataset_id,),
        )
        table_row = cursor.fetchone()
        if not table_row:
            raise HTTPException(status_code=404, detail="Dataset (table) not found")

        safe_table_name = validate_identifier(table_row["name"])

        safe_execute_query(cursor, 'SELECT COUNT(*) FROM "{table}"', table=safe_table_name)
        row_count = cursor.fetchone()[0]

        # Get columns
        safe_execute_query(cursor, 'PRAGMA table_info("{table}")', table=safe_table_name)
        columns_info = cursor.fetchall()
        columns = []
        for col in columns_info:
            columns.append(
                {
                    "name": col["name"],
                    "type": col["type"],
                    "notnull": bool(col["notnull"]),
                    "pk": bool(col["pk"]),
                }
            )

        return DatasetMetadata(
            id=dataset_id,
            name=dataset_id.replace("_", " ").title(),
            description=f"Data from the {dataset_id} table.",
            row_count=row_count,
            columns=columns,
        )
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    finally:
        if conn:
            conn.close()


@app.post("/datasets/{dataset_id}/query", response_model=QueryResult)
async def query_dataset(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Number of items per page"),
    filters: dict[str, Any] | None = None,  # Example: {"column_name": "value"}
    current_auth_entity: Any = Depends(require_read_scope),
):
    """
    Query data from a specific dataset (table) with optional filters and pagination.
    """

    conn = None
    try:
        # Validate input format immediately
        validate_identifier(dataset_id)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
            (dataset_id,),
        )
        table_row = cursor.fetchone()
        if not table_row:
            raise HTTPException(status_code=404, detail="Dataset (table) not found")

        safe_table_name = validate_identifier(table_row["name"])

        # Get valid columns for the table to validate filters
        safe_execute_query(cursor, 'PRAGMA table_info("{table}")', table=safe_table_name)
        valid_columns = {c["name"] for c in cursor.fetchall()}

        # Build WHERE clause for filters
        where_clause = ""
        params = []
        if filters:
            filter_clauses = []
            for col, val in filters.items():
                # Check if column exists in table (sanitization allow-list)
                if col not in valid_columns:
                    raise HTTPException(status_code=400, detail=f"Invalid filter column: {col}")

                # Since col is verified to be in valid_columns (from DB metadata),
                # it is safe to use in the query as a column identifier.
                filter_clauses.append(f'"{col}" = ?')
                params.append(val)

            if filter_clauses:
                where_clause = " WHERE " + " AND ".join(filter_clauses)

        # Get total rows matching filters
        safe_execute_query(
            cursor, 'SELECT COUNT(*) FROM "{table}"{where}', params=params, table=safe_table_name, where=where_clause
        )
        total_rows = cursor.fetchone()[0]

        # Get data with pagination
        offset = (page - 1) * page_size
        safe_execute_query(
            cursor,
            'SELECT * FROM "{table}"{where} LIMIT ? OFFSET ?',
            params=[*params, page_size, offset],
            table=safe_table_name,
            where=where_clause,
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(dict(row))

        return QueryResult(data=results, total_rows=total_rows, page=page, page_size=page_size)

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
