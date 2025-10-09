from fastapi import FastAPI, HTTPException, Query, Depends, Security, status, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import os

from security.api_authentication import AuthenticationSystem, UserRole, PermissionLevel, User, APIKey
from security.fastapi_auth_middleware import AuthenticationDependencies, bearer_scheme, api_key_header

# Initialize Authentication System (use a strong secret key in production)
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "super-secret-key-for-dev")
auth_system = AuthenticationSystem(AUTH_SECRET_KEY)
auth_deps = AuthenticationDependencies(auth_system)

# Create a test API key for demonstration purposes
# In a real application, API keys would be managed securely (e.g., via admin interface)
TEST_API_KEY, _ = auth_system.create_api_key(
    "test_dataset_api_key",
    [PermissionLevel.READ, PermissionLevel.WRITE],
    expires_in_days=365
)
print(f"DEBUG: Test API Key for /datasets endpoint: {TEST_API_KEY}")

app = FastAPI(title="Dataset Access API", description="API for accessing and querying datasets.")

DATABASE_URL = "/home/vivi/pixelated/ai/data/conversation_system.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row  # This enables name-based access to columns
    return conn

class DatasetMetadata(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    row_count: int
    columns: List[Dict[str, Any]] # Changed to list of dicts for more detail
    created_at: str = "N/A"
    updated_at: str = "N/A"

class QueryResult(BaseModel):
    data: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int



async def get_api_key_user(
    api_key: str = Security(api_key_header)
) -> Dict[str, Any]:
    """Dedicated API key authentication dependency"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate API key using the authentication system
    api_key_obj = auth_system.validate_api_key(api_key)
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return {"username": api_key_obj.name, "scopes": api_key_obj.permissions, "auth_type": "api_key"}

async def get_current_active_user_or_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header)
):
    """Modified authentication function that supports both user tokens and API keys"""
    # First try to get authenticated user from request state (JWT token auth)
    user = getattr(request.state, 'authenticated_user', None)
    if user:
        return {"username": user.username, "scopes": user.permissions, "auth_type": "user_token"}
    
    # If no user token, try API key authentication
    if api_key:
        api_key_obj = auth_system.validate_api_key(api_key)
        if api_key_obj:
            return {"username": api_key_obj.name, "scopes": api_key_obj.permissions, "auth_type": "api_key"}
    
    # Check if there's an authenticated API key in request state (from middleware)
    api_key_obj = getattr(request.state, 'authenticated_api_key', None)
    if api_key_obj:
        return {"username": api_key_obj.name, "scopes": api_key_obj.permissions, "auth_type": "api_key"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required - provide either user token or API key"
    )

@app.get("/datasets", response_model=List[DatasetMetadata])
async def list_datasets(current_auth_entity: Any = Depends(get_current_active_user_or_api_key)):
    """List all available datasets (tables in the database)."""
    datasets = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table["name"]
            if table_name == "sqlite_sequence": # Skip internal SQLite table
                continue

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            columns = []
            for col in columns_info:
                columns.append({
                    "name": col["name"],
                    "type": col["type"],
                    "notnull": bool(col["notnull"]),
                    "pk": bool(col["pk"])
                })

            datasets.append(
                DatasetMetadata(
                    id=table_name,
                    name=table_name.replace("_", " ").title(),
                    description=f"Data from the {table_name} table.",
                    row_count=row_count,
                    columns=columns
                )
            )
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()
    return datasets

@app.get("/datasets/{dataset_id}/metadata", response_model=DatasetMetadata)
async def get_dataset_metadata(dataset_id: str, current_auth_entity: Any = Depends(get_current_active_user_or_api_key)):
    """Get metadata (schema) for a specific dataset (table)."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if table exists and get row count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?;", (dataset_id,))
        table_exists = cursor.fetchone()
        if not table_exists:
            raise HTTPException(status_code=404, detail="Dataset (table) not found")
        
        cursor.execute(f"SELECT COUNT(*) FROM {dataset_id}")
        row_count = cursor.fetchone()[0]

        # Get columns
        cursor.execute(f"PRAGMA table_info({dataset_id});")
        columns_info = cursor.fetchall()
        columns = []
        for col in columns_info:
            columns.append({
                "name": col["name"],
                "type": col["type"],
                "notnull": bool(col["notnull"]),
                "pk": bool(col["pk"])
            })

        return DatasetMetadata(
            id=dataset_id,
            name=dataset_id.replace("_", " ").title(),
            description=f"Data from the {dataset_id} table.",
            row_count=row_count,
            columns=columns
        )
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()

@app.post("/datasets/{dataset_id}/query", response_model=QueryResult)
async def query_dataset(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Number of items per page"),
    filters: Optional[Dict[str, Any]] = None, # Example: {"column_name": "value"}
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key)
):
    """Query data from a specific dataset (table) with optional filters and pagination."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?;", (dataset_id,))
        table_exists = cursor.fetchone()
        if not table_exists:
            raise HTTPException(status_code=404, detail="Dataset (table) not found")

        # Build WHERE clause for filters
        where_clause = ""
        params = []
        if filters:
            filter_clauses = []
            for col, val in filters.items():
                # Basic sanitization: check if column exists in table
                cursor.execute(f"PRAGMA table_info({dataset_id});")
                columns_info = [c["name"] for c in cursor.fetchall()]
                if col not in columns_info:
                    raise HTTPException(status_code=400, detail=f"Invalid filter column: {col}")
                
                filter_clauses.append(f"{col} = ?")
                params.append(val)
            where_clause = " WHERE " + " AND ".join(filter_clauses)

        # Get total rows matching filters
        count_query = f"SELECT COUNT(*) FROM {dataset_id}{where_clause}"
        cursor.execute(count_query, params)
        total_rows = cursor.fetchone()[0]

        # Get data with pagination
        offset = (page - 1) * page_size
        data_query = f"SELECT * FROM {dataset_id}{where_clause} LIMIT ? OFFSET ?"
        cursor.execute(data_query, params + [page_size, offset])
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(dict(row))

        return QueryResult(data=results, total_rows=total_rows, page=page, page_size=page_size)

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)