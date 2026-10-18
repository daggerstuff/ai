#!/usr/bin/env python3
"""
Consolidated Database Persistence Layer for Pixelated Empathy AI

Provides a single source of truth for all database operations:
- Unified CRUD interface for all data models
- Batch operations with transaction support
- Connection pooling and resource management
- Caching layer for frequently accessed data
- Integration with checkpoint system
- Support for multiple storage backends (SQLite, PostgreSQL, etc.)
- Comprehensive error handling and retry logic
- Migration and schema management
- Performance monitoring and metrics

Usage:
    from persistence import DatabaseManager, PersistenceConfig

    config = PersistenceConfig(database_type="postgresql")

    with DatabaseManager(config) as db:
        # Create a conversation
        conversation = await db.conversations.create({
            "messages": [...],
            "metadata": {...}
        })

        # Batch insert
        await db.conversations.bulk_create(conversations_list)

        # Search with filters
        results = await db.conversations.search(
            filters={"emotion": "sadness"},
            limit=100
        )
"""

import builtins
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

# Type aliases for better readability
ModelT = TypeVar("ModelT")
ID = str | int | uuid.UUID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class QueryOperator(Enum):
    """Query operators for filtering."""

    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


@dataclass
class PersistenceConfig:
    """Configuration for database persistence."""

    # Database connection
    database_type: DatabaseType = DatabaseType.SQLITE
    database_path: str | Path = "/home/vivi/pixelated/ai/database/consolidated.db"

    # For PostgreSQL/MySQL
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    database_name: str | None = None

    # Connection pooling
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # Performance settings
    enable_wal_mode: bool = True  # Write-Ahead Logging for SQLite
    enable_foreign_keys: bool = True
    cache_size_mb: int = 64
    page_size: int = 4096

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 1000

    # Transactions
    isolation_level: str = "READ COMMITTED"
    autocommit: bool = False

    # Retry logic
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Logging and monitoring
    log_queries: bool = False
    log_slow_queries_threshold: float = 1.0  # seconds
    enable_metrics: bool = True

    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.database_type == DatabaseType.SQLITE:
            self.database_path = Path(self.database_path)
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        elif self.database_type in (DatabaseType.POSTGRESQL, DatabaseType.MYSQL):
            if not all([self.host, self.port, self.username, self.database_name]):
                raise ValueError(
                    f"For {self.database_type.value}, host, port, username, and database_name must be provided"
                )


@dataclass
class QueryFilter:
    """Filter for database queries."""

    field: str
    operator: QueryOperator = QueryOperator.EQ
    value: Any = None

    def to_sql(self) -> tuple[str, list[Any]]:
        """Convert filter to SQL clause."""
        placeholders: list[str] = []
        params: list[Any] = []

        # Column names are identifiers and cannot be parameterized; validate
        # them against a strict allowlist before interpolating into SQL.
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.field):
            raise ValueError(f"Invalid SQL field name: {self.field!r}")

        if self.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
            return f"{self.field} {self.operator.value}", []
        if self.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
            if isinstance(self.value, (list, tuple)):
                placeholders = ["?" for _ in self.value]
                params.extend(self.value)
                return (
                    f"{self.field} {self.operator.value} ({', '.join(placeholders)})",
                    params,
                )
            raise ValueError("IN/NOT IN operator requires list or tuple value")
        if isinstance(self.value, (list, tuple)):
            raise ValueError(f"Operator {self.operator.value} requires single value, not list")
        return f"{self.field} {self.operator.value} ?", [self.value]


@dataclass
class QueryOptions:
    """Options for database queries."""

    filters: list[QueryFilter] = field(default_factory=list)
    order_by: str | None = None
    order_desc: bool = False
    limit: int | None = None
    offset: int | None = None
    include_deleted: bool = False

    def to_sql(self) -> tuple[str, list[str], list[Any]]:
        """Convert options to SQL WHERE clause and arguments."""
        where_clauses: list[str] = []
        params: list[Any] = []

        # Add filters
        for query_filter in self.filters:
            clause, filter_params = query_filter.to_sql()
            where_clauses.append(clause)
            params.extend(filter_params)

        # Add deleted filter
        if not self.include_deleted:
            where_clauses.append("deleted_at IS NULL")

        # Build WHERE clause
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Build ORDER BY clause
        order_sql = ""
        if self.order_by:
            # ORDER BY takes an identifier, which cannot be parameterized;
            # reject anything that is not a plain column name.
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.order_by):
                raise ValueError(f"Invalid ORDER BY column: {self.order_by!r}")
            direction = "DESC" if self.order_desc else "ASC"
            order_sql = f"ORDER BY {self.order_by} {direction}"

        # Build LIMIT and OFFSET
        limit_sql = ""
        if self.limit is not None:
            if not isinstance(self.limit, int) or self.limit < 0:
                raise ValueError(f"Invalid LIMIT value: {self.limit!r}")
            limit_sql = f"LIMIT {self.limit}"
            if self.offset is not None:
                if not isinstance(self.offset, int) or self.offset < 0:
                    raise ValueError(f"Invalid OFFSET value: {self.offset!r}")
                limit_sql += f" OFFSET {self.offset}"

        return where_sql, [order_sql, limit_sql], params


@dataclass
class BatchResult:
    """Result of batch operations."""

    success_count: int
    failed_count: int
    errors: list[tuple[int, str]] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def total_count(self) -> int:
        """Total items processed."""
        return self.success_count + self.failed_count

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.success_count / self.total_count) * 100


@dataclass
class DatabaseMetrics:
    """Metrics for database operations."""

    queries_executed: int = 0
    queries_failed: int = 0
    total_query_time: float = 0.0
    slow_queries: int = 0

    operations_created: int = 0
    operations_read: int = 0
    operations_updated: int = 0
    operations_deleted: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    connection_pool_size: int = 0
    active_connections: int = 0

    last_reset: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        avg_query_time = self.total_query_time / self.queries_executed if self.queries_executed > 0 else 0.0

        return {
            "queries_executed": self.queries_executed,
            "queries_failed": self.queries_failed,
            "query_failure_rate": (
                self.queries_failed / self.queries_executed * 100 if self.queries_executed > 0 else 0.0
            ),
            "total_query_time": round(self.total_query_time, 3),
            "avg_query_time": round(avg_query_time, 3),
            "slow_queries": self.slow_queries,
            "operations_created": self.operations_created,
            "operations_read": self.operations_read,
            "operations_updated": self.operations_updated,
            "operations_deleted": self.operations_deleted,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "connection_pool_size": self.connection_pool_size,
            "active_connections": self.active_connections,
            "last_reset": self.last_reset.isoformat(),
        }


class ConnectionPool:
    """Thread-safe connection pool for database connections."""

    def __init__(self, config: PersistenceConfig, create_connection: Callable):
        self.config = config
        self.create_connection = create_connection
        self._pool: list[sqlite3.Connection] = []
        self._in_use: set = set()
        self._created_at: dict = {}
        self._lock = threading.Lock()
        self.metrics = DatabaseMetrics()

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        with self._lock:
            # Try to get an existing connection
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.execute("SELECT 1")
                    self._in_use.add(conn)
                    self.metrics.active_connections = len(self._in_use)
                    return conn
                except sqlite3.Error:
                    # Connection is stale, discard it
                    conn.close()

            # No available connections, create a new one
            if len(self._in_use) < self.config.pool_size:
                conn = self.create_connection()
                self._in_use.add(conn)
                self._created_at[id(conn)] = time.time()
                self.metrics.connection_pool_size = len(self._in_use) + len(self._pool)
                self.metrics.active_connections = len(self._in_use)
                return conn

            # Pool is full, wait for a connection
            raise TimeoutError("Connection pool exhausted")

    def release_connection(self, conn: sqlite3.Connection):
        """Release a connection back to the pool."""
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)

                # Check if connection should be recycled
                age_seconds = time.time() - self._created_at.get(id(conn), 0)
                if age_seconds > self.config.pool_recycle:
                    conn.close()
                    self._created_at.pop(id(conn), None)
                else:
                    self._pool.append(conn)

                self.metrics.active_connections = len(self._in_use)

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool + list(self._in_use):
                with suppress(Exception):
                    conn.close()
            self._pool.clear()
            self._in_use.clear()
            self._created_at.clear()


class SimpleCache:
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._access_order.remove(key)
                return None

            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)

            return value

    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            # Update if key exists
            if key in self._cache:
                self._cache[key] = (value, time.time())
                self._access_order.remove(key)
                self._access_order.append(key)
                return

            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            # Add new item
            self._cache[key] = (value, time.time())
            self._access_order.append(key)

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class BaseModelRepository[ModelT](ABC):
    """Abstract base class for model repositories."""

    def __init__(self, db_manager: "DatabaseManager", table_name: str):
        self.db_manager = db_manager
        self.table_name = table_name
        self.logger = logging.getLogger(f"persistence.{table_name}")

    @abstractmethod
    async def create(self, data: dict[str, Any]) -> ModelT:
        """Create a new record."""
        raise NotImplementedError()

    @abstractmethod
    async def get(self, entity_id: ID) -> ModelT | None:
        """Get a record by ID."""
        raise NotImplementedError()

    @abstractmethod
    async def update(self, entity_id: ID, data: dict[str, Any]) -> ModelT | None:
        """Update a record."""
        raise NotImplementedError()

    @abstractmethod
    async def delete(self, entity_id: ID, soft_delete: bool = True) -> bool:
        """Delete a record."""
        raise NotImplementedError()

    @abstractmethod
    async def list(self, options: QueryOptions | None = None) -> list[ModelT]:
        """List records with optional filtering."""
        raise NotImplementedError()

    @abstractmethod
    async def search(
        self,
        query: str,
        fields: builtins.list[str] | None = None,
        limit: int = 100,
    ) -> builtins.list[ModelT]:
        """Full-text search across records."""
        raise NotImplementedError()

    @abstractmethod
    async def bulk_create(self, items: builtins.list[dict[str, Any]]) -> BatchResult:
        """Bulk create records."""
        raise NotImplementedError()

    @abstractmethod
    async def bulk_update(self, updates: builtins.list[tuple[ID, dict[str, Any]]]) -> BatchResult:
        """Bulk update records."""
        raise NotImplementedError()

    @abstractmethod
    async def bulk_delete(self, ids: builtins.list[ID], soft_delete: bool = True) -> BatchResult:
        """Bulk delete records."""
        raise NotImplementedError()

    @abstractmethod
    async def count(self, options: QueryOptions | None = None) -> int:
        """Count records."""
        raise NotImplementedError()

    async def exists(self, entity_id: ID) -> bool:
        """Check if a record exists."""
        return (await self.get(entity_id)) is not None


class ConversationRepository(BaseModelRepository[dict]):
    """Repository for conversation data."""

    def __init__(self, db_manager: "DatabaseManager"):
        super().__init__(db_manager, "conversations")

    async def create(self, data: dict[str, Any]) -> dict:
        """Create a new conversation."""
        sql = """
        INSERT INTO conversations (
            conversation_id,
            messages,
            metadata,
            tier,
            processing_status,
            emotion_tags,
            crisis_detected,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        timestamp = datetime.now(UTC).isoformat()

        params = [
            conversation_id,
            json.dumps(data.get("messages", [])),
            json.dumps(data.get("metadata", {})),
            data.get("tier", "standard"),
            data.get("processing_status", "pending"),
            json.dumps(data.get("emotion_tags", [])),
            data.get("crisis_detected", False),
            timestamp,
            timestamp,
        ]

        await self.db_manager.execute(sql, params)
        self.db_manager.metrics.operations_created += 1

        # Cache the newly created conversation
        cache_key = f"conversation:{conversation_id}"
        data["conversation_id"] = conversation_id
        self.db_manager.cache.set(cache_key, data)

        return data

    async def get(self, entity_id: ID) -> dict | None:
        """Get a conversation by ID."""
        cache_key = f"conversation:{entity_id}"

        # Try cache first
        cached = self.db_manager.cache.get(cache_key)
        if cached is not None:
            self.db_manager.metrics.cache_hits += 1
            return cached

        self.db_manager.metrics.cache_misses += 1

        sql = """
        SELECT conversation_id, messages, metadata, tier, processing_status,
               emotion_tags, crisis_detected, created_at, updated_at
        FROM conversations
        WHERE conversation_id = ? AND deleted_at IS NULL
        """

        result = await self.db_manager.fetch_one(sql, [str(entity_id)])
        if result:
            self.db_manager.metrics.operations_read += 1
            conversation = self._row_to_dict(result)
            self.db_manager.cache.set(cache_key, conversation)
            return conversation

        return None

    async def update(self, entity_id: ID, data: dict[str, Any]) -> dict | None:
        """Update a conversation."""
        build_clauses = []
        params = []

        allowed_keys = {
            "messages",
            "metadata",
            "tier",
            "processing_status",
            "emotion_tags",
            "crisis_detected",
            "updated_at",
            "deleted_at",
        }

        # Build update clauses dynamically
        for key, value in data.items():
            if key not in allowed_keys:
                raise ValueError(f"Invalid update key: {key}")
            if key in {"messages", "metadata", "emotion_tags"}:
                build_clauses.append(f'"{key}" = ?')
                params.append(json.dumps(value))
            elif key not in ("conversation_id", "created_at"):
                build_clauses.append(f'"{key}" = ?')
                params.append(value)

        if not build_clauses:
            return await self.get(entity_id)

        build_clauses.append('"updated_at" = ?')
        params.append(datetime.now(UTC).isoformat())
        params.append(str(entity_id))

        set_clause = ", ".join(build_clauses)
        # Validate set_clause format to prevent SQL injection
        if not re.match(r'^("[a-zA-Z0-9_]+" = \?(?:,\s*)?)+$', set_clause):
            raise ValueError(f"Invalid SET clause format: {set_clause}")

        sql_template = """
        UPDATE conversations
        SET {set_clause}
        WHERE conversation_id = ? AND deleted_at IS NULL
        """
        sql = sql_template.format(set_clause=set_clause)

        await self.db_manager.execute(sql, params)
        self.db_manager.metrics.operations_updated += 1

        # Invalidate cache
        cache_key = f"conversation:{entity_id}"
        self.db_manager.cache.invalidate(cache_key)

        return await self.get(entity_id)

    async def delete(self, entity_id: ID, soft_delete: bool = True) -> bool:
        """Delete a conversation."""
        if soft_delete:
            sql = """
            UPDATE conversations
            SET deleted_at = ?, updated_at = ?
            WHERE conversation_id = ? AND deleted_at IS NULL
            """
            timestamp = datetime.now(UTC).isoformat()
            params = [timestamp, timestamp, str(entity_id)]
        else:
            sql = """
            DELETE FROM conversations
            WHERE conversation_id = ? AND deleted_at IS NULL
            """
            params = [str(entity_id)]

        result = await self.db_manager.execute(sql, params)
        self.db_manager.metrics.operations_deleted += 1

        # Invalidate cache
        cache_key = f"conversation:{entity_id}"
        self.db_manager.cache.invalidate(cache_key)

        return result > 0

    async def list(self, options: QueryOptions | None = None) -> list[dict]:
        """List conversations with optional filtering."""
        options = options or QueryOptions()

        where_sql, [order_sql, limit_sql], params = options.to_sql()

        sql = f"""
        SELECT conversation_id, messages, metadata, tier, processing_status,
               emotion_tags, crisis_detected, created_at, updated_at
        FROM conversations
        {where_sql}
        {order_sql}
        {limit_sql}
        """

        results = await self.db_manager.fetch_all(sql, params)
        self.db_manager.metrics.operations_read += len(results)

        return [self._row_to_dict(row) for row in results]

    async def search(
        self,
        query: str,
        fields: builtins.list[str] | None = None,
        limit: int = 100,
    ) -> builtins.list[dict]:
        """Search conversations by text."""
        # For SQLite, use LIKE searches
        search_fields = fields or ["messages", "metadata"]

        # Column names are identifiers and cannot be parameterized; validate
        # them against a strict allowlist before interpolating into SQL.
        for search_field in search_fields:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", search_field):
                raise ValueError(f"Invalid search field: {search_field!r}")

        like_clauses = [f"{search_field} LIKE ?" for search_field in search_fields]
        search_term = f"%{query}%"
        params: list[Any] = [search_term] * len(like_clauses)

        sql = f"""
        SELECT conversation_id, messages, metadata, tier, processing_status,
               emotion_tags, crisis_detected, created_at, updated_at
        FROM conversations
        WHERE {" OR ".join(like_clauses)}
        AND deleted_at IS NULL
        LIMIT ?
        """
        params.append(limit)

        results = await self.db_manager.fetch_all(sql, params)
        self.db_manager.metrics.operations_read += len(results)

        return [self._row_to_dict(row) for row in results]

    async def bulk_create(self, items: builtins.list[dict[str, Any]]) -> BatchResult:
        """Bulk create conversations."""
        start_time = time.time()
        success_count = 0
        failed_count = 0
        errors = []

        sql = """
        INSERT INTO conversations (
            conversation_id, messages, metadata, tier, processing_status,
            emotion_tags, crisis_detected, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        timestamp = datetime.now(UTC).isoformat()

        for item in items:
            if "conversation_id" not in item:
                item["conversation_id"] = str(uuid.uuid4())

        params_generator = (
            [
                item.get("conversation_id"),
                json.dumps(item.get("messages", [])),
                json.dumps(item.get("metadata", {})),
                item.get("tier", "standard"),
                item.get("processing_status", "pending"),
                json.dumps(item.get("emotion_tags", [])),
                item.get("crisis_detected", False),
                timestamp,
                timestamp,
            ]
            for item in items
        )

        try:
            row_count = await self.db_manager.executemany(sql, params_generator)
            success_count = row_count

            # Cache the new conversations
            for item in items:
                cache_key = f"conversation:{item['conversation_id']}"
                self.db_manager.cache.set(cache_key, item)
        except Exception as e:
            failed_count = len(items)
            errors.append((0, str(e)))
            self.logger.error(f"Failed to bulk create conversations: {e}")

        self.db_manager.metrics.operations_created += success_count
        total_time = time.time() - start_time

        return BatchResult(success_count, failed_count, errors, total_time)

    async def bulk_update(self, updates: builtins.list[tuple[ID, dict[str, Any]]]) -> BatchResult:
        """Bulk update conversations."""
        start_time = time.time()
        success_count = 0
        failed_count = 0
        errors = []

        async for idx, (conversation_id, data) in self._async_enumerate(updates):
            try:
                # Invalidate cache
                cache_key = f"conversation:{conversation_id}"
                self.db_manager.cache.invalidate(cache_key)

                result = await self.update(conversation_id, data)
                if result:
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append((idx, "Conversation not found"))

            except Exception as e:
                failed_count += 1
                errors.append((idx, str(e)))
                self.logger.error(f"Failed to update conversation {conversation_id}: {e}")

        total_time = time.time() - start_time

        return BatchResult(success_count, failed_count, errors, total_time)

    async def bulk_delete(self, ids: builtins.list[ID], soft_delete: bool = True) -> BatchResult:
        """Bulk delete conversations.

        ⚡ Bolt: Optimized N+1 queries by batching SELECT and UPDATE/DELETE operations.
        """
        start_time = time.time()
        success_count = 0
        failed_count = 0
        errors = []
        chunk_size = 900
        timestamp = datetime.now(UTC).isoformat()

        # Convert to list to support chunking
        ids_list = list(ids)

        for i in range(0, len(ids_list), chunk_size):
            chunk_ids = ids_list[i : i + chunk_size]
            chunk_start_idx = i
            found_ids = None

            try:
                placeholders = ",".join(["?"] * len(chunk_ids))
                # 1. Identify which conversations actually exist
                if soft_delete:
                    select_sql = (
                        "SELECT conversation_id FROM conversations "
                        f"WHERE conversation_id IN ({placeholders}) AND deleted_at IS NULL"
                    )
                else:
                    select_sql = f"SELECT conversation_id FROM conversations WHERE conversation_id IN ({placeholders})"

                rows = await self.db_manager.fetch_all(select_sql, [str(cid) for cid in chunk_ids])
                found_ids = {str(row[0]) for row in rows}

                # Invalidate cache for found items
                for cid in found_ids:
                    self.db_manager.cache.invalidate(f"conversation:{cid}")

                # Record failures for any IDs not found in the database
                for j, cid in enumerate(chunk_ids):
                    if str(cid) not in found_ids:
                        failed_count += 1
                        errors.append((chunk_start_idx + j, "Conversation not found"))

                if not found_ids:
                    continue

                # 2. Perform the batch update/delete
                valid_ids_list = list(found_ids)
                valid_placeholders = ",".join(["?"] * len(valid_ids_list))

                if soft_delete:
                    update_sql = (
                        "UPDATE conversations "
                        "SET deleted_at = ?, updated_at = ? "
                        f"WHERE conversation_id IN ({valid_placeholders})"
                    )
                    params = [timestamp, timestamp] + [str(cid) for cid in valid_ids_list]
                else:
                    update_sql = f"DELETE FROM conversations WHERE conversation_id IN ({valid_placeholders})"
                    params = [str(cid) for cid in valid_ids_list]

                await self.db_manager.execute(update_sql, params)
                self.db_manager.metrics.operations_deleted += len(valid_ids_list)
                success_count += len(valid_ids_list)

            except Exception as e:
                # If a chunk fails, record all of its items as failures
                for j, _ in enumerate(chunk_ids):
                    failed_count += 1
                    errors.append((chunk_start_idx + j, str(e)))
                self.logger.error(f"Failed to bulk delete conversation chunk starting at {i}: {e}")

        total_time = time.time() - start_time
        return BatchResult(success_count, failed_count, errors, total_time)

    async def count(self, options: QueryOptions | None = None) -> int:
        """Count conversations."""
        options = options or QueryOptions()
        where_sql, _, params = options.to_sql()

        sql = f"SELECT COUNT(*) FROM conversations {where_sql}"

        result = await self.db_manager.fetch_one(sql, params)
        return int(result[0]) if result else 0

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict:
        """Convert database row to dictionary."""
        return {
            "conversation_id": row[0],
            "messages": json.loads(row[1]) if row[1] else [],
            "metadata": json.loads(row[2]) if row[2] else {},
            "tier": row[3],
            "processing_status": row[4],
            "emotion_tags": json.loads(row[5]) if row[5] else [],
            "crisis_detected": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }

    async def _async_enumerate(self, iterable: builtins.list[Any]) -> AsyncIterator[tuple[int, Any]]:
        """Async enumerate helper."""
        for idx, item in enumerate(iterable):
            yield idx, item


class DatabaseManager:
    """Consolidated database manager for all persistence operations."""

    def __init__(self, config: PersistenceConfig | None = None):
        self.config = config or PersistenceConfig()
        self._pool: ConnectionPool | None = None
        self._initialized = False
        self._lock = threading.Lock()

        # Initialize cache
        if self.config.enable_caching:
            self.cache = SimpleCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
        else:
            self.cache = SimpleCache(max_size=0)  # Disabled

        # Initialize metrics
        self.metrics = DatabaseMetrics()

        # Set up logger
        self.logger = logging.getLogger("persistence.database_manager")
        self.logger.setLevel(logging.INFO)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def initialize(self):
        """Initialize database connection and schema."""
        with self._lock:
            if self._initialized:
                return

            self.logger.info(f"Initializing {self.config.database_type.value} database...")

            # Create connection pool
            if self.config.database_type == DatabaseType.SQLITE:
                self._pool = ConnectionPool(self.config, self._create_sqlite_connection)

            # Initialize schema
            self._initialize_schema()

            self._initialized = True
            self.logger.info("Database initialized successfully")

    def close(self):
        """Close database connections and cleanup."""
        with self._lock:
            if not self._initialized:
                return

            self.logger.info("Closing database connections...")

            if self._pool:
                self._pool.close_all()
                self._pool = None

            self._initialized = False
            self.logger.info("Database connections closed")

    def _create_sqlite_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimizations."""
        conn = sqlite3.connect(str(self.config.database_path), check_same_thread=False)

        # Set optimization pragmas
        if self.config.enable_wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")

        if self.config.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")

        conn.execute(f"PRAGMA cache_size=-{int(self.config.cache_size_mb) * 1024}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA page_size={int(self.config.page_size)}")

        return conn

    def _initialize_schema(self):
        """Initialize database schema."""
        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            # Create conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    metadata TEXT,
                    tier TEXT NOT NULL DEFAULT 'standard',
                    processing_status TEXT NOT NULL DEFAULT 'pending',
                    emotion_tags TEXT,
                    crisis_detected BOOLEAN NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                )
            """)

            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_tier
                ON conversations(tier)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_status
                ON conversations(processing_status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_created_at
                ON conversations(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_deleted_at
                ON conversations(deleted_at)
            """)

            # Create full-text search virtual table for SQLite
            if self.config.database_type == DatabaseType.SQLITE:
                try:
                    conn.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                        USING fts5(conversation_id, messages, metadata,
                        content=conversations, content_rowid=rowid)
                    """)
                    # Note: content_rowid mapping would need rowid to be
                    # added to conversations table
                except sqlite3.Error:
                    # FTS5 might not be available
                    self.logger.warning("FTS5 not available, falling back to LIKE searches")

            conn.commit()
            self.logger.info("Database schema initialized")

        finally:
            self._pool.release_connection(conn)

    # Repository accessors
    @property
    def conversations(self) -> ConversationRepository:
        """Get the conversations repository."""
        return ConversationRepository(self)

    # Database operations
    async def executemany(self, sql: str, params_list: builtins.list[Sequence[Any]] | Any) -> int:
        """Execute a SQL statement multiple times with different parameters and return affected row count."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            start_time = time.time()
            cursor = conn.executemany(str(sql), params_list)
            row_count = cursor.rowcount
            conn.commit()

            execution_time = time.time() - start_time

            self.metrics.queries_executed += 1
            self.metrics.total_query_time += execution_time

            if self.config.log_queries:
                self.logger.debug(f"Executed many: {sql[:100]}... in {execution_time:.3f}s")

            if execution_time > self.config.log_slow_queries_threshold:
                self.metrics.slow_queries += 1
                self.logger.warning(f"Slow query (executemany, {execution_time:.3f}s): {sql[:200]}...")

            return int(row_count)

        except Exception as e:
            self.metrics.queries_failed += 1
            conn.rollback()
            self.logger.error(f"Executemany query failed: {e}\nSQL: {sql[:200]}")
            raise
        finally:
            self._pool.release_connection(conn)

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Execute a SQL statement and return affected row count."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            start_time = time.time()
            cursor = conn.execute(str(sql), list(params) if params is not None else [])
            row_count = cursor.rowcount
            conn.commit()

            execution_time = time.time() - start_time

            self.metrics.queries_executed += 1
            self.metrics.total_query_time += execution_time

            if self.config.log_queries:
                self.logger.debug(f"Executed: {sql[:100]}... in {execution_time:.3f}s")

            if execution_time > self.config.log_slow_queries_threshold:
                self.metrics.slow_queries += 1
                self.logger.warning(f"Slow query ({execution_time:.3f}s): {sql[:200]}...")

            return int(row_count)

        except Exception as e:
            self.metrics.queries_failed += 1
            conn.rollback()
            self.logger.error(f"Query failed: {e}\nSQL: {sql[:200]}")
            raise
        finally:
            self._pool.release_connection(conn)

    async def fetch_one(self, sql: str, params: Sequence[Any] | None = None) -> tuple[Any, ...] | None:
        """Fetch a single row from the database."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            start_time = time.time()
            cursor = conn.execute(str(sql), list(params) if params is not None else [])
            row = cursor.fetchone()

            execution_time = time.time() - start_time

            self.metrics.queries_executed += 1
            self.metrics.total_query_time += execution_time

            if self.config.log_queries:
                self.logger.debug(f"Fetched: {sql[:100]}... in {execution_time:.3f}s")

            return row if row is None else tuple(row)

        except Exception as e:
            self.metrics.queries_failed += 1
            self.logger.error(f"Fetch failed: {e}\nSQL: {sql[:200]}")
            raise
        finally:
            self._pool.release_connection(conn)

    async def fetch_all(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        """Fetch all rows from the database."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            start_time = time.time()
            cursor = conn.execute(str(sql), list(params) if params is not None else [])
            rows = cursor.fetchall()

            execution_time = time.time() - start_time

            self.metrics.queries_executed += 1
            self.metrics.total_query_time += execution_time

            if self.config.log_queries:
                self.logger.debug(f"Fetched {len(rows)} rows in {execution_time:.3f}s")

            return [tuple(row) for row in rows]

        except Exception as e:
            self.metrics.queries_failed += 1
            self.logger.error(f"Fetch failed: {e}\nSQL: {sql[:200]}")
            raise
        finally:
            self._pool.release_connection(conn)

    async def execute_transaction(self, operations: list[tuple[str, Sequence[Any] | None]]) -> list[int]:
        """Execute multiple operations in a single transaction."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            start_time = time.time()
            row_counts: list[int] = []

            # Begin transaction
            conn.execute("BEGIN")

            for sql, params in operations:
                cursor = conn.execute(
                    str(sql),
                    list(params) if params is not None else [],
                )
                row_counts.append(int(cursor.rowcount))

            conn.commit()

            execution_time = time.time() - start_time

            self.metrics.queries_executed += len(operations)
            self.metrics.total_query_time += execution_time

            self.logger.debug(f"Transaction completed: {len(operations)} operations in {execution_time:.3f}s")

            return row_counts

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise
        finally:
            self._pool.release_connection(conn)

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise RuntimeError("Database not initialized")
        conn = self._pool.get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release_connection(conn)

    def get_metrics(self) -> DatabaseMetrics:
        """Get database metrics."""
        if self._pool:
            self.metrics.connection_pool_size = len(self._pool._pool) + len(self._pool._in_use)
            self.metrics.active_connections = len(self._pool._in_use)

        return self.metrics

    def reset_metrics(self):
        """Reset database metrics."""
        self.metrics = DatabaseMetrics()

    def clear_cache(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def vacuum_database(self):
        """Vacuum the database (SQLite only)."""
        if self.config.database_type == DatabaseType.SQLITE:
            if self._pool is None:
                raise RuntimeError("Database not initialized")
            conn = self._pool.get_connection()
            try:
                conn.execute("VACUUM")
                self.logger.info("Database vacuumed")
            finally:
                self._pool.release_connection(conn)
        else:
            self.logger.warning("VACUUM only supported for SQLite")


# Global database manager instance (held in a mutable container to avoid module-level global mutation)
_db_state: dict[str, DatabaseManager | None] = {"manager": None}
_db_lock = threading.Lock()


def get_database_manager(config: PersistenceConfig | None = None) -> DatabaseManager:
    """Get or create the global database manager instance."""
    if _db_state["manager"] is None:
        with _db_lock:
            if _db_state["manager"] is None:
                _db_state["manager"] = DatabaseManager(config)
                _db_state["manager"].initialize()

    manager = _db_state["manager"]
    assert manager is not None, "DatabaseManager failed to initialize"
    return manager


def close_database_manager():
    """Close the global database manager instance."""
    with _db_lock:
        if _db_state["manager"] is not None:
            _db_state["manager"].close()
            _db_state["manager"] = None


# Convenience functions for common operations
async def create_conversation(data: dict[str, Any]) -> dict:
    """Create a conversation."""
    db = get_database_manager()
    return await db.conversations.create(data)


async def get_conversation(conversation_id: ID) -> dict | None:
    """Get a conversation by ID."""
    db = get_database_manager()
    return await db.conversations.get(conversation_id)


async def update_conversation(conversation_id: ID, data: dict[str, Any]) -> dict | None:
    """Update a conversation."""
    db = get_database_manager()
    return await db.conversations.update(conversation_id, data)


async def delete_conversation(conversation_id: ID, soft_delete: bool = True) -> bool:
    """Delete a conversation."""
    db = get_database_manager()
    return await db.conversations.delete(conversation_id, soft_delete)


async def list_conversations(
    filters: list[QueryFilter] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict]:
    """List conversations."""
    db = get_database_manager()
    options = QueryOptions(filters=filters or [], limit=limit, offset=offset)
    return await db.conversations.list(options)


async def search_conversations(query: str, fields: list[str] | None = None, limit: int = 100) -> list[dict]:
    """Search conversations."""
    db = get_database_manager()
    return await db.conversations.search(query, fields, limit)


async def bulk_create_conversations(items: list[dict[str, Any]]) -> BatchResult:
    """Bulk create conversations."""
    db = get_database_manager()
    return await db.conversations.bulk_create(items)


def get_database_metrics() -> dict[str, Any]:
    """Get database metrics."""
    db = get_database_manager()
    return db.get_metrics().to_dict()


def reset_database_metrics():
    """Reset database metrics."""
    db = get_database_manager()
    db.reset_metrics()


# Example usage (commented out)
"""
if __name__ == "__main__":
    import asyncio

    async def example_usage():
        # Create a conversation
        conversation = await create_conversation({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "metadata": {"source": "test"},
            "tier": "standard"
        })
        print(f"Created conversation: {conversation['conversation_id']}")

        # Get the conversation
        retrieved = await get_conversation(conversation["conversation_id"])
        print(f"Retrieved: {retrieved['messages']}")

        # List conversations
        conversations = await list_conversations(limit=10)
        print(f"Total conversations: {len(conversations)}")

        # Get metrics
        metrics = get_database_metrics()
        print(f"Database metrics: {metrics}")

        # Cleanup
        await delete_conversation(conversation["conversation_id"])
        close_database_manager()

    asyncio.run(example_usage())
"""
