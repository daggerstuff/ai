"""CMS Business Strategy multi-database connection manager.

Unified connection management for MongoDB (Atlas), PostgreSQL (Supabase),
and Redis. Provides connection pooling, health checks, and graceful
shutdown for the CMS business strategy system.

Reuses patterns from:
  - ai/api/mcp_server/integration/mongodb_client.py (Motor async MongoDB)
  - ai/api/techdeck_integration/integration/redis_client.py (sync Redis)
  - ai/infrastructure/database/migrations/001_initial_schema.sql (PostgreSQL)
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import redis
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

logger = logging.getLogger(__name__)


@dataclass
class CMSDatabaseConfig:
    """Connection configuration for all three CMS databases."""

    # MongoDB Atlas
    mongo_uri: str = ""
    mongo_db_name: str = "pixelated-business-strategy"

    # PostgreSQL (Supabase)
    postgres_uri: str = ""
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 5

    # Redis
    redis_url: str = ""
    redis_db: int = 0

    @classmethod
    def from_env(cls) -> "CMSDatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            mongo_uri=os.getenv("CMS_MONGODB_URI", ""),
            mongo_db_name=os.getenv("CMS_MONGODB_DB", "pixelated-business-strategy"),
            postgres_uri=os.getenv("CMS_POSTGRES_URI", ""),
            redis_url=os.getenv("CMS_REDIS_URL", ""),
            redis_db=int(os.getenv("CMS_REDIS_DB", "0")),
        )


class CMSMongoConnection:
    """Async MongoDB connection with health checking."""

    def __init__(self, uri: str, db_name: str):
        self._uri = uri
        self._db_name = db_name
        self._client: AsyncIOMotorClient | None = None
        self._sync_client: MongoClient | None = None
        self._db = None
        self._sync_db = None

    async def connect(self) -> None:
        """Initialize async and sync MongoDB clients."""
        if self._client is not None:
            return

        self._client = AsyncIOMotorClient(self._uri)
        self._db = self._client[self._db_name]

        self._sync_client = MongoClient(self._uri)
        self._sync_db = self._sync_client[self._db_name]

        logger.info(f"MongoDB connected to {self._db_name}")

    @property
    def db(self):
        """Async database handle."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db

    @property
    def sync_db(self):
        """Sync database handle for admin/migration operations."""
        if self._sync_db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._sync_db

    def collection(self, name: str):
        """Get an async collection by name."""
        return self.db[name]

    def sync_collection(self, name: str):
        """Get a sync collection by name."""
        return self.sync_db[name]

    async def health_check(self) -> dict[str, Any]:
        """Verify MongoDB connectivity and return status."""
        if self._client is None:
            return {"status": "disconnected", "error": "Not initialized"}

        try:
            await self._client.admin.command("ping")
            return {
                "status": "healthy",
                "database": self._db_name,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def close(self) -> None:
        """Close MongoDB connections."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None

        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
            self._sync_db = None

        logger.info("MongoDB connections closed")


class CMSRedisConnection:
    """Redis connection for CMS caching, collaboration, and rate limiting."""

    def __init__(self, url: str, db: int = 0):
        self._url = url
        self._db = db
        self._client: redis.Redis | None = None

    def connect(self) -> None:
        """Initialize Redis client with connection pooling."""
        if self._client is not None:
            return

        self._client = redis.from_url(
            self._url,
            db=self._db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self._client.ping()
        logger.info(f"Redis connected (DB {self._db})")

    @property
    def client(self) -> redis.Redis:
        """Raw Redis client for direct operations."""
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    def health_check(self) -> dict[str, Any]:
        """Verify Redis connectivity and return status."""
        if self._client is None:
            return {"status": "disconnected", "error": "Not initialized"}

        try:
            start = datetime.now(UTC)
            self._client.ping()
            latency_ms = (datetime.now(UTC) - start).total_seconds() * 1000

            raw_info = self._client.info()
            # redis-py's sync info() returns dict[str, Any] but stubs use
            # ResponseT = Union[Awaitable[Any], Any], so we narrow explicitly
            info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
            connected_clients = info.get("connected_clients", 0)
            used_memory = info.get("used_memory_human", "unknown")
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "connected_clients": connected_clients,
                "used_memory_human": used_memory,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")


class CMSPostgresConnection:
    """PostgreSQL (Supabase) connection manager.

    Uses psycopg2 for synchronous operations. Async support via
    asyncpg can be added later when the DAL layer is built (Phase 2).
    """

    def __init__(self, uri: str, pool_size: int = 10, max_overflow: int = 5):
        self._uri = uri
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool = None

    def connect(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if self._pool is not None:
            return

        try:
            import importlib

            psycopg2_pool = importlib.import_module("psycopg2.pool")
            ThreadedConnectionPool = psycopg2_pool.ThreadedConnectionPool

            self._pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=self._pool_size,
                dsn=self._uri,
            )
            logger.info(f"PostgreSQL pool initialized (size {self._pool_size})")
        except ImportError:
            logger.warning(
                "psycopg2 not installed — PostgreSQL operations unavailable. Install with: pip install psycopg2-binary"
            )
            raise

    @property
    def pool(self):
        """Access the connection pool."""
        if self._pool is None:
            raise RuntimeError("PostgreSQL not connected. Call connect() first.")
        return self._pool

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def put_connection(self, conn) -> None:
        """Return a connection to the pool."""
        self.pool.putconn(conn)

    def health_check(self) -> dict[str, Any]:
        """Verify PostgreSQL connectivity."""
        if self._pool is None:
            return {"status": "disconnected", "error": "Not initialized"}

        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                return {
                    "status": "healthy",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            finally:
                self.pool.putconn(conn)
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def close(self) -> None:
        """Close all PostgreSQL connections."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("PostgreSQL connections closed")


class CMSConnectionManager:
    """Unified connection manager for the CMS Business Strategy system.

    Manages lifecycle of MongoDB, PostgreSQL, and Redis connections.
    Provides a single health-check endpoint and graceful shutdown.
    """

    def __init__(self, config: CMSDatabaseConfig | None = None):
        self._config = config or CMSDatabaseConfig.from_env()
        self._mongo: CMSMongoConnection | None = None
        self._postgres: CMSPostgresConnection | None = None
        self._redis: CMSRedisConnection | None = None
        self._initialized = False

    @property
    def mongo(self) -> CMSMongoConnection:
        """MongoDB connection handler."""
        if self._mongo is None:
            raise RuntimeError("MongoDB not initialized")
        return self._mongo

    @property
    def postgres(self) -> CMSPostgresConnection:
        """PostgreSQL connection handler."""
        if self._postgres is None:
            raise RuntimeError("PostgreSQL not initialized")
        return self._postgres

    @property
    def redis(self) -> CMSRedisConnection:
        """Redis connection handler."""
        if self._redis is None:
            raise RuntimeError("Redis not initialized")
        return self._redis

    async def initialize(self) -> None:
        """Initialize all database connections."""
        if self._initialized:
            return

        config = self._config

        if config.mongo_uri:
            self._mongo = CMSMongoConnection(config.mongo_uri, config.mongo_db_name)
            await self._mongo.connect()
        else:
            logger.warning("CMS_MONGODB_URI not set — MongoDB unavailable")

        if config.postgres_uri:
            self._postgres = CMSPostgresConnection(
                config.postgres_uri,
                pool_size=config.postgres_pool_size,
                max_overflow=config.postgres_max_overflow,
            )
            self._postgres.connect()
        else:
            logger.warning("CMS_POSTGRES_URI not set — PostgreSQL unavailable")

        if config.redis_url:
            self._redis = CMSRedisConnection(config.redis_url, db=config.redis_db)
            self._redis.connect()
        else:
            logger.warning("CMS_REDIS_URL not set — Redis unavailable")

        self._initialized = True
        logger.info("CMS connection manager initialized")

    async def health_check(self) -> dict[str, Any]:
        """Aggregate health status from all connected databases."""
        results: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if self._mongo is not None:
            results["mongodb"] = await self._mongo.health_check()
        else:
            results["mongodb"] = {"status": "not_configured"}

        if self._postgres is not None:
            results["postgresql"] = self._postgres.health_check()
        else:
            results["postgresql"] = {"status": "not_configured"}

        if self._redis is not None:
            results["redis"] = self._redis.health_check()
        else:
            results["redis"] = {"status": "not_configured"}

        overall = "healthy"
        for db_status in [results["mongodb"]["status"], results["postgresql"]["status"], results["redis"]["status"]]:
            if db_status == "unhealthy":
                overall = "degraded"
                break
            if db_status == "not_configured" and overall == "healthy":
                overall = "partial"

        results["overall"] = overall
        return results

    async def close(self) -> None:
        """Gracefully close all database connections."""
        if self._mongo is not None:
            await self._mongo.close()
            self._mongo = None

        if self._postgres is not None:
            self._postgres.close()
            self._postgres = None

        if self._redis is not None:
            self._redis.close()
            self._redis = None

        self._initialized = False
        logger.info("CMS connection manager shut down")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_manager: CMSConnectionManager | None = None


async def get_cms_connection_manager(config: CMSDatabaseConfig | None = None) -> CMSConnectionManager:
    """Get or initialize the global CMS connection manager."""
    global _manager
    if _manager is None:
        _manager = CMSConnectionManager(config)
        await _manager.initialize()
    return _manager


async def close_cms_connection_manager() -> None:
    """Shut down the global CMS connection manager."""
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None
