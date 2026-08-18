import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient


class MCPMongoDBClient:
    """
    MongoDB client for MCP server persistent storage.
    Supports both sync and async operations if needed.
    """

    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self.logger = logging.getLogger(__name__)

        # Initialize sync client for basic operations
        self.sync_client = MongoClient(uri)
        self.db = self.sync_client[db_name]

        # Initialize async client for higher performance
        self.async_client = AsyncIOMotorClient(uri)
        self.async_db = self.async_client[db_name]

        self.logger.info(f"MCP MongoDB client connected to {db_name}")

    def get_collection(self, name: str):
        """Get a sync collection."""
        return self.db[name]

    def get_async_collection(self, name: str):
        """Get an async collection."""
        return self.async_db[name]

    async def find_one(self, collection: str, query: dict[str, Any]) -> dict[str, Any] | None:
        """Find a single document asynchronously."""
        # Type hint for basedpyright which sometimes misses motor's return types
        return await self.async_db[collection].find_one(query)

    async def find_many(
        self, collection: str, query: dict[str, Any], sort: list[tuple] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Find multiple documents asynchronously with optional sorting and limit."""
        cursor = self.async_db[collection].find(query)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        # Using length=None returns all documents matching the query
        return await cursor.to_list(length=None)

    async def update_one(self, collection: str, query: dict[str, Any], update: dict[str, Any]) -> bool:
        """Update a single document asynchronously."""
        result = await self.async_db[collection].update_one(query, update)
        return getattr(result, "modified_count", 0) > 0

    async def insert_one(self, collection: str, document: dict[str, Any]) -> str:
        """Insert a document asynchronously."""
        result = await self.async_db[collection].insert_one(document)
        inserted_id = getattr(result, "inserted_id", None)
        return str(inserted_id) if inserted_id else ""
