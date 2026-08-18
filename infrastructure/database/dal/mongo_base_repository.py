"""Async MongoDB base repository for CMS Business Strategy collections.

Provides CRUD, text search, bulk operations, and count with
Motor async client. Concrete repositories extend this with
collection-specific query helpers.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase


class MongoBaseRepository:
    """Async CRUD repository for a single MongoDB collection."""

    collection_name: str = ""

    def __init__(self, db: AsyncIOMotorDatabase):
        if not self.collection_name:
            raise ValueError(f"{type(self).__name__} must define collection_name")
        self._db = db
        self._collection: AsyncIOMotorCollection | None = None
        self.logger = logging.getLogger(f"cms.dal.mongo.{self.collection_name}")

    @property
    def collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            self._collection = self._db[self.collection_name]
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, document: dict[str, Any]) -> dict[str, Any]:
        """Insert a document, adding createdAt/updatedAt if missing."""
        now = datetime.now(UTC)
        document.setdefault("createdAt", now)
        document.setdefault("updatedAt", now)

        result = await self.collection.insert_one(document)
        document["_id"] = result.inserted_id
        self.logger.debug("Created %s", result.inserted_id)
        return document

    async def get(self, doc_id: str, id_field: str = "_id") -> dict[str, Any] | None:
        """Fetch a single document by ID field (default MongoDB _id)."""
        if id_field == "_id":
            from bson import ObjectId

            try:
                query = {"_id": ObjectId(doc_id)}
            except Exception:
                return None
        else:
            query = {id_field: doc_id}

        return await self.collection.find_one(query)

    async def update(self, doc_id: str, data: dict[str, Any], id_field: str = "_id") -> dict[str, Any] | None:
        """Update fields on a document. Returns the updated document."""
        data["updatedAt"] = datetime.now(UTC)
        # Never overwrite _id or createdAt
        data.pop("_id", None)
        data.pop("createdAt", None)

        if id_field == "_id":
            from bson import ObjectId

            try:
                query = {"_id": ObjectId(doc_id)}
            except Exception:
                return None
        else:
            query = {id_field: doc_id}

        await self.collection.update_one(query, {"$set": data})
        return await self.collection.find_one(query)

    async def delete(self, doc_id: str, id_field: str = "_id") -> bool:
        """Hard-delete a document by ID field. Returns True if deleted."""
        if id_field == "_id":
            from bson import ObjectId

            try:
                query = {"_id": ObjectId(doc_id)}
            except Exception:
                return False
        else:
            query = {id_field: doc_id}

        result = await self.collection.delete_one(query)
        deleted = result.deleted_count > 0
        if deleted:
            self.logger.debug("Deleted %s=%s", id_field, doc_id)
        return deleted

    # ------------------------------------------------------------------
    # LIST / FILTER
    # ------------------------------------------------------------------

    async def find_many(
        self,
        filter_query: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find documents with optional filter, sort, and pagination."""
        cursor = self.collection.find(filter_query or {})
        if sort:
            cursor = cursor.sort(sort)
        cursor = cursor.skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def count(self, filter_query: dict[str, Any] | None = None) -> int:
        """Count documents matching a filter."""
        return await self.collection.count_documents(filter_query or {})

    # ------------------------------------------------------------------
    # TEXT SEARCH
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        filter_query: dict[str, Any] | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Full-text search using the collection's text index."""
        combined: dict[str, Any] = {"$text": {"$search": query}}
        if filter_query:
            combined = {"$and": [combined, filter_query]}

        cursor = (
            self.collection.find(combined, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # BULK OPERATIONS
    # ------------------------------------------------------------------

    async def bulk_create(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert multiple documents at once."""
        now = datetime.now(UTC)
        for doc in documents:
            doc.setdefault("createdAt", now)
            doc.setdefault("updatedAt", now)

        result = await self.collection.insert_many(documents)
        self.logger.debug("Bulk created %d documents", len(result.inserted_ids))
        return documents

    async def bulk_update(self, updates: list[tuple[str, dict[str, Any]]], id_field: str = "_id") -> int:
        """Update multiple documents. Returns count of updated documents."""
        now = datetime.now(UTC)
        updated_count = 0

        for doc_id, data in updates:
            data["updatedAt"] = now
            data.pop("_id", None)
            data.pop("createdAt", None)

            if id_field == "_id":
                from bson import ObjectId

                try:
                    query = {"_id": ObjectId(doc_id)}
                except Exception:
                    continue
            else:
                query = {id_field: doc_id}

            result = await self.collection.update_one(query, {"$set": data})
            updated_count += result.modified_count

        self.logger.debug("Bulk updated %d documents", updated_count)
        return updated_count

    async def bulk_delete(self, doc_ids: list[str], id_field: str = "_id") -> int:
        """Delete multiple documents. Returns count of deleted documents."""
        if not doc_ids:
            return 0

        if id_field == "_id":
            from bson import ObjectId

            object_ids = []
            for doc_id in doc_ids:
                try:
                    object_ids.append(ObjectId(doc_id))
                except Exception:
                    continue
            query: dict[str, Any] = {"_id": {"$in": object_ids}}
        else:
            query = {id_field: {"$in": doc_ids}}

        result = await self.collection.delete_many(query)
        self.logger.debug("Bulk deleted %d documents", result.deleted_count)
        return result.deleted_count

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    async def exists(self, doc_id: str, id_field: str = "_id") -> bool:
        """Check if a document exists."""
        if id_field == "_id":
            from bson import ObjectId

            try:
                query = {"_id": ObjectId(doc_id)}
            except Exception:
                return False
        else:
            query = {id_field: doc_id}

        return await self.collection.count_documents(query, limit=1) > 0

    async def get_by_field(self, field: str, value: Any) -> dict[str, Any] | None:
        """Fetch a single document by an arbitrary field value."""
        return await self.collection.find_one({field: value})

    async def list_by_field(
        self,
        field: str,
        value: Any,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List documents where field equals value."""
        return await self.find_many(filter_query={field: value}, sort=sort, skip=skip, limit=limit)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique string ID for business identifiers."""
        return str(uuid4())
