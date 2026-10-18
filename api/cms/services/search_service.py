"""Cross-collection search service for CMS Business Strategy.

Provides unified text search across MongoDB collections with
optional type filtering and pagination.
"""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai.infrastructure.database.dal.repositories.business_documents import (
    BusinessDocumentRepository,
    KnowledgeArticleRepository,
    MarketResearchRepository,
    ProjectRepository,
    SalesOpportunityRepository,
    StrategicPlanRepository,
)

logger = logging.getLogger(__name__)

COLLECTION_TYPES = [
    "documents",
    "projects",
    "strategies",
    "research",
    "sales",
    "knowledge",
]


class SearchService:
    """Search across multiple MongoDB collections in a single query."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self._db = db
        self._repos: dict[str, Any] = {
            "documents": BusinessDocumentRepository(db),
            "projects": ProjectRepository(db),
            "strategies": StrategicPlanRepository(db),
            "research": MarketResearchRepository(db),
            "sales": SalesOpportunityRepository(db),
            "knowledge": KnowledgeArticleRepository(db),
        }

    async def search(
        self,
        query: str,
        collections: list[str] | None = None,
        limit_per_collection: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search across specified collections. Returns results keyed by collection name."""
        target_collections = collections or COLLECTION_TYPES
        invalid = set(target_collections) - set(self._repos.keys())
        if invalid:
            raise ValueError(f"Unknown collection types: {invalid}")

        results: dict[str, list[dict[str, Any]]] = {}
        for name in target_collections:
            repo = self._repos[name]
            try:
                docs = await repo.search(query, limit=limit_per_collection)
                for doc in docs:
                    doc.pop("_id", None)
                    doc.pop("score", None)
                results[name] = docs
            except Exception as exc:
                logger.warning("Search failed for collection %s: %s", name, exc)
                results[name] = []

        return results

    async def search_all(
        self,
        query: str,
        limit_per_collection: int = 10,
    ) -> list[dict[str, Any]]:
        """Search all collections and return a flat, ranked list."""
        grouped = await self.search(query, limit_per_collection=limit_per_collection)
        flat: list[dict[str, Any]] = []
        for collection_name, docs in grouped.items():
            for doc in docs:
                doc["_collection"] = collection_name
                flat.append(doc)
        return flat
