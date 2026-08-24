"""Concrete MongoDB repositories for CMS Business Strategy collections.

Each repository extends MongoBaseRepository with collection-specific
query helpers (by_status, by_owner, slug lookups, etc.).
"""

from datetime import UTC
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai.inference.deployment.database.database.dal.mongo_base_repository import MongoBaseRepository


class BusinessDocumentRepository(MongoBaseRepository):
    collection_name = "business_documents"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_document_id(self, document_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("documentId", document_id)

    async def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        return await self.get_by_field("slug", slug)

    async def find_by_status(self, status: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"status": status},
            sort=[("updatedAt", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_by_owner(self, owner_id: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("owner", owner_id, sort=[("updatedAt", -1)], skip=skip, limit=limit)

    async def find_by_type(self, doc_type: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("type", doc_type, sort=[("updatedAt", -1)], skip=skip, limit=limit)

    async def find_by_tag(self, tag: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"tags": tag},
            sort=[("updatedAt", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_accessible_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        """Find documents a user can view (owner or in permissions.view)."""
        return await self.find_many(
            filter_query={
                "$or": [
                    {"owner": user_id},
                    {"permissions.view": user_id},
                    {"permissions.edit": user_id},
                ]
            },
            sort=[("updatedAt", -1)],
            skip=skip,
            limit=limit,
        )

    async def add_revision(self, document_id: str, revision: dict[str, Any]) -> dict[str, Any] | None:
        """Push a new revision onto the document's revisions array."""
        return await self.update(
            document_id,
            {
                "revisions": revision,
                "version": revision.get("version", 1),
            },
            id_field="documentId",
        )


class ProjectRepository(MongoBaseRepository):
    collection_name = "projects"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_project_id(self, project_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("projectId", project_id)

    async def find_by_status(self, status: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"status": status},
            sort=[("startDate", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_by_owner(self, owner_id: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("owner", owner_id, sort=[("updatedAt", -1)], skip=skip, limit=limit)

    async def find_by_stakeholder(self, user_id: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"stakeholders.userId": user_id},
            sort=[("updatedAt", -1)],
            skip=skip,
            limit=limit,
        )


class MarketResearchRepository(MongoBaseRepository):
    collection_name = "market_research"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_research_id(self, research_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("researchId", research_id)

    async def find_by_type(self, research_type: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"type": research_type},
            sort=[("researchDate", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_due_for_review(self, limit: int = 50) -> list[dict[str, Any]]:
        """Find research entries whose nextReviewDate is past or today."""
        from datetime import datetime

        now = datetime.now(UTC)
        return await self.find_many(
            filter_query={"nextReviewDate": {"$lte": now}},
            sort=[("nextReviewDate", 1)],
            limit=limit,
        )


class StrategicPlanRepository(MongoBaseRepository):
    collection_name = "strategic_plans"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_plan_id(self, plan_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("planId", plan_id)

    async def find_by_status(self, status: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"status": status},
            sort=[("startDate", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_by_fiscal_year(self, fiscal_year: int, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("fiscalYear", fiscal_year, sort=[("quarter", 1)], skip=skip, limit=limit)


class SalesOpportunityRepository(MongoBaseRepository):
    collection_name = "sales_opportunities"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_opportunity_id(self, opportunity_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("opportunityId", opportunity_id)

    async def find_by_stage(self, stage: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"stage": stage},
            sort=[("expectedCloseDate", 1)],
            skip=skip,
            limit=limit,
        )

    async def find_by_owner(self, owner_id: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("owner", owner_id, sort=[("expectedCloseDate", 1)], skip=skip, limit=limit)

    async def find_pipeline(self, skip: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        """Active pipeline: all non-won/lost opportunities sorted by stage."""
        return await self.find_many(
            filter_query={"stage": {"$nin": ["won", "lost"]}},
            sort=[("stage", 1), ("priority", -1)],
            skip=skip,
            limit=limit,
        )

    async def get_pipeline_value(self) -> dict[str, float]:
        """Aggregate total value by stage."""
        pipeline: list[dict[str, Any]] = [{"$group": {"_id": "$stage", "total": {"$sum": "$value"}}}]
        results = await self.collection.aggregate(pipeline).to_list(length=20)
        return {str(r["_id"]): r["total"] for r in results if r["_id"] is not None}


class KnowledgeArticleRepository(MongoBaseRepository):
    collection_name = "knowledge_articles"

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)

    async def get_by_article_id(self, article_id: str) -> dict[str, Any] | None:
        return await self.get_by_field("articleId", article_id)

    async def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        return await self.get_by_field("slug", slug)

    async def find_published(self, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"status": "published"},
            sort=[("publishedDate", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_featured(self, skip: int = 0, limit: int = 10) -> list[dict[str, Any]]:
        return await self.find_many(
            filter_query={"featured": True, "status": "published"},
            sort=[("publishedDate", -1)],
            skip=skip,
            limit=limit,
        )

    async def find_by_category(self, category: str, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return await self.list_by_field("category", category, sort=[("publishedDate", -1)], skip=skip, limit=limit)

    async def increment_views(self, article_id: str) -> None:
        """Atomically increment view count."""
        await self.collection.update_one({"articleId": article_id}, {"$inc": {"views": 1}})
