"""Knowledge articles API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.inference.deployment.database.database.dal.repositories.business_documents import (
    KnowledgeArticleRepository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo(request: Request) -> KnowledgeArticleRepository:
    return KnowledgeArticleRepository(request.app.state.cms_db.mongo.db)


@router.get("")
async def list_articles(
    request: Request,
    category: str | None = None,
    featured: bool = False,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _repo(request)
    if featured:
        docs = await repo.find_featured(skip=skip, limit=limit)
    elif category:
        docs = await repo.find_by_category(category, skip=skip, limit=limit)
    else:
        docs = await repo.find_published(skip=skip, limit=limit)
    for d in docs:
        d.pop("_id", None)
    return {"success": True, "data": docs}


@router.post("", status_code=201)
async def create_article(request: Request) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    body.setdefault("articleId", repo.generate_id())
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/slug/{slug}")
async def get_by_slug(request: Request, slug: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_slug(slug)
    if doc is None:
        raise HTTPException(status_code=404, detail="Article not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/{article_id}")
async def get_article(request: Request, article_id: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_article_id(article_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Article not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/{article_id}")
async def update_article(request: Request, article_id: str) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.update(article_id, body, id_field="articleId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Article not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/{article_id}")
async def delete_article(request: Request, article_id: str) -> dict[str, Any]:
    repo = _repo(request)
    deleted = await repo.delete(article_id, id_field="articleId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"success": True}


@router.post("/{article_id}/view")
async def increment_views(request: Request, article_id: str) -> dict[str, Any]:
    repo = _repo(request)
    await repo.increment_views(article_id)
    return {"success": True}


@router.get("/search/{query}")
async def search_articles(request: Request, query: str, skip: int = 0, limit: int = 25) -> dict[str, Any]:
    repo = _repo(request)
    results = await repo.search(query, skip=skip, limit=limit)
    for doc in results:
        doc.pop("_id", None)
        doc.pop("score", None)
    return {"success": True, "data": results}
