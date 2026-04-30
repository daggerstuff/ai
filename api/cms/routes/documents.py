"""Document management API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.infrastructure.database.dal.repositories.business_documents import (
    BusinessDocumentRepository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo(request: Request) -> BusinessDocumentRepository:
    db = request.app.state.cms_db.mongo.db
    return BusinessDocumentRepository(db)


@router.get("")
async def list_documents(
    request: Request,
    status: str | None = None,
    type: str | None = None,
    tag: str | None = None,
    owner: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _repo(request)
    if status:
        docs = await repo.find_by_status(status, skip=skip, limit=limit)
    elif type:
        docs = await repo.find_by_type(type, skip=skip, limit=limit)
    elif tag:
        docs = await repo.find_by_tag(tag, skip=skip, limit=limit)
    elif owner:
        docs = await repo.find_by_owner(owner, skip=skip, limit=limit)
    else:
        docs = await repo.find_many(sort=[("updatedAt", -1)], skip=skip, limit=limit)

    for doc in docs:
        doc.pop("_id", None)

    return {"success": True, "data": docs}


@router.post("", status_code=201)
async def create_document(request: Request) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/{document_id}")
async def get_document(request: Request, document_id: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_document_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/slug/{slug}")
async def get_document_by_slug(request: Request, slug: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_slug(slug)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/{document_id}")
async def update_document(request: Request, document_id: str) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.update(document_id, body, id_field="documentId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/{document_id}")
async def delete_document(request: Request, document_id: str) -> dict[str, Any]:
    repo = _repo(request)
    deleted = await repo.delete(document_id, id_field="documentId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}


@router.get("/{document_id}/revisions")
async def get_revisions(request: Request, document_id: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_document_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "data": doc.get("revisions", [])}


@router.post("/{document_id}/revisions")
async def add_revision(request: Request, document_id: str) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.add_revision(document_id, body)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/search/{query}")
async def search_documents(
    request: Request,
    query: str,
    skip: int = 0,
    limit: int = 25,
) -> dict[str, Any]:
    repo = _repo(request)
    results = await repo.search(query, skip=skip, limit=limit)
    for doc in results:
        doc.pop("_id", None)
        doc.pop("score", None)
    return {"success": True, "data": results}
