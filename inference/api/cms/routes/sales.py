"""Sales opportunities API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.infrastructure.database.dal.repositories.business_documents import (
    SalesOpportunityRepository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo(request: Request) -> SalesOpportunityRepository:
    return SalesOpportunityRepository(request.app.state.cms_db.mongo.db)


@router.get("")
async def list_opportunities(
    request: Request,
    stage: str | None = None,
    owner: str | None = None,
    pipeline: bool = False,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _repo(request)
    if pipeline:
        docs = await repo.find_pipeline(skip=skip, limit=limit)
    elif stage:
        docs = await repo.find_by_stage(stage, skip=skip, limit=limit)
    elif owner:
        docs = await repo.find_by_owner(owner, skip=skip, limit=limit)
    else:
        docs = await repo.find_many(sort=[("expectedCloseDate", 1)], skip=skip, limit=limit)
    for d in docs:
        d.pop("_id", None)
    return {"success": True, "data": docs}


@router.get("/pipeline-value")
async def pipeline_value(request: Request) -> dict[str, Any]:
    repo = _repo(request)
    values = await repo.get_pipeline_value()
    return {"success": True, "data": values}


@router.post("", status_code=201)
async def create_opportunity(request: Request) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    body.setdefault("opportunityId", repo.generate_id())
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/{opportunity_id}")
async def get_opportunity(request: Request, opportunity_id: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_opportunity_id(opportunity_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/{opportunity_id}")
async def update_opportunity(request: Request, opportunity_id: str) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.update(opportunity_id, body, id_field="opportunityId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/{opportunity_id}")
async def delete_opportunity(request: Request, opportunity_id: str) -> dict[str, Any]:
    repo = _repo(request)
    deleted = await repo.delete(opportunity_id, id_field="opportunityId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    return {"success": True}
