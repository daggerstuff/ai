"""Strategic plans and market research API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.infrastructure.database.dal.repositories.business_documents import (
    MarketResearchRepository,
    StrategicPlanRepository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _strategy_repo(request: Request) -> StrategicPlanRepository:
    return StrategicPlanRepository(request.app.state.cms_db.mongo.db)


def _research_repo(request: Request) -> MarketResearchRepository:
    return MarketResearchRepository(request.app.state.cms_db.mongo.db)


# --- Strategic Plans ---


@router.get("/plans")
async def list_plans(
    request: Request,
    status: str | None = None,
    fiscal_year: int | None = None,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _strategy_repo(request)
    if fiscal_year is not None:
        plans = await repo.find_by_fiscal_year(fiscal_year, skip=skip, limit=limit)
    elif status:
        plans = await repo.find_by_status(status, skip=skip, limit=limit)
    else:
        plans = await repo.find_many(sort=[("startDate", -1)], skip=skip, limit=limit)
    for p in plans:
        p.pop("_id", None)
    return {"success": True, "data": plans}


@router.post("/plans", status_code=201)
async def create_plan(request: Request) -> dict[str, Any]:
    repo = _strategy_repo(request)
    body = await request.json()
    body.setdefault("planId", repo.generate_id())
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/plans/{plan_id}")
async def get_plan(request: Request, plan_id: str) -> dict[str, Any]:
    repo = _strategy_repo(request)
    doc = await repo.get_by_plan_id(plan_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Strategic plan not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/plans/{plan_id}")
async def update_plan(request: Request, plan_id: str) -> dict[str, Any]:
    repo = _strategy_repo(request)
    body = await request.json()
    doc = await repo.update(plan_id, body, id_field="planId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Strategic plan not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/plans/{plan_id}")
async def delete_plan(request: Request, plan_id: str) -> dict[str, Any]:
    repo = _strategy_repo(request)
    deleted = await repo.delete(plan_id, id_field="planId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Strategic plan not found")
    return {"success": True}


# --- Market Research ---


@router.get("/research")
async def list_research(
    request: Request,
    type: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _research_repo(request)
    if type:
        results = await repo.find_by_type(type, skip=skip, limit=limit)
    else:
        results = await repo.find_many(sort=[("researchDate", -1)], skip=skip, limit=limit)
    for r in results:
        r.pop("_id", None)
    return {"success": True, "data": results}


@router.get("/research/due-review")
async def due_for_review(request: Request, limit: int = 50) -> dict[str, Any]:
    repo = _research_repo(request)
    results = await repo.find_due_for_review(limit=limit)
    for r in results:
        r.pop("_id", None)
    return {"success": True, "data": results}


@router.post("/research", status_code=201)
async def create_research(request: Request) -> dict[str, Any]:
    repo = _research_repo(request)
    body = await request.json()
    body.setdefault("researchId", repo.generate_id())
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/research/{research_id}")
async def get_research(request: Request, research_id: str) -> dict[str, Any]:
    repo = _research_repo(request)
    doc = await repo.get_by_research_id(research_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Research not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/research/{research_id}")
async def update_research(request: Request, research_id: str) -> dict[str, Any]:
    repo = _research_repo(request)
    body = await request.json()
    doc = await repo.update(research_id, body, id_field="researchId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Research not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/research/{research_id}")
async def delete_research(request: Request, research_id: str) -> dict[str, Any]:
    repo = _research_repo(request)
    deleted = await repo.delete(research_id, id_field="researchId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Research not found")
    return {"success": True}
