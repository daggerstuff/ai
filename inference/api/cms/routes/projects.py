"""Project management API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.inference.deployment.database.database.dal.repositories.business_documents import ProjectRepository

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo(request: Request) -> ProjectRepository:
    return ProjectRepository(request.app.state.cms_db.mongo.db)


@router.get("")
async def list_projects(
    request: Request,
    status: str | None = None,
    owner: str | None = None,
    stakeholder: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    repo = _repo(request)
    if status:
        projects = await repo.find_by_status(status, skip=skip, limit=limit)
    elif owner:
        projects = await repo.find_by_owner(owner, skip=skip, limit=limit)
    elif stakeholder:
        projects = await repo.find_by_stakeholder(stakeholder, skip=skip, limit=limit)
    else:
        projects = await repo.find_many(sort=[("updatedAt", -1)], skip=skip, limit=limit)

    for p in projects:
        p.pop("_id", None)
    return {"success": True, "data": projects}


@router.post("", status_code=201)
async def create_project(request: Request) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    body.setdefault("projectId", repo.generate_id())
    doc = await repo.create(body)
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.get("/{project_id}")
async def get_project(request: Request, project_id: str) -> dict[str, Any]:
    repo = _repo(request)
    doc = await repo.get_by_project_id(project_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Project not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.patch("/{project_id}")
async def update_project(request: Request, project_id: str) -> dict[str, Any]:
    repo = _repo(request)
    body = await request.json()
    doc = await repo.update(project_id, body, id_field="projectId")
    if doc is None:
        raise HTTPException(status_code=404, detail="Project not found")
    doc.pop("_id", None)
    return {"success": True, "data": doc}


@router.delete("/{project_id}")
async def delete_project(request: Request, project_id: str) -> dict[str, Any]:
    repo = _repo(request)
    deleted = await repo.delete(project_id, id_field="projectId")
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True}


@router.get("/search/{query}")
async def search_projects(request: Request, query: str, skip: int = 0, limit: int = 25) -> dict[str, Any]:
    repo = _repo(request)
    results = await repo.search(query, skip=skip, limit=limit)
    for doc in results:
        doc.pop("_id", None)
        doc.pop("score", None)
    return {"success": True, "data": results}
