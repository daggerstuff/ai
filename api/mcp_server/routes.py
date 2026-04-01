"""
Route factories for the shared memory server.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, cast

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai.api.mcp_server.memory_auth import (
    MemoryAccessContext,
    authorize_memory_access,
    readiness_details,
    required_user_id,
)
from ai.api.mcp_server.memory_query_service import get_scoped_memories
from ai.api.mcp_server.memory_query_service import recall_memories_for_user
from ai.api.mcp_server.memory_scope import (
    build_scope_metadata,
    filter_memories_by_scope,
    memory_in_scope,
    scope_from_kwargs,
    search_with_overfetch,
)
from ai.memory.base import (
    HealthReportingMemoryManager,
    HindsightCompatibleMemoryManager,
    ScopedMemoryManager,
)
from ai.memory.local_hindsight_document_service import DocumentAccessError
from ai.memory.hindsight_local_retention import scope_metadata

logger = logging.getLogger(__name__)

ManagerGetter = Callable[[], Any]
_ALLOWED_USER_PROFILE_METADATA_KEYS = {
    "display_name",
    "timezone",
    "locale",
    "team",
    "org_id",
    "project_id",
    "notes",
}


def _route_call(action: str, handler: Callable[[], Any]) -> Any:
    try:
        return handler()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error %s", action)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while {action}",
        ) from exc


async def _run_authorized(
    *,
    action: str,
    authorization: Optional[str],
    actor_id: Optional[str],
    handler: Callable[[MemoryAccessContext], Any],
) -> Any:
    def _handler() -> Any:
        access = authorize_memory_access(
            authorization=authorization,
            actor_id=actor_id,
        )
        return handler(access)

    return await run_in_threadpool(_route_call, action, _handler)


def _require_scoped_manager(manager: Any) -> ScopedMemoryManager:
    if not isinstance(manager, ScopedMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Configured memory manager does not support scoped memory listing",
        )
    return cast(ScopedMemoryManager, manager)


def _require_hindsight_manager(manager: Any) -> HindsightCompatibleMemoryManager:
    if not isinstance(manager, HindsightCompatibleMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Configured memory manager does not support Hindsight-compatible operations",
        )
    return cast(HindsightCompatibleMemoryManager, manager)


def _ensure_memory_owner(manager: Any, memory_id: str, user_id: str) -> None:
    record = manager.get_memory(memory_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    owner = record.get("user_id")
    if owner and owner != user_id:
        raise HTTPException(status_code=404, detail="Memory not found in provided scope")


def _enforce_user_scope(
    *,
    access: MemoryAccessContext,
    expected_user_id: Optional[str] = None,
    scoped_user_id: Optional[str] = None,
) -> str:
    resolved_user_id = required_user_id(scoped_user_id)
    if expected_user_id is not None and resolved_user_id != expected_user_id:
        raise HTTPException(
            status_code=400,
            detail="X-Memory-User-Id must match the requested user scope",
        )
    return access.assert_user_scope(resolved_user_id)


class AddMemoryRequest(BaseModel):
    content: str
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    visibility: Optional[str] = "private"
    include_shared: Optional[bool] = True
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    include_shared: Optional[bool] = True
    limit: Optional[int] = 10


class UpdateMemoryRequest(BaseModel):
    text: str
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    include_shared: Optional[bool] = True
    metadata: Optional[Dict[str, Any]] = None


class HindsightRetainItem(BaseModel):
    content: str
    document_id: Optional[str] = None
    context: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class HindsightRetainRequest(BaseModel):
    items: List[HindsightRetainItem]


class HindsightRecallRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    tags: Optional[List[str]] = None
    tags_match: Optional[str] = "any"


class _ScopeRequest(Protocol):
    user_id: str
    org_id: Optional[str]
    project_id: Optional[str]
    session_id: Optional[str]
    agent_id: Optional[str]
    run_id: Optional[str]
    include_shared: Optional[bool]


def _ensure_document_write_access(
    manager: Any,
    *,
    bank_id: str,
    document_id: str,
    user_id: str,
) -> None:
    hindsight_manager = _require_hindsight_manager(manager)
    if not hindsight_manager.can_write_document(bank_id, document_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Document not found")


def _prepare_hindsight_retain_items(
    manager: Any,
    *,
    bank_id: str,
    items: List[Dict[str, Any]],
    user_id: str,
    actor_metadata: Dict[str, Any],
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    visibility: Optional[str],
) -> List[Dict[str, Any]]:
    hindsight_manager = _require_hindsight_manager(manager)
    base_metadata = scope_metadata(
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        visibility=visibility,
    )
    base_metadata.update(actor_metadata)
    try:
        return hindsight_manager.prepare_retained_items(
            bank_id=bank_id,
            user_id=user_id,
            items=items,
            base_metadata=base_metadata,
        )
    except DocumentAccessError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if exc.__class__.__name__ == "DocumentAccessError":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise


def _build_scope_from_request(request: _ScopeRequest, *, visibility: Optional[str] = None):
    return scope_from_kwargs(
        user_id=request.user_id,
        org_id=request.org_id,
        project_id=request.project_id,
        session_id=request.session_id,
        agent_id=request.agent_id,
        run_id=request.run_id,
        visibility=visibility,
        include_shared=request.include_shared is not False,
    )


def _merge_actor_metadata(
    *,
    metadata: Optional[Dict[str, Any]],
    access: MemoryAccessContext,
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    merged.update(access.audit_metadata())
    return merged


def _sanitize_user_profile_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key not in _ALLOWED_USER_PROFILE_METADATA_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
    return sanitized


def _guard_user_scope(
    *,
    access: MemoryAccessContext,
    expected_user_id: str,
    scoped_user_id: Optional[str],
    callback: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    _enforce_user_scope(
        access=access,
        expected_user_id=expected_user_id,
        scoped_user_id=scoped_user_id,
    )
    return callback()


def _create_user_response(
    *,
    manager: Any,
    email: str,
    name: str,
    role: str,
    metadata: Optional[Dict[str, Any]],
    scoped_user_id: Optional[str],
    access: MemoryAccessContext,
) -> Dict[str, Any]:
    user_id = _enforce_user_scope(
        access=access,
        expected_user_id=email,
        scoped_user_id=scoped_user_id,
    )
    profile_metadata = _sanitize_user_profile_metadata(metadata)
    profile_metadata.update(access.audit_metadata())
    profile_metadata["name"] = name
    profile_metadata["role"] = role
    profile_metadata["record_type"] = "user_registration"
    manager.add_memory(
        f"Registered memory user profile for {name} ({email})",
        user_id=user_id,
        metadata=profile_metadata,
        category="user_profile",
    )
    return {
        "success": True,
        "user_id": email,
        "email": email,
        "name": name,
        "role": role,
        "message": "User registered for Hindsight context",
    }


def _get_user_response(
    *,
    manager: Any,
    requested_user_id: str,
    scoped_user_id: Optional[str],
    access: MemoryAccessContext,
) -> Dict[str, Any]:
    resolved_user_id = _enforce_user_scope(
        access=access,
        expected_user_id=requested_user_id,
        scoped_user_id=scoped_user_id,
    )
    memories = manager.get_all_memories(resolved_user_id, limit=100)
    has_history = len(memories) > 0
    return {
        "success": True,
        "user_id": resolved_user_id,
        "has_history": has_history,
        "memory_count": len(memories) if isinstance(memories, list) else 0,
    }


def _add_memory_response(
    manager: Any,
    request: AddMemoryRequest,
    *,
    access: MemoryAccessContext,
) -> Dict[str, Any]:
    scope = _build_scope_from_request(request, visibility=request.visibility or "private")
    metadata = build_scope_metadata(
        scope=scope,
        incoming_metadata=_merge_actor_metadata(metadata=request.metadata, access=access),
        category=request.category,
    )
    memory_id = manager.add_memory(
        request.content,
        user_id=request.user_id,
        metadata=metadata or None,
        category=request.category,
    )
    if not memory_id:
        raise HTTPException(status_code=400, detail="Memory could not be added")
    return {
        "success": True,
        "memory_id": memory_id,
        "message": "Memory added successfully",
    }


def _search_memory_response(manager: Any, request: SearchMemoryRequest) -> Dict[str, Any]:
    scope = _build_scope_from_request(request)
    limit = request.limit or 10
    memories = search_with_overfetch(
        manager=manager,
        query=request.query,
        user_id=request.user_id,
        requested_limit=limit,
    )
    memories = filter_memories_by_scope(
        scope=scope,
        memories=memories or [],
        limit=limit,
    )
    return {"success": True, "memories": memories, "count": len(memories)}


def _update_memory_response(
    manager: Any,
    memory_id: str,
    request: UpdateMemoryRequest,
    *,
    access: MemoryAccessContext,
) -> Dict[str, Any]:
    scope = _build_scope_from_request(request)
    allowed = memory_in_scope(manager=manager, scope=scope, memory_id=memory_id)
    if not allowed:
        raise HTTPException(status_code=404, detail="Memory not found in provided scope")
    _ensure_memory_owner(manager, memory_id, request.user_id)
    success = manager.update_memory(
        memory_id=memory_id,
        new_content=request.text,
        metadata=_merge_actor_metadata(metadata=request.metadata, access=access),
        user_id=request.user_id,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Update rejected or failed")
    return {"success": True, "message": "Memory updated"}


def _delete_memory_response(
    manager: Any,
    *,
    memory_id: str,
    user_id: str,
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    include_shared: bool,
) -> Dict[str, Any]:
    scope = scope_from_kwargs(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    allowed = memory_in_scope(manager=manager, scope=scope, memory_id=memory_id)
    if not allowed:
        raise HTTPException(status_code=404, detail="Memory not found in provided scope")
    _ensure_memory_owner(manager, memory_id, user_id)
    deleted = manager.delete_memory(memory_id=memory_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"success": True, "message": "Memory deleted"}


def _get_all_memories_response(
    manager: Any,
    *,
    user_id: str,
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    include_shared: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    requested_limit = limit or 100
    memories = get_scoped_memories(
        _require_scoped_manager(manager),
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
        limit=requested_limit,
    )
    if isinstance(memories, dict) and "results" in memories:
        memories = memories["results"]
    return {"success": True, "memories": memories, "count": len(memories)}


def create_mcp_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["MCP Tools"])

    @router.post("/api/memory/add")
    async def add_memory(
        request: AddMemoryRequest,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="adding memory",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _guard_user_scope(
                access=access,
                expected_user_id=request.user_id,
                scoped_user_id=x_memory_user_id,
                callback=lambda: _add_memory_response(
                    get_manager(),
                    request,
                    access=access,
                ),
            )
        )

    @router.post("/api/memory/search")
    async def search_memory(
        request: SearchMemoryRequest,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="searching memory",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _guard_user_scope(
                access=access,
                expected_user_id=request.user_id,
                scoped_user_id=x_memory_user_id,
                callback=lambda: _search_memory_response(get_manager(), request),
            )
        )

    @router.patch("/api/memory/{memory_id}")
    async def update_memory(
        memory_id: str,
        request: UpdateMemoryRequest,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="updating memory",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _guard_user_scope(
                access=access,
                expected_user_id=request.user_id,
                scoped_user_id=x_memory_user_id,
                callback=lambda: _update_memory_response(
                    get_manager(),
                    memory_id,
                    request,
                    access=access,
                ),
            )
        )

    @router.delete("/api/memory/{memory_id}")
    async def delete_memory_endpoint(
        memory_id: str,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="deleting memory",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _guard_user_scope(
                access=access,
                expected_user_id=user_id,
                scoped_user_id=x_memory_user_id,
                callback=lambda: _delete_memory_response(
                    get_manager(),
                    memory_id=memory_id,
                    user_id=user_id,
                    org_id=org_id,
                    project_id=project_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    include_shared=include_shared,
                ),
            )
        )

    @router.get("/api/memory/all/{user_id}")
    async def get_all_memories_endpoint(
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: Optional[int] = None,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="fetching all memories",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _guard_user_scope(
                access=access,
                expected_user_id=user_id,
                scoped_user_id=x_memory_user_id,
                callback=lambda: _get_all_memories_response(
                    get_manager(),
                    user_id=user_id,
                    org_id=org_id,
                    project_id=project_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    include_shared=include_shared,
                    limit=limit,
                ),
            )
        )

    return router


def create_hindsight_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["Hindsight Compatibility"])

    @router.post("/v1/default/banks/{bank_id}/memories")
    async def hindsight_retain(
        bank_id: str,
        request: HindsightRetainRequest,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
        x_memory_org_id: Optional[str] = Header(default=None),
        x_memory_project_id: Optional[str] = Header(default=None),
        x_memory_session_id: Optional[str] = Header(default=None),
        x_memory_agent_id: Optional[str] = Header(default=None),
        x_memory_run_id: Optional[str] = Header(default=None),
        x_memory_visibility: Optional[str] = Header(default=None),
    ):
        def _handler() -> Dict[str, Any]:
            access = authorize_memory_access(
                authorization=authorization,
                actor_id=x_memory_actor_id,
            )
            user_id = _enforce_user_scope(access=access, scoped_user_id=x_memory_user_id)
            manager = _require_hindsight_manager(get_manager())
            return manager.retain_items(
                bank_id,
                _prepare_hindsight_retain_items(
                    manager,
                    bank_id=bank_id,
                    items=[item.model_dump() for item in request.items],
                    user_id=user_id,
                    actor_metadata=access.audit_metadata(),
                    org_id=x_memory_org_id,
                    project_id=x_memory_project_id,
                    session_id=x_memory_session_id,
                    agent_id=x_memory_agent_id,
                    run_id=x_memory_run_id,
                    visibility=x_memory_visibility,
                ),
            )

        return await run_in_threadpool(_route_call, "retaining hindsight memory", _handler)

    @router.post("/v1/default/banks/{bank_id}/memories/recall")
    async def hindsight_recall(
        bank_id: str,
        request: HindsightRecallRequest,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="recalling hindsight memory",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: recall_memories_for_user(
                _require_hindsight_manager(get_manager()),
                bank_id=bank_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                query=request.query,
                limit=request.limit or 10,
                tags=request.tags,
                tags_match=request.tags_match,
            ),
        )

    @router.get("/v1/default/banks/{bank_id}/documents")
    async def hindsight_list_documents(
        bank_id: str,
        limit: int = 100,
        offset: int = 0,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        def _handler() -> Dict[str, Any]:
            access = authorize_memory_access(
                authorization=authorization,
                actor_id=x_memory_actor_id,
            )
            user_id = _enforce_user_scope(access=access, scoped_user_id=x_memory_user_id)
            manager = _require_hindsight_manager(get_manager())
            return manager.list_documents(bank_id, user_id=user_id, limit=limit, offset=offset)

        return await run_in_threadpool(_route_call, "listing hindsight documents", _handler)

    @router.get("/v1/default/banks/{bank_id}/documents/{document_id}")
    async def hindsight_get_document(
        bank_id: str,
        document_id: str,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        def _handler() -> Dict[str, Any]:
            access = authorize_memory_access(
                authorization=authorization,
                actor_id=x_memory_actor_id,
            )
            user_id = _enforce_user_scope(access=access, scoped_user_id=x_memory_user_id)
            manager = _require_hindsight_manager(get_manager())
            document = manager.get_document(bank_id, document_id, user_id=user_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            return document

        return await run_in_threadpool(_route_call, "fetching hindsight document", _handler)

    @router.delete("/v1/default/banks/{bank_id}/documents/{document_id}")
    async def hindsight_delete_document(
        bank_id: str,
        document_id: str,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        def _handler() -> Response:
            access = authorize_memory_access(
                authorization=authorization,
                actor_id=x_memory_actor_id,
            )
            user_id = _enforce_user_scope(access=access, scoped_user_id=x_memory_user_id)
            manager = _require_hindsight_manager(get_manager())
            deleted = manager.delete_document(bank_id, document_id, user_id=user_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Document not found")
            return Response(status_code=204)

        return await run_in_threadpool(_route_call, "deleting hindsight document", _handler)

    return router


def create_legacy_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["Legacy"])

    @router.post("/api/memory/users")
    async def create_user(
        email: str,
        name: str,
        role: str = "patient",
        metadata: Optional[Dict[str, Any]] = None,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="registering memory user",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _create_user_response(
                manager=get_manager(),
                email=email,
                name=name,
                role=role,
                metadata=metadata,
                scoped_user_id=x_memory_user_id,
                access=access,
            ),
        )

    @router.get("/api/memory/users/{user_id}")
    async def get_user(
        user_id: str,
        authorization: Optional[str] = Header(default=None),
        x_memory_actor_id: Optional[str] = Header(default=None),
        x_memory_user_id: Optional[str] = Header(default=None),
    ):
        return await _run_authorized(
            action="checking user",
            authorization=authorization,
            actor_id=x_memory_actor_id,
            handler=lambda access: _get_user_response(
                manager=get_manager(),
                requested_user_id=user_id,
                scoped_user_id=x_memory_user_id,
                access=access,
            ),
        )

    return router


def create_health_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get("/health")
    def health_check():
        try:
            manager = get_manager()
            if isinstance(manager, HealthReportingMemoryManager):
                health_payload = manager.get_health_status()
            else:
                health_payload = {"status": "degraded", "provider": "unconfigured"}
            return {
                "status": health_payload.get("status", "degraded"),
                "provider": health_payload.get("provider", "unconfigured"),
                "service": "pixelated-memory-server",
                "version": "2.0.0",
                "auth_model": "internal_service_actor_policies",
                "readiness": readiness_details(),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

    return router
