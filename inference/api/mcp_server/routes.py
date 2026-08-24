"""
Route factories for the shared memory server.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ai.inference.api.mcp_server.memory_auth import (
    MemoryAccessContext,
    authorize_memory_access,
    readiness_details,
    resolve_authorized_user_id,
)
from ai.inference.api.mcp_server.memory_query_service import (
    get_scoped_memories,
    get_scoped_memory_stats,
    recall_memories_for_user,
)
from ai.inference.api.mcp_server.memory_scope import (
    build_scope_metadata,
    memory_in_scope,
    scope_from_kwargs,
    search_with_overfetch,
)
from ai.inference.api.mcp_server.schemas import (
    AddMemoryRequest,
    ForesightRecallRequest,
    ForesightRetainRequest,
    ScopeRequest,
    SearchMemoryRequest,
    UpdateMemoryRequest,
)
from ai.research.base import (
    CategoryScopedMemoryManager,
    ForesightCompatibleMemoryManager,
    HealthReportingMemoryManager,
    ScopedMemoryManager,
)
from ai.research.foresight_local_retention import RetainScopeConflictError, scope_metadata
from ai.research.local_foresight_document_service import DocumentAccessError

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
    request: Request,
    actor_id: str | None,
    user_id: str | None,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    handler: Callable[[MemoryAccessContext], Any],
) -> Any:
    request_body = await request.body()
    request_target = request.url.path
    if request.url.query:
        request_target = f"{request_target}?{request.url.query}"

    def _handler() -> Any:
        access = authorize_memory_access(
            actor_id=actor_id,
            user_id=user_id,
            request_method=request.method,
            request_target=request_target,
            request_body=request_body,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            authorization=request.headers.get("Authorization"),
        )
        return handler(access)

    return await run_in_threadpool(_route_call, action, _handler)


async def _run_authorized_for_expected_user(
    *,
    action: str,
    request: Request,
    actor_id: str | None,
    scoped_user_id: str | None,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    expected_user_id: str,
    callback: Callable[[MemoryAccessContext], Any],
) -> Any:
    return await _run_authorized(
        action=action,
        request=request,
        actor_id=actor_id,
        user_id=scoped_user_id,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
        handler=lambda access: _guard_user_scope(
            access=access,
            expected_user_id=expected_user_id,
            scoped_user_id=scoped_user_id,
            callback=lambda: callback(access),
        ),
    )


def _require_scoped_manager(manager: Any) -> ScopedMemoryManager:
    if not isinstance(manager, ScopedMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Configured memory manager does not support scoped memory listing",
        )
    return manager


def _require_foresight_manager(manager: Any) -> ForesightCompatibleMemoryManager:
    if not isinstance(manager, ForesightCompatibleMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Configured memory manager does not support Foresight-compatible operations",
        )
    return manager


def _require_category_scoped_manager(manager: Any) -> CategoryScopedMemoryManager:
    if not isinstance(manager, CategoryScopedMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Configured memory manager does not support scoped memory stats",
        )
    return manager


def _scope_for_memory(
    *,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
):
    return scope_from_kwargs(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )


def _get_authorized_memory_record(
    manager: Any,
    *,
    memory_id: str,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
) -> dict[str, Any]:
    scope = _scope_for_memory(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    if not memory_in_scope(manager=manager, scope=scope, memory_id=memory_id):
        raise HTTPException(status_code=404, detail="Memory not found in provided scope")
    record = manager.get_memory(memory_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    owner = record.get("user_id")
    if owner and owner != user_id:
        raise HTTPException(status_code=404, detail="Memory not found in provided scope")
    return record


def _enforce_user_scope(
    *,
    access: MemoryAccessContext,
    expected_user_id: str | None = None,
    scoped_user_id: str | None = None,
) -> str:
    resolved_user_id = resolve_authorized_user_id(access, scoped_user_id)
    if expected_user_id is not None and resolved_user_id != expected_user_id:
        raise HTTPException(
            status_code=400,
            detail="X-Memory-User-Id must match the requested user scope",
        )
    return access.assert_user_scope(resolved_user_id)


def _ensure_document_write_access(
    manager: Any,
    *,
    bank_id: str,
    document_id: str,
    user_id: str,
) -> None:
    foresight_manager = _require_foresight_manager(manager)
    if not foresight_manager.can_write_document(bank_id, document_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Document not found")


def _get_authorized_foresight_document(
    manager: ForesightCompatibleMemoryManager,
    *,
    bank_id: str,
    document_id: str,
    user_id: str,
) -> dict[str, Any]:
    _ensure_document_write_access(
        manager,
        bank_id=bank_id,
        document_id=document_id,
        user_id=user_id,
    )
    document = manager.get_document(bank_id, document_id, user_id=user_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _prepare_foresight_retain_items(
    manager: Any,
    *,
    bank_id: str,
    items: list[dict[str, Any]],
    user_id: str,
    actor_metadata: dict[str, Any],
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    visibility: str | None,
) -> list[dict[str, Any]]:
    foresight_manager = _require_foresight_manager(manager)
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
        return foresight_manager.prepare_retained_items(
            bank_id=bank_id,
            user_id=user_id,
            items=items,
            base_metadata=base_metadata,
        )
    except RetainScopeConflictError as exc:
        logger.error("Retain scope conflict while preparing retained items: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="The requested retain operation could not be completed due to a scope conflict.",
        ) from exc
    except DocumentAccessError as exc:
        logger.error("Document access denied while preparing retained items: %s", exc)
        raise HTTPException(
            status_code=404,
            detail="The requested document could not be accessed.",
        ) from exc
    except Exception as exc:
        if exc.__class__.__name__ == "DocumentAccessError":
            logger.error("Document access denied while preparing retained items: %s", exc)
            raise HTTPException(
                status_code=404,
                detail="The requested document could not be accessed.",
            ) from exc
        raise


def _build_scope_from_request(request: ScopeRequest, *, visibility: str | None = None):
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
    metadata: dict[str, Any] | None,
    access: MemoryAccessContext,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    merged.update(access.audit_metadata())
    return merged


def _sanitize_user_profile_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
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
    scoped_user_id: str | None,
    callback: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
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
    metadata: dict[str, Any] | None,
    scoped_user_id: str | None,
    access: MemoryAccessContext,
) -> dict[str, Any]:
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
        "message": "User registered for Foresight context",
    }


def _get_user_response(
    *,
    manager: Any,
    requested_user_id: str,
    scoped_user_id: str | None,
    access: MemoryAccessContext,
) -> dict[str, Any]:
    resolved_user_id = _enforce_user_scope(
        access=access,
        expected_user_id=requested_user_id,
        scoped_user_id=scoped_user_id,
    )
    memories = manager.get_all_memories(resolved_user_id, limit=1)
    has_history = bool(memories)
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
) -> dict[str, Any]:
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


def _search_memory_response(manager: Any, request: SearchMemoryRequest) -> dict[str, Any]:
    scope = _build_scope_from_request(request)
    limit = request.limit or 10
    memories = search_with_overfetch(
        manager=manager,
        query=request.query,
        user_id=request.user_id,
        requested_limit=limit,
        scope=scope,
    )
    return {"success": True, "memories": memories, "count": len(memories)}


def _update_memory_response(
    manager: Any,
    memory_id: str,
    request: UpdateMemoryRequest,
    *,
    access: MemoryAccessContext,
) -> dict[str, Any]:
    _get_authorized_memory_record(
        manager,
        memory_id=memory_id,
        user_id=request.user_id,
        org_id=request.org_id,
        project_id=request.project_id,
        session_id=request.session_id,
        agent_id=request.agent_id,
        run_id=request.run_id,
        include_shared=request.include_shared is not False,
    )
    new_content = request.content
    if not new_content:
        raise HTTPException(status_code=400, detail="Update content is required")
    success = manager.update_memory(
        memory_id=memory_id,
        new_content=new_content,
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
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
) -> dict[str, Any]:
    _get_authorized_memory_record(
        manager,
        memory_id=memory_id,
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    deleted = manager.delete_memory(memory_id=memory_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"success": True, "message": "Memory deleted"}


def _get_memory_response(
    manager: Any,
    *,
    memory_id: str,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
) -> dict[str, Any]:
    return _get_authorized_memory_record(
        manager,
        memory_id=memory_id,
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )


def _get_all_memories_response(
    manager: Any,
    *,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
    limit: int | None,
    offset: int,
    category: str | None,
    tags: list[str] | None,
) -> dict[str, Any]:
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
        offset=offset,
        category=category,
        tags=tags,
    )
    if isinstance(memories, dict) and "results" in memories:
        memories = memories["results"]
    return {"success": True, "memories": memories, "count": len(memories)}


def _get_memory_stats_response(
    manager: Any,
    *,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
) -> dict[str, Any]:
    category_counts = get_scoped_memory_stats(
        _require_category_scoped_manager(manager),
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    return {
        "success": True,
        "totalMemories": sum(category_counts.values()),
        "categoryCounts": category_counts,
    }


def _foresight_document_response(
    *,
    manager: ForesightCompatibleMemoryManager,
    bank_id: str,
    document_id: str,
    user_id: str,
) -> dict[str, Any]:
    return _get_authorized_foresight_document(
        manager,
        bank_id=bank_id,
        document_id=document_id,
        user_id=user_id,
    )


def _foresight_delete_document_response(
    *,
    manager: ForesightCompatibleMemoryManager,
    bank_id: str,
    document_id: str,
    user_id: str,
) -> Response:
    _get_authorized_foresight_document(
        manager,
        bank_id=bank_id,
        document_id=document_id,
        user_id=user_id,
    )
    deleted = manager.delete_document(bank_id, document_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


def _foresight_retain_response(
    *,
    manager: ForesightCompatibleMemoryManager,
    bank_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    response = manager.retain_items(bank_id, items)
    if isinstance(response, dict):
        results = response.get("results", [])
        items_count = len(results) if isinstance(results, list) else 0
        response = {
            "success": True,
            "bank_id": bank_id,
            "async": False,
            "items_count": items_count,
            **response,
        }
    return response


def _register_add_memory(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.post("/api/memory/add")
    async def add_memory(
        request_context: Request,
        request: AddMemoryRequest,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized_for_expected_user(
            action="adding memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            scoped_user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            expected_user_id=request.user_id,
            callback=lambda access: _add_memory_response(
                get_manager(),
                request,
                access=access,
            ),
        )


def _register_memory_search(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.post("/api/memory/search")
    async def search_memory(
        request_context: Request,
        request: SearchMemoryRequest,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized_for_expected_user(
            action="searching memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            scoped_user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            expected_user_id=request.user_id,
            callback=lambda access: _search_memory_response(get_manager(), request),
        )


def _register_memory_update(
    router: APIRouter,
    get_manager: ManagerGetter,
) -> None:
    @router.patch("/api/memory/{memory_id}")
    async def update_memory(
        request_context: Request,
        memory_id: str,
        request: UpdateMemoryRequest,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized_for_expected_user(
            action="updating memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            scoped_user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            expected_user_id=request.user_id,
            callback=lambda access: _update_memory_response(
                get_manager(),
                memory_id,
                request,
                access=access,
            ),
        )


def _register_memory_get(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.get("/api/memory/{memory_id}")
    async def get_memory_endpoint(
        request_context: Request,
        memory_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="fetching memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _get_memory_response(
                get_manager(),
                memory_id=memory_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
            ),
        )


def _register_memory_delete(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.delete("/api/memory/{memory_id}")
    async def delete_memory_endpoint(
        request_context: Request,
        memory_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="deleting memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _delete_memory_response(
                get_manager(),
                memory_id=memory_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
            ),
        )


def _register_memory_list(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.get("/api/memory/all/{user_id}")
    async def get_all_memories_endpoint(
        request_context: Request,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int | None = None,
        offset: int = 0,
        category: str | None = None,
        tags: list[str] | None = Query(default=None),
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized_for_expected_user(
            action="fetching all memories",
            request=request_context,
            actor_id=x_memory_actor_id,
            scoped_user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            expected_user_id=user_id,
            callback=lambda access: _get_all_memories_response(
                get_manager(),
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
                limit=limit,
                offset=offset,
                category=category,
                tags=tags,
            ),
        )


def _register_memory_stats(router: APIRouter, get_manager: ManagerGetter) -> None:
    @router.get("/api/memory/stats/{user_id}")
    async def get_memory_stats_endpoint(
        request_context: Request,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized_for_expected_user(
            action="fetching memory stats",
            request=request_context,
            actor_id=x_memory_actor_id,
            scoped_user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            expected_user_id=user_id,
            callback=lambda access: _get_memory_stats_response(
                get_manager(),
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
            ),
        )


def create_mcp_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["MCP Tools"])
    _register_add_memory(router, get_manager)
    _register_memory_search(router, get_manager)
    _register_memory_update(router, get_manager)
    _register_memory_get(router, get_manager)
    _register_memory_delete(router, get_manager)
    _register_memory_list(router, get_manager)
    _register_memory_stats(router, get_manager)
    return router


def create_foresight_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["Foresight Compatibility"])

    @router.post("/v1/default/banks/{bank_id}/memories")
    async def foresight_retain(
        request_context: Request,
        bank_id: str,
        request: ForesightRetainRequest,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
        x_memory_org_id: str | None = Header(default=None),
        x_memory_project_id: str | None = Header(default=None),
        x_memory_session_id: str | None = Header(default=None),
        x_memory_agent_id: str | None = Header(default=None),
        x_memory_run_id: str | None = Header(default=None),
        x_memory_visibility: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="retaining foresight memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _foresight_retain_response(
                manager=_require_foresight_manager(get_manager()),
                bank_id=bank_id,
                items=_prepare_foresight_retain_items(
                    get_manager(),
                    bank_id=bank_id,
                    items=[item.model_dump() for item in request.items],
                    user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                    actor_metadata=access.audit_metadata(),
                    org_id=x_memory_org_id,
                    project_id=x_memory_project_id,
                    session_id=x_memory_session_id,
                    agent_id=x_memory_agent_id,
                    run_id=x_memory_run_id,
                    visibility=x_memory_visibility,
                ),
            ),
        )

    @router.post("/v1/default/banks/{bank_id}/memories/recall")
    async def foresight_recall(
        request_context: Request,
        bank_id: str,
        request: ForesightRecallRequest,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="recalling foresight memory",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: recall_memories_for_user(
                _require_foresight_manager(get_manager()),
                bank_id=bank_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                query=request.query,
                limit=request.limit or 10,
                tags=request.tags,
                tags_match=request.tags_match,
            ),
        )

    @router.get("/v1/default/banks/{bank_id}/documents")
    async def foresight_list_documents(
        request_context: Request,
        bank_id: str,
        limit: int = 100,
        offset: int = 0,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="listing foresight documents",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _require_foresight_manager(get_manager()).list_documents(
                bank_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
                limit=limit,
                offset=offset,
            ),
        )

    @router.get("/v1/default/banks/{bank_id}/documents/{document_id}")
    async def foresight_get_document(
        request_context: Request,
        bank_id: str,
        document_id: str,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="fetching foresight document",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _foresight_document_response(
                manager=_require_foresight_manager(get_manager()),
                bank_id=bank_id,
                document_id=document_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
            ),
        )

    @router.delete("/v1/default/banks/{bank_id}/documents/{document_id}")
    async def foresight_delete_document(
        request_context: Request,
        bank_id: str,
        document_id: str,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="deleting foresight document",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
            handler=lambda access: _foresight_delete_document_response(
                manager=_require_foresight_manager(get_manager()),
                bank_id=bank_id,
                document_id=document_id,
                user_id=_enforce_user_scope(access=access, scoped_user_id=x_memory_user_id),
            ),
        )

    return router


def create_legacy_router(get_manager: ManagerGetter) -> APIRouter:
    router = APIRouter(tags=["Legacy"])

    @router.post("/api/memory/users")
    async def create_user(
        request_context: Request,
        email: str,
        name: str,
        role: str = "patient",
        metadata: dict[str, Any] | None = None,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="registering memory user",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
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
        request_context: Request,
        user_id: str,
        x_memory_actor_id: str | None = Header(default=None),
        x_memory_user_id: str | None = Header(default=None),
        x_memory_timestamp: str | None = Header(default=None),
        x_memory_nonce: str | None = Header(default=None),
        x_memory_signature: str | None = Header(default=None),
    ):
        return await _run_authorized(
            action="checking user",
            request=request_context,
            actor_id=x_memory_actor_id,
            user_id=x_memory_user_id,
            timestamp=x_memory_timestamp,
            nonce=x_memory_nonce,
            signature=x_memory_signature,
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
            combined_readiness = dict(readiness_details())
            manager_readiness = health_payload.get("readiness")
            if isinstance(manager_readiness, dict):
                combined_readiness.update(manager_readiness)
            return {
                "status": health_payload.get("status", "degraded"),
                "provider": health_payload.get("provider", "unconfigured"),
                "service": "pixelated-memory-server",
                "version": "2.0.0",
                "auth_model": "internal_service_hmac_actor_policies",
                "readiness": combined_readiness,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "Service unavailable"})

    return router
