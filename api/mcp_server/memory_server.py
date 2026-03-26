"""
MCP Memory Integration Server.

Provides MCP (Model Control Protocol) compatible endpoints for memory operations
with hybrid Mem0 and Gemini backend.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai.api.mcp_server.memory_scope import (
    build_scope_metadata,
    filter_memories_by_scope,
    memory_in_scope,
    scope_from_kwargs,
    search_with_overfetch,
)
from ai.api.memory.null_memory import NullMemoryManager
from ai.memory.manager_factory import get_memory_manager

logger = logging.getLogger(__name__)


# --- Request Models ---


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


def create_memory_server() -> FastAPI:
    """
    Create FastAPI server for memory operations.

    Returns:
        Configured FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize memory services on startup."""

        try:
            manager = get_memory_manager()
            if manager:
                app.state.memory_manager = manager
                logger.info(f"Initialized Memory Manager: {type(manager).__name__}")
            else:
                app.state.memory_manager = NullMemoryManager()
                logger.warning(
                    "No memory manager configuration found. Using NullMemoryManager."
                )
        except Exception as e:
            logger.error(f"Error initializing memory services: {e}")
            app.state.memory_manager = NullMemoryManager()
        yield
        app.state.memory_manager = None

    app = FastAPI(
        title="Pixelated Memory Server",
        description="Memory management with Mem0/Gemini integration",
        version="2.0.0",
        lifespan=lifespan,
    )

    def get_mcp_manager():
        """Get Memory Manager from app state."""
        return getattr(app.state, "memory_manager", NullMemoryManager())

    # ==================== MCP MEMORY TOOLS ====================

    @app.post("/api/memory/add", tags=["MCP Tools"])
    async def add_memory(request: AddMemoryRequest):
        """
        Add memory to Mem0 (MCP Tool: add_memory).

        Stores information in long-term memory with optional metadata.
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
                )

            scope = scope_from_kwargs(
                user_id=request.user_id,
                org_id=request.org_id,
                project_id=request.project_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                visibility=request.visibility or "private",
                include_shared=request.include_shared is not False,
            )
            metadata = build_scope_metadata(
                scope=scope,
                incoming_metadata=request.metadata,
                category=request.category,
            )

            # Call add_memory on the manager (GeminiMem0Manager or Wrapper)
            # Note: GeminiMem0Manager.add_memory arguments:
            # content, user_id, metadata=None, category=None
            memory_id = manager.add_memory(
                request.content,
                user_id=request.user_id,
                metadata=metadata or None,
                category=request.category,
            )

            if not memory_id:
                raise HTTPException(
                    status_code=400, detail="Memory rejected by safety filters"
                )

            return {
                "success": True,
                "memory_id": memory_id,
                "message": "Memory added successfully",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/search", tags=["MCP Tools"])
    async def search_memory(request: SearchMemoryRequest):
        """
        Search memories (MCP Tool: search_memory).

        Performs semantic search across user's memories.
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
                )

            scope = scope_from_kwargs(
                user_id=request.user_id,
                org_id=request.org_id,
                project_id=request.project_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                include_shared=request.include_shared is not False,
            )
            limit = request.limit or 10
            memories = search_with_overfetch(
                manager=manager,
                query=request.query,
                user_id=request.user_id,
                requested_limit=limit,
            )

            # Handle case where result might be a dict with 'results' key
            # (wrapper vs manager)
            # GeminiMem0Manager.search_memories already returns a list.
            # Wrapper returns client.search which might be list or dict.
            if isinstance(memories, dict) and "results" in memories:
                memories = memories["results"]

            memories = filter_memories_by_scope(
                scope=scope,
                memories=memories or [],
                limit=limit,
            )

            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.patch("/api/memory/{memory_id}", tags=["MCP Tools"])
    async def update_memory(memory_id: str, request: UpdateMemoryRequest):
        """
        Update an existing memory (MCP Tool: update_memory).

        Updates the content of an existing memory without creating duplicates.
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
                )

            scope = scope_from_kwargs(
                user_id=request.user_id,
                org_id=request.org_id,
                project_id=request.project_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                include_shared=request.include_shared is not False,
            )
            allowed = memory_in_scope(
                manager=manager,
                scope=scope,
                memory_id=memory_id,
            )
            if not allowed:
                raise HTTPException(
                    status_code=404,
                    detail="Memory not found in provided scope",
                )

            success = manager.update_memory(
                memory_id=memory_id,
                new_content=request.text,
                metadata=request.metadata,
            )

            # manager.update_memory returns bool.
            # Wrapper client.update returns dict/list/bool.
            if isinstance(success, (dict, list)):
                success = True

            if not success:
                raise HTTPException(status_code=400, detail="Update rejected or failed")

            return {"success": True, "message": "Memory updated"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/memory/{memory_id}", tags=["MCP Tools"])
    async def delete_memory_endpoint(
        memory_id: str,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
    ):
        """
        Delete a memory (MCP Tool: delete_memory).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
                )

            scope = scope_from_kwargs(
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
            )
            allowed = memory_in_scope(
                manager=manager,
                scope=scope,
                memory_id=memory_id,
            )
            if not allowed:
                raise HTTPException(
                    status_code=404,
                    detail="Memory not found in provided scope",
                )

            manager.delete_memory(memory_id=memory_id)
            return {"success": True, "message": "Memory deleted"}
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== LEGACY/COMPATIBILITY ROUTES ====================

    @app.post("/api/memory/users")
    async def create_user(
        email: str,
        name: str,
        role: str = "patient",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create new user (Mem0 compatibility shim)."""
        # Mem0 handles users implicitly via user_id. We'll use email as the user_id.
        return {
            "success": True,
            "user_id": email,
            "email": email,
            "name": name,
            "role": role,
            "message": "User registered for Mem0 context",
        }

    @app.get("/api/memory/users/{user_id}")
    async def get_user(user_id: str):
        """Get user profile (Shim: checks if user has memories)."""
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
                )

            # Check if we can fetch memories for this user
            memories = getattr(manager, "get_all_memories", lambda uid: [])(user_id)
            has_history = len(memories) > 0

            return {
                "success": True,
                "user_id": user_id,
                "has_history": has_history,
                "memory_count": len(memories) if isinstance(memories, list) else 0,
            }
        except Exception as e:
            logger.error(f"Error checking user: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== HEALTH CHECK ====================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            manager = get_mcp_manager()
            status = "healthy" if manager else "degraded"
            
            manager_type = str(type(manager))
            if "NvidiaMem0Manager" in manager_type:
                provider = "NvidiaMem0"
            elif "GeminiMem0Manager" in manager_type:
                provider = "GeminiMem0"
            else:
                provider = manager_type

            return {
                "status": status,
                "provider": provider,
                "service": "pixelated-memory-server",
                "version": "2.0.0",
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}, 503

    return app


# Create app instance
app = create_memory_server()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MEMORY_SERVER_PORT", 5003))
    uvicorn.run(app, host="0.0.0.0", port=port)
