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

from ai.api.memory.null_memory import NullMemoryManager

logger = logging.getLogger(__name__)


# --- Request Models ---


class AddMemoryRequest(BaseModel):
    content: str
    user_id: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    limit: Optional[int] = 10


class UpdateMemoryRequest(BaseModel):
    text: str
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
        gemini_key = os.environ.get("GEMINI_API_KEY")
        mem0_key = os.environ.get("MEM0_API_KEY")

        try:
            # 1. Try GeminiMem0Manager (Preferred)
            if gemini_key:
                from ai.memory.mem0_gemini.manager import (
                    GeminiMem0Config,
                    GeminiMem0Manager,
                )

                config = GeminiMem0Config(
                    gemini_api_key=gemini_key,
                    mem0_api_key=mem0_key,
                    user_id="mcp_http_user",
                )
                app.state.memory_manager = GeminiMem0Manager(config)
                logger.info("Initialized GeminiMem0Manager (Enhanced Memory)")

            # 2. Try simple Mem0 Client
            elif mem0_key:
                from mem0 import MemoryClient

                # Wrap Mem0 client to match interface
                class Mem0Wrapper:
                    def __init__(self, client):
                        self.client = client

                    def add_memory(
                        self, content, user_id, metadata=None, category=None, **kwargs
                    ):
                        final_metadata = metadata or {}
                        if category:
                            final_metadata["category"] = category
                        res = self.client.add(
                            content, user_id=user_id, metadata=final_metadata
                        )
                        # Normalize return ID
                        if (
                            isinstance(res, dict)
                            and "results" in res
                            and res["results"]
                        ):
                            return res["results"][0].get("id")
                        elif isinstance(res, list) and res:
                            return res[0].get("id")
                        else:
                            return "stored"

                    def search_memories(self, query, user_id, **kwargs):
                        return self.client.search(query, user_id=user_id, **kwargs)

                    def get_all_memories(self, user_id):
                        return self.client.get_all(user_id=user_id)

                    def update_memory(self, memory_id, new_content, **kwargs):
                        return self.client.update(memory_id, new_content)

                    def delete_memory(self, memory_id):
                        return self.client.delete(memory_id)

                    def get_memory(self, memory_id):
                        return self.client.get(memory_id)

                app.state.memory_manager = Mem0Wrapper(MemoryClient(api_key=mem0_key))
                logger.info("Initialized Mem0 Platform Client (Basic Memory)")

            else:
                # 3. Fallback to NullMemory
                logger.warning("No API keys found for Memory Server")
                app.state.memory_manager = NullMemoryManager()

        except Exception as e:
            logger.warning(f"Using NullMemoryManager (fallback): {e}")
            app.state.memory_manager = NullMemoryManager()

        yield

        # Cleanup if needed
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

            metadata = request.metadata or {}
            if request.session_id:
                metadata["session_id"] = request.session_id
            if request.agent_id:
                metadata["agent_id"] = request.agent_id

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

            memories = manager.search_memories(
                request.query,
                user_id=request.user_id,
            )

            # Handle case where result might be a dict with 'results' key
            # (wrapper vs manager)
            # GeminiMem0Manager.search_memories already returns a list.
            # Wrapper returns client.search which might be list or dict.
            if isinstance(memories, dict) and "results" in memories:
                memories = memories["results"]

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
    async def delete_memory_endpoint(memory_id: str):
        """
        Delete a memory (MCP Tool: delete_memory).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Memory service unavailable"
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
            provider = (
                "GeminiMem0"
                if "GeminiMem0Manager" in str(type(manager))
                else str(type(manager))
            )

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
