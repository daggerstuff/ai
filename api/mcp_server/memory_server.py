"""
MCP Memory Integration Server.

Provides MCP (Model Control Protocol) compatible endpoints for memory operations
with Mem0 backend. Implements full Mem0 MCP tools including:
- add_memory, search_memory, get_all_memory
- update_memory, delete_memory
- list_entities, delete_entity

Based on: https://docs.mem0.ai/platform/mem0-mcp
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ai.api.memory import (
    MemoryType,
    MessageRole,
    get_memory_manager,
)
from ai.api.memory.null_memory import NullMemoryManager

logger = logging.getLogger(__name__)


# ==================== Pydantic Models ====================


class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    content: str
    user_id: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateMemoryRequest(BaseModel):
    """Request model for updating memory."""

    text: str
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    """Request model for searching memories."""

    query: str
    user_id: str
    limit: int = 10


class AddMessageRequest(BaseModel):
    """Request model for adding a message."""

    user_id: str
    session_id: str
    content: str
    role: str
    memory_type: str = "conversation"
    metadata: Optional[Dict[str, Any]] = None


class SessionSummaryRequest(BaseModel):
    """Request model for storing session summary."""

    user_id: str
    summary: str
    key_points: List[str] = []
    emotional_insights: Dict[str, Any] = {}
    next_steps: List[str] = []


def create_memory_server() -> FastAPI:
    """
    Create FastAPI server for memory operations.

    Returns:
        Configured FastAPI application
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize memory services on startup."""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        mem0_key = os.environ.get("MEM0_API_KEY")

        try:
            # 1. Try GeminiMem0Manager (Preferred)
            if gemini_key:
                from ai.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager

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

                    def add_memory(self, content, user_id, metadata=None, **kwargs):
                        res = self.client.add(content, user_id=user_id, metadata=metadata)
                        # Normalize return ID
                        if isinstance(res, dict) and "results" in res and res["results"]:
                            return res["results"][0].get("id")
                        if isinstance(res, list) and res:
                            return res[0].get("id")
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
                raise ImportWarning("No API keys found")

        except Exception as e:
            logger.warning(f"Using NullMemoryManager (fallback): {e}")
            app.state.memory_manager = NullMemoryManager()

        yield

        # Cleanup if needed
        app.state.memory_manager = None

    app = FastAPI(
        title="Pixelated Memory Server",
        description="Memory management with Mem0 integration - Full MCP Tools",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
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
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            metadata = request.metadata or {}
            if request.session_id:
                metadata["session_id"] = request.session_id
            if request.agent_id:
                metadata["agent_id"] = request.agent_id

            memory_id = manager.add_memory(
                request.content,
                user_id=request.user_id,
                metadata=metadata or None,
                category=request.category,
            )

            if not memory_id:
                raise HTTPException(status_code=400, detail="Memory rejected by safety filters")

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
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            memories = manager.search_memories(
                request.query,
                user_id=request.user_id,
            )

            # Handle case where result might be a dict with 'results' key (wrapper vs manager)
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

    @app.get("/api/memory/all/{user_id}", tags=["MCP Tools"])
    async def get_all_memories(
        user_id: str,
        limit: int = Query(100, ge=1, le=500),
    ):
        """
        Get all memories for a user (MCP Tool: get_all_memory).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = manager.get_all_memories(user_id=user_id)

            # Unpack if needed
            memories = []
            if isinstance(result, dict) and "results" in result:
                memories = result["results"]
            elif isinstance(result, list):
                memories = result

            return {
                "success": True,
                "memories": memories[:limit],
                "total": len(memories),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/{memory_id}", tags=["MCP Tools"])
    async def get_memory(memory_id: str):
        """
        Get a specific memory by ID (MCP Tool: get_memory).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = manager.get_memory(memory_id=memory_id)

            if not result:
                raise HTTPException(status_code=404, detail="Memory not found")

            return {"success": True, "memory": result}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
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
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            success = manager.update_memory(
                memory_id=memory_id, new_content=request.text, metadata=request.metadata
            )

            # manager.update_memory returns bool. Wrapper client.update returns dict.
            # Handle both cases if needed, but mainly bool for GeminiManager.

            # If wrapper returns dict, assume success for now or check contents
            if isinstance(success, dict):
                success = True

            if not success:
                raise HTTPException(status_code=400, detail="Update rejected or failed")

            return {
                "success": True,
                "memory_id": memory_id,
                "message": "Memory updated successfully",
            }
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/memory/{memory_id}", tags=["MCP Tools"])
    async def delete_memory(memory_id: str):
        """
        Delete a specific memory (MCP Tool: delete_memory).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            manager.delete_memory(memory_id=memory_id)

            return {
                "success": True,
                "memory_id": memory_id,
                "message": "Memory deleted successfully",
            }
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/memory/user/{user_id}", tags=["MCP Tools"])
    async def delete_all_user_memories(user_id: str):
        """
        Delete all memories for a user (MCP Tool: delete_all).
        """
        try:
            manager = get_mcp_manager()
            if not manager:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            if hasattr(manager, "clear_memory"):
                manager.clear_memory(user_id=user_id)
            elif hasattr(manager, "client") and hasattr(manager.client, "delete_all"):
                # Wrapper fallback
                manager.client.delete_all(user_id=user_id)

            return {
                "success": True,
                "user_id": user_id,
                "message": "All user memories deleted",
            }
        except Exception as e:
            logger.error(f"Error deleting all memories: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== LEGACY MEMORY OPERATIONS ====================

    @app.post("/api/memory/messages", tags=["Legacy"])
    async def add_message(request: AddMessageRequest):
        """Add message to session memory (legacy endpoint)."""
        try:
            mem_manager = get_memory_manager()

            msg_role = MessageRole(request.role)
            mem_type = MemoryType(request.memory_type)

            success = mem_manager.add_message(
                user_id=request.user_id,
                session_id=request.session_id,
                content=request.content,
                role=msg_role,
                memory_type=mem_type,
                metadata=request.metadata,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to add message")

            return {"success": True, "message": "Message added to memory"}
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/conversations/{session_id}", tags=["Legacy"])
    async def get_conversation(user_id: str, session_id: str, limit: int = Query(50, ge=1, le=200)):
        """Get conversation history."""
        try:
            mem_manager = get_memory_manager()

            messages = mem_manager.get_conversation_history(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
            )

            return {
                "success": True,
                "session_id": session_id,
                "messages": [
                    {
                        "content": msg.content,
                        "role": msg.role.value,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata,
                    }
                    for msg in messages
                ],
            }
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/sessions/{session_id}/summary", tags=["Legacy"])
    async def store_session_summary(session_id: str, request: SessionSummaryRequest):
        """Store session summary."""
        try:
            mem_manager = get_memory_manager()

            success = mem_manager.store_session_summary(
                user_id=request.user_id,
                session_id=session_id,
                summary=request.summary,
                key_points=request.key_points,
                emotional_insights=request.emotional_insights,
                next_steps=request.next_steps,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to store summary")

            return {"success": True, "message": "Summary stored"}
        except Exception as e:
            logger.error(f"Error storing summary: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== HEALTH & INFO ====================

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "pixelated-memory-server",
            "provider": "mem0",
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/memory/tools", tags=["Info"])
    async def list_mcp_tools():
        """List available MCP memory tools."""
        return {
            "tools": [
                {"name": "add_memory", "method": "POST", "path": "/api/memory/add"},
                {"name": "search_memory", "method": "POST", "path": "/api/memory/search"},
                {"name": "get_all_memory", "method": "GET", "path": "/api/memory/all/{user_id}"},
                {"name": "get_memory", "method": "GET", "path": "/api/memory/{memory_id}"},
                {"name": "update_memory", "method": "PATCH", "path": "/api/memory/{memory_id}"},
                {"name": "delete_memory", "method": "DELETE", "path": "/api/memory/{memory_id}"},
                {"name": "delete_all", "method": "DELETE", "path": "/api/memory/user/{user_id}"},
            ]
        }

    return app


# Create app instance
app = create_memory_server()


def run_server():
    """Main entry point for running the memory server CLI."""
    import uvicorn

    port = int(os.environ.get("MEMORY_SERVER_PORT", 5003))
    uvicorn.run("ai.api.mcp_server.memory_server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    run_server()
