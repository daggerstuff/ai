"""
MCP Memory Integration Server.

Provides MCP (Model Control Protocol) compatible endpoints for memory operations
with Zep Cloud backend.
"""


import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query

from ai.api.memory import (
    MemoryType,
    MessageRole,
    UserRole,
    UserStatus,
    get_memory_manager,
    get_zep_manager,
)

logger = logging.getLogger(__name__)


def create_memory_server() -> FastAPI:
    """
    Create FastAPI server for memory operations.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Pixelated Memory Server",
        description="Memory management with Zep Cloud integration",
        version="1.0.0",
    )

    # Initialize managers at startup
    @app.on_event("startup")
    async def startup():
        """Initialize memory services on startup."""
        try:
            api_key = os.environ.get("ZEP_API_KEY")
            if not api_key:
                logger.error("ZEP_API_KEY environment variable not set")
                raise ValueError("ZEP_API_KEY required")

            # Get managers
            get_zep_manager(api_key=api_key)
            logger.info("Memory services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory services: {e}")
            raise

    # ==================== USER MANAGEMENT ====================

    @app.post("/api/memory/users")
    async def create_user(
        email: str,
        name: str,
        role: str = "patient",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create new user with Zep."""
        try:
            manager = get_zep_manager()

            user_role = UserRole(role)
            profile = manager.create_user(
                email=email, name=name, role=user_role, metadata=metadata
            )

            return {
                "success": True,
                "user_id": profile.user_id,
                "email": profile.email,
                "name": profile.name,
                "role": profile.role.value,
                "created_at": profile.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/users/{user_id}")
    async def get_user(user_id: str):
        """Get user profile."""
        try:
            manager = get_zep_manager()
            profile = manager.get_user(user_id)

            if not profile:
                raise HTTPException(status_code=404, detail="User not found")

            return {
                "success": True,
                "user_id": profile.user_id,
                "email": profile.email,
                "name": profile.name,
                "role": profile.role.value,
                "status": profile.status.value,
                "last_login": profile.last_login.isoformat()
                if profile.last_login
                else None,
            }
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/memory/users/{user_id}")
    async def update_user(
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ):
        """Update user profile."""
        try:
            manager = get_zep_manager()

            user_status = UserStatus(status) if status else None
            success = manager.update_user(
                user_id=user_id, name=name, metadata=metadata, status=user_status
            )

            if not success:
                raise HTTPException(status_code=404, detail="User not found")

            return {"success": True, "message": "User updated"}
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== SESSION MANAGEMENT ====================

    @app.post("/api/memory/sessions")
    async def create_session(
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout_minutes: int = 30,
    ):
        """Create user session."""
        try:
            manager = get_zep_manager()
            session = manager.create_session(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                session_timeout_minutes=timeout_minutes,
            )

            return {
                "success": True,
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session info."""
        try:
            manager = get_zep_manager()
            session = manager.get_session(session_id)

            if not session:
                raise HTTPException(
                    status_code=404, detail="Session not found or expired"
                )

            return {
                "success": True,
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/sessions/{session_id}/close")
    async def close_session(session_id: str):
        """Close user session."""
        try:
            manager = get_zep_manager()
            success = manager.close_session(session_id)

            if not success:
                raise HTTPException(status_code=404, detail="Session not found")

            return {"success": True, "message": "Session closed"}
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/users/{user_id}/sessions")
    async def list_user_sessions(user_id: str):
        """List active sessions for user."""
        try:
            manager = get_zep_manager()
            sessions = manager.list_user_sessions(user_id)

            return {
                "success": True,
                "user_id": user_id,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "created_at": s.created_at.isoformat(),
                        "expires_at": s.expires_at.isoformat(),
                        "last_activity": s.last_activity.isoformat(),
                    }
                    for s in sessions
                ],
            }
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== MEMORY OPERATIONS ====================

    @app.post("/api/memory/messages")
    async def add_message(
        user_id: str,
        session_id: str,
        content: str,
        role: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add message to session memory."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            msg_role = MessageRole(role)
            mem_type = MemoryType(memory_type)

            success = mem_manager.add_message(
                user_id=user_id,
                session_id=session_id,
                content=content,
                role=msg_role,
                memory_type=mem_type,
                metadata=metadata,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to add message")

            return {"success": True, "message": "Message added to memory"}
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/conversations/{session_id}")
    async def get_conversation(session_id: str, limit: int = Query(50, ge=1, le=200)):
        """Get conversation history."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            # Get a user ID from session if needed
            messages = mem_manager.get_conversation_history(
                user_id="",  # Can be derived from session
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

    @app.post("/api/memory/sessions/{session_id}/summary")
    async def store_session_summary(
        session_id: str,
        user_id: str,
        summary: str,
        key_points: List[str],
        emotional_insights: Dict[str, Any],
        next_steps: List[str],
    ):
        """Store session summary."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            success = mem_manager.store_session_summary(
                user_id=user_id,
                session_id=session_id,
                summary=summary,
                key_points=key_points,
                emotional_insights=emotional_insights,
                next_steps=next_steps,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to store summary")

            return {"success": True, "message": "Summary stored"}
        except Exception as e:
            logger.error(f"Error storing summary: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/sessions/{session_id}/emotional-state")
    async def get_emotional_state(session_id: str, user_id: str):
        """Get emotional state."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            emotional_state = mem_manager.get_emotional_state(
                user_id=user_id, session_id=session_id
            )

            return {
                "success": True,
                "session_id": session_id,
                "emotional_state": emotional_state,
            }
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/sessions/{session_id}/emotional-state")
    async def store_emotional_state(
        session_id: str,
        user_id: str,
        emotions: Dict[str, float],
        context: str,
        triggers: Optional[List[str]] = None,
    ):
        """Store emotional state."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            success = mem_manager.store_emotional_state(
                user_id=user_id,
                session_id=session_id,
                emotions=emotions,
                context=context,
                triggers=triggers,
            )

            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to store emotional state"
                )

            return {"success": True, "message": "Emotional state stored"}
        except Exception as e:
            logger.error(f"Error storing emotional state: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/sessions/{session_id}/treatment-plan")
    async def get_treatment_plan(session_id: str, user_id: str):
        """Get treatment plan."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            plan = mem_manager.get_treatment_plan(
                user_id=user_id, session_id=session_id
            )

            return {"success": True, "session_id": session_id, "treatment_plan": plan}
        except Exception as e:
            logger.error(f"Error getting treatment plan: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/sessions/{session_id}/treatment-plan")
    async def store_treatment_plan(
        session_id: str,
        user_id: str,
        goals: List[str],
        interventions: List[str],
        progress_metrics: Dict[str, Any],
        duration_weeks: int,
    ):
        """Store treatment plan."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            success = mem_manager.store_treatment_plan(
                user_id=user_id,
                session_id=session_id,
                goals=goals,
                interventions=interventions,
                progress_metrics=progress_metrics,
                duration_weeks=duration_weeks,
            )

            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to store treatment plan"
                )

            return {"success": True, "message": "Treatment plan stored"}
        except Exception as e:
            logger.error(f"Error storing treatment plan: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/memory/sessions/{session_id}/stats")
    async def get_memory_stats(session_id: str):
        """Get memory statistics."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            stats = mem_manager.get_memory_stats(session_id)

            return {"success": True, "session_id": session_id, "stats": stats}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/memory/sessions/{session_id}/clear")
    async def clear_session_memory(session_id: str):
        """Clear session memory (privacy)."""
        try:
            manager = get_zep_manager()
            mem_manager = get_memory_manager(manager.client)

            success = mem_manager.clear_session_memory(session_id)

            if not success:
                raise HTTPException(status_code=500, detail="Failed to clear memory")

            return {"success": True, "message": "Session memory cleared"}
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== HEALTH CHECK ====================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            get_zep_manager()
            return {
                "status": "healthy",
                "service": "pixelated-memory-server",
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
