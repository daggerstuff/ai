"""
Pixelated Empathy AI - ASGI Entry Point

Exports a Starlette ASGI app for production deployment.
FastAPI/Starlette are ASGI-native and work with any ASGI server.
"""
import logging
from contextlib import asynccontextmanager

# Activate subconscious memory injection - auto-injects context into ALL LLM calls
import ai.memory.subconscious_autopatch  # noqa: F401

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Global reflection bootstrap instance
_reflection_bootstrap = None


@asynccontextmanager
async def lifespan(app):
    """Manage application lifespan - startup and shutdown."""
    global _reflection_bootstrap

    # Startup: initialize reflection subagent
    logger.info("Starting up - initializing reflection subagent...")
    try:
        from ai.memory.reflection_bootstrap import ReflectionBootstrap
        _reflection_bootstrap = await ReflectionBootstrap.create_and_start()
        logger.info("Reflection subagent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize reflection subagent: {e}")
        _reflection_bootstrap = None

    yield

    # Shutdown: clean up reflection subagent
    if _reflection_bootstrap:
        await _reflection_bootstrap.stop()
        logger.info("Reflection subagent stopped")


async def root(request):
    return JSONResponse({"status": "ok", "service": "pixelated-empathy-ai"})


async def health(request):
    del request  # Mark as intentionally unused
    return JSONResponse({"status": "healthy"})


async def reflect(request):
    """
    Trigger reflection on-demand.

    POST /reflect with body:
    {
        "conversation_text": "...",
        "user_id": "user-123"
    }
    """
    if request.method != 'POST':
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    try:
        body = await request.json()
        conversation_text = body.get("conversation_text")
        user_id = body.get("user_id", "unknown")

        if not conversation_text:
            return JSONResponse({"error": "conversation_text required"}, status_code=400)

        if _reflection_bootstrap is None:
            return JSONResponse({"error": "Reflection subagent not initialized"}, status_code=503)

        result = await _reflection_bootstrap.reflect_now(
            conversation_text=conversation_text,
            user_id=user_id,
        )

        return JSONResponse({
            "status": "success",
            "crisis_detected": result.crisis_detected,
            "requires_review": result.requires_manual_review,
            "memories_preserved": len(result.memories_preserved),
            "memories_consolidated": len(result.memories_consolidated),
        })
    except Exception as e:
        logger.error(f"Reflection error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/reflect", reflect, methods=["POST"]),
]

middleware = [
    # NOTE: For production, restrict allow_origins to known frontend domains
    # Current wildcard is for development convenience
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
