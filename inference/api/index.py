"""
Pixelated Empathy AI - ASGI Entry Point

Exports a Starlette ASGI app for production deployment.
FastAPI/Starlette are ASGI-native and work with any ASGI server.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from ai.inference.api.emotions_service import app as emotions_app
from ai.inference.api.mcp_server.memory_auth import authorize_memory_access
from ai.research.reflection_bootstrap import create_and_start

logger = logging.getLogger(__name__)

# Global reflection bootstrap instance
_reflection_bootstrap = None


def _parse_reflection_body(raw_body: bytes) -> tuple[str, str]:
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc

    conversation_text = body.get("conversation_text")
    raw_user_id = body.get("user_id")
    if raw_user_id is None:
        raise ValueError("user_id required")
    user_id = str(raw_user_id).strip()
    if not conversation_text:
        raise ValueError("conversation_text required")
    if not user_id:
        raise ValueError("user_id required")
    return conversation_text, user_id


def _request_target_for(request) -> str:
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return target


def _authorize_reflection_request(request, raw_body: bytes, user_id: str) -> str:
    header_user_id = request.headers.get("X-Memory-User-Id")
    if not header_user_id:
        raise ValueError("Missing X-Memory-User-Id header")
    normalized_header_user_id = header_user_id.strip()
    if normalized_header_user_id != user_id:
        raise ValueError("X-Memory-User-Id must match the requested user scope")
    auth_context = authorize_memory_access(
        actor_id=request.headers.get("X-Memory-Actor-Id"),
        user_id=normalized_header_user_id,
        request_method=request.method,
        request_target=_request_target_for(request),
        request_body=raw_body,
        timestamp=request.headers.get("X-Memory-Timestamp"),
        nonce=request.headers.get("X-Memory-Nonce"),
        signature=request.headers.get("X-Memory-Signature"),
    )
    return auth_context.assert_user_scope(normalized_header_user_id)


async def _run_reflection(request) -> JSONResponse:
    raw_body = await request.body()
    conversation_text, user_id = _parse_reflection_body(raw_body)

    if _reflection_bootstrap is None:
        return JSONResponse({"error": "Reflection subagent not initialized"}, status_code=503)

    authorized_user_id = _authorize_reflection_request(request, raw_body, user_id)

    result = await _reflection_bootstrap.reflect_now(
        conversation_text=conversation_text,
        user_id=authorized_user_id,
    )

    return JSONResponse(
        {
            "status": "success",
            "crisis_detected": result.crisis_detected,
            "requires_review": result.requires_manual_review,
            "memories_preserved": len(result.memories_preserved),
            "memories_consolidated": len(result.memories_consolidated),
        }
    )


@asynccontextmanager
async def lifespan(_app):
    """Manage application lifespan - startup and shutdown."""
    global _reflection_bootstrap

    # Startup: initialize reflection subagent
    logger.info("Starting up - initializing reflection subagent...")
    _reflection_bootstrap = await create_and_start()
    logger.info("Reflection subagent initialized successfully")

    yield

    # Shutdown: clean up reflection subagent
    if _reflection_bootstrap:
        await _reflection_bootstrap.stop()
        logger.info("Reflection subagent stopped")


async def root(_request):
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
    if request.method != "POST":
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    try:
        return await _run_reflection(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    except Exception as e:
        logger.error(f"Reflection error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


from starlette.routing import Mount

from ai.tools.utilities.platform.patient_psi.api import create_app

patient_psi_app = create_app(prefix="/api/v1/patient-psi")

routes = [
    Route("/", root),
    Route("/health", health),
    Route("/reflect", reflect, methods=["POST"]),
    Mount("/analyze", app=emotions_app),
    Mount("", app=patient_psi_app),
]


def get_cors_origins() -> list[str]:
    """Parse allowed CORS origins from the CORS_ORIGINS environment variable.

    Returns a list of allowed origins. Defaults to an empty list (no cross-origin
    requests allowed). Set CORS_ORIGINS as a comma-separated list of origins.
    """
    raw = os.getenv("CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


middleware = [
    # NOTE: For production, restrict allow_origins to known frontend domains
    # Must be configured explicitly via environment variables
    Middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
