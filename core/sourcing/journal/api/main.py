"""
Main FastAPI application for Journal Dataset Research API.

This module creates the FastAPI app instance that is imported by both
the HTTP API server and the MCP server for compatibility.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from .config import get_settings
from .dependencies import get_command_handler_service
from .middleware.auth import AuthMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routers import api_router
from .websocket.routes import router as websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Journal Dataset Research API")

    # Initialize command handler service
    try:
        settings = get_settings()
        service = get_command_handler_service()
        logger.info("Command handler service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Journal Dataset Research API")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title="Journal Dataset Research API",
        description=(
            "API for journal dataset research operations including discovery, "
            "evaluation, acquisition, and integration planning."
        ),
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (in order)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Include API router
    app.include_router(api_router)

    # Include WebSocket routes
    app.include_router(websocket_router)

    # Add health check endpoint
    @app.get("/health", status_code=status.HTTP_200_OK, tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "journal-dataset-research-api"}

    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Journal Dataset Research API",
            "version": settings.api_version,
            "documentation": "/docs",
            "health": "/health",
        }

    return app


# Create the app instance for import
app = create_app()

# Export app for use by server.py and __init__.py
__all__ = ["app"]
