"""
FastAPI application for Journal Dataset Research System.

This module provides the main FastAPI application with CORS, authentication,
and routing configuration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.sourcing.journal.api.config import get_settings
from ai.sourcing.journal.api.middleware.auth import AuthMiddleware
from ai.sourcing.journal.api.middleware.error_handler import (
    ErrorHandlerMiddleware,
)
from ai.sourcing.journal.api.middleware.logging import LoggingMiddleware
from ai.sourcing.journal.api.middleware.rate_limit import RateLimitMiddleware
from ai.sourcing.journal.api.routers import api_router
from ai.sourcing.journal.api.websocket import websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Journal Dataset Research API server")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Version: {settings.api_version}")
    yield
    # Shutdown
    logger.info("Shutting down Journal Dataset Research API server")


# Create FastAPI application
app = FastAPI(
    title="Journal Dataset Research API",
    description="API for managing journal dataset research operations",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/api/docs" if settings.environment != "production" else None,
    redoc_url="/api/redoc" if settings.environment != "production" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware (should be first)
app.add_middleware(ErrorHandlerMiddleware)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add rate limiting middleware (if enabled)
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware)

# Add authentication middleware (if enabled)
if settings.auth_enabled:
    app.add_middleware(AuthMiddleware)

# Include API router
app.include_router(api_router)

# Include WebSocket router
app.include_router(websocket_router)

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.api_version,
        "environment": settings.environment,
    }


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Journal Dataset Research API",
        "version": settings.api_version,
        "docs": "/api/docs" if settings.environment != "production" else "disabled",
    }

