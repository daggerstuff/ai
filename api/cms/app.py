"""CMS Business Strategy FastAPI application.

Provides REST API endpoints for documents, projects, strategies,
sales, approvals, and search. Uses CMSConnectionManager for
multi-database access (MongoDB, PostgreSQL, Redis).
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.infrastructure.database.cms_connection_manager import (
    CMSDatabaseConfig,
    close_cms_connection_manager,
    get_cms_connection_manager,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connections on startup, close on shutdown."""
    config = CMSDatabaseConfig.from_env()
    manager = await get_cms_connection_manager(config)
    app.state.cms_db = manager
    logger.info("CMS API started")
    yield
    await close_cms_connection_manager()
    logger.info("CMS API shut down")


def create_app() -> FastAPI:
    """Create and configure the CMS FastAPI application."""
    app = FastAPI(
        title="Pixelated CMS Business Strategy API",
        version="1.0",
        lifespan=lifespan,
    )

    _add_cors(app)
    _add_error_handlers(app)
    _register_routes(app)

    return app


def _add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(404)
    async def not_found(_request: Request, _exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Resource not found"},
            },
        )

    @app.exception_handler(422)
    async def validation_error(_request: Request, _exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {"code": "VALIDATION_ERROR", "message": "Request validation failed"},
            },
        )

    @app.exception_handler(500)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Internal error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"},
            },
        )


def _register_routes(app: FastAPI) -> None:
    from ai.api.cms.routes import documents, knowledge, projects, sales, strategies, workflows

    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
    app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
    app.include_router(sales.router, prefix="/api/v1/sales", tags=["Sales"])
    app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["Workflows"])
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["Knowledge"])

    @app.get("/api/v1/search")
    async def global_search(request: Request, q: str, limit: int = 10) -> dict[str, Any]:
        from ai.api.cms.services.search_service import SearchService

        service = SearchService(request.app.state.cms_db.mongo.db)
        results = await service.search(q, limit_per_collection=limit)
        return {"success": True, "data": results}

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        manager = request.app.state.cms_db
        return await manager.health_check()


app = create_app()
