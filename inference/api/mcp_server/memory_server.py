"""
MCP Memory Integration Server.

Provides MCP-compatible memory operations and Foresight-compatible local routes.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai.inference.api.mcp_server.memory_auth import validate_memory_auth_configuration
from ai.inference.api.mcp_server.routes import (
    create_foresight_router,
    create_health_router,
    create_legacy_router,
    create_mcp_router,
)
from ai.research.manager_factory import get_required_memory_manager

logger = logging.getLogger(__name__)


def create_memory_server() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        validate_memory_auth_configuration()
        manager = get_required_memory_manager()
        app.state.memory_manager = manager
        logger.info(f"Initialized Memory Manager: {type(manager).__name__}")
        yield
        manager.close()
        app.state.memory_manager = None

    app = FastAPI(
        title="Pixelated Memory Server",
        description="Memory management with shared local Foresight compatibility",
        version="2.0.0",
        lifespan=lifespan,
    )

    def get_mcp_manager():
        return getattr(app.state, "memory_manager", None)

    app.include_router(create_mcp_router(get_mcp_manager))
    app.include_router(create_foresight_router(get_mcp_manager))
    app.include_router(create_legacy_router(get_mcp_manager))
    app.include_router(create_health_router(get_mcp_manager))

    return app


app = create_memory_server()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MEMORY_SERVER_PORT", 54321))
    uvicorn.run(app, host="0.0.0.0", port=port)
