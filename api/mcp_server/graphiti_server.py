"""
Graphiti Context Server.

Provides Server-Sent Events (SSE) endpoint for graphiti-context integration.
This server handles real-time context streaming for VS Code extensions.
"""


import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


def create_graphiti_server() -> FastAPI:
    """
    Create FastAPI server for graphiti-context.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Graphiti Context Server",
        description="Server-Sent Events endpoint for graphiti-context",
        version="1.0.0",
    )

    @app.on_event("startup")
    async def startup():
        """Initialize server on startup."""
        logger.info("Graphiti Context Server started")

    @app.get("/sse")
    async def sse() -> StreamingResponse:
        """
        Server-Sent Events endpoint for graphiti-context.

        Returns:
            StreamingResponse with SSE data
        """

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events."""
            try:
                while True:
                    # Send heartbeat event to keep connection alive
                    event_data = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "connected",
                    }
                    yield f"data: {event_data}\n\n"

                    # Send event every 30 seconds
                    await asyncio.sleep(30)

            except asyncio.CancelledError:
                logger.info("SSE connection closed")
                raise
            except Exception as e:
                logger.error(f"Error in SSE event generator: {e}")
                raise

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "graphiti-context",
        }

    @app.get("/")
    async def root() -> dict:
        """Root endpoint."""
        return {
            "service": "graphiti-context",
            "version": "1.0.0",
            "endpoints": {"/sse": "Server-Sent Events", "/health": "Health check"},
        }

    return app


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get configuration from environment
    host = os.environ.get("GRAPHITI_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPHITI_PORT", "8000"))
    debug = os.environ.get("GRAPHITI_DEBUG", "False").lower() == "true"

    # Create and run app
    app = create_graphiti_server()
    uvicorn.run(app, host=host, port=port, log_level="info")
