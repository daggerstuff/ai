"""
API module for Journal Dataset Research System.

This module provides a FastAPI-based HTTP API server that wraps the CommandHandler
functionality for use by web frontends and other clients.

Note: The FastAPI app is created lazily via create_app() to avoid heavy imports
for consumers that only need the service layer (e.g., MCP server).
"""

from .services.command_handler_service import CommandHandlerService

__all__ = ["CommandHandlerService", "create_app"]


def create_app():
    """Create and return the FastAPI application."""
    from .main import app

    return app
