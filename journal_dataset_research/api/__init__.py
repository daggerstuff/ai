"""
API module for Journal Dataset Research System.

This module provides a FastAPI-based HTTP API server that wraps the CommandHandler
functionality for use by web frontends and other clients.
"""

from ai.journal_dataset_research.api.main import app

__all__ = ["app"]

