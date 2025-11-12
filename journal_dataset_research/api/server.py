#!/usr/bin/env python3
"""
Server runner for Journal Dataset Research API.

This script starts the FastAPI server with uvicorn.
"""

import uvicorn
from ai.journal_dataset_research.api.config import get_settings

settings = get_settings()


def main() -> None:
    """Run the API server."""
    uvicorn.run(
        "ai.journal_dataset_research.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

