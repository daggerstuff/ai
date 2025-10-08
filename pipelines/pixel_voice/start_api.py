#!/usr/bin/env python3
"""
Startup script for the Pixel Voice API server using uv.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path


def check_uv_environment():
    """Check if we're running in a uv environment and activate if needed."""
    venv_path = Path(".venv")
    if venv_path.exists() and "VIRTUAL_ENV" not in os.environ:
        print("Activating uv virtual environment...")
        # Use uv run to execute in the virtual environment
        return True
    return False


def main():
    """Start the FastAPI server using uv."""
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if we need to use uv run
    if check_uv_environment():
        # Use uv run to start the server
        cmd = [
            "uv",
            "run",
            "uvicorn",
            "pixel_voice.api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
        print(f"Starting API server with uv: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        # Direct execution (fallback)
        import uvicorn

        # Add the parent directory to the Python path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from pixel_voice.api.config import config

        # Ensure all directories exist
        config.ensure_directories()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.api.log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        logger = logging.getLogger(__name__)
        logger.info("Starting Pixel Voice API server...")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")
        logger.info(f"Server will run on {config.api.host}:{config.api.port}")

        # Start the server
        uvicorn.run(
            "pixel_voice.api.server:app",
            host=config.api.host,
            port=config.api.port,
            reload=config.api.reload,
            log_level=config.api.log_level,
            access_log=True,
        )


if __name__ == "__main__":
    main()
