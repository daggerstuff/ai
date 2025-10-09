#!/usr/bin/env python3
"""
Startup script for the Pixel Voice MCP server using uv.
"""
import asyncio
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
        return True
    return False


def main():
    """Start the MCP server using uv."""
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if we need to use uv run
    if check_uv_environment():
        # Use uv run to start the MCP server
        cmd = ["uv", "run", "python", "-m", "pixel_voice.mcp_server"]
        print(f"Starting MCP server with uv: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        # Direct execution (fallback)
        # Add the parent directory to the Python path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from pixel_voice.api.config import config

        # Ensure all directories exist
        config.ensure_directories()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        logger = logging.getLogger(__name__)
        logger.info("Starting Pixel Voice MCP server...")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")

        # Import and run the MCP server
        from pixel_voice.mcp_server import main as mcp_main

        asyncio.run(mcp_main())


if __name__ == "__main__":
    main()
