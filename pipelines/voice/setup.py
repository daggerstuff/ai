#!/usr/bin/env python3
"""
Setup script for Pixel Voice API and MCP server using uv.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def check_uv_installed():
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        print(f"✓ uv is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        return False


def install_uv():
    """Install uv if not present."""
    print("Installing uv...")
    try:
        # Try to install uv using the official installer
        result = subprocess.run(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh"], capture_output=True
        )

        if result.returncode != 0:
            # Fallback to pip installation
            return run_command([sys.executable, "-m", "pip", "install", "uv"])
        # Run the installer
        subprocess.run(["sh"], input=result.stdout, check=True)
        return True
    except Exception as e:
        print(f"Failed to install uv: {e}")
        return False


def create_virtual_environment():
    """Create a virtual environment using uv."""
    print("Creating virtual environment with uv...")

    venv_path = Path(".venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return True

    return run_command(["uv", "venv", ".venv"])


def install_dependencies():
    """Install required dependencies using uv."""
    print("Installing dependencies with uv...")

    # Install the project in development mode
    if not run_command(["uv", "pip", "install", "-e", "."]):
        print("Failed to install project dependencies")
        return False

    # Install development dependencies
    if not run_command(["uv", "pip", "install", "-e", ".[dev]"]):
        print("Warning: Failed to install development dependencies")

    # Install pipeline dependencies if requirements file exists
    pipeline_req = Path("requirements/pixel_voice_pipeline.txt")
    if pipeline_req.exists() and not run_command(
        ["uv", "pip", "install", "-r", str(pipeline_req)]
    ):
        print("Warning: Failed to install pipeline dependencies")

    return True


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")

    directories = [
        "data/voice_raw",
        "data/voice_segments",
        "data/voice_transcripts",
        "data/voice_transcripts_filtered",
        "data/voice",
        "data/dialogue_pairs",
        "data/therapeutic_pairs",
        "data/voice_consistency",
        "data/voice_optimized",
        "logs",
        "reports",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

    return True


def create_env_file():
    """Create a default .env file if it doesn't exist."""
    env_file = Path(".env")

    if env_file.exists():
        print(".env file already exists")
        return True

    print("Creating default .env file...")

    env_content = """# Pixel Voice Configuration
PIXEL_VOICE_ENV=development
PIXEL_VOICE_DEBUG=true

# API Server Configuration
PIXEL_VOICE_API_HOST=0.0.0.0
PIXEL_VOICE_API_PORT=8000
PIXEL_VOICE_API_RELOAD=true

# MCP Server Configuration
PIXEL_VOICE_MCP_HOST=localhost
PIXEL_VOICE_MCP_PORT=8001

# Whisper Configuration
PIXEL_VOICE_WHISPER_MODEL=large-v2
PIXEL_VOICE_WHISPER_LANGUAGE=en
PIXEL_VOICE_WHISPER_BATCH_SIZE=4
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    print("Created .env file with default configuration")
    return True


def make_scripts_executable():
    """Make startup scripts executable."""
    scripts = ["start_api.py", "start_mcp.py"]

    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            os.chmod(script_path, 0o755)
            print(f"Made {script} executable")


def verify_installation():
    """Verify that the installation is working."""
    import importlib.util

    print("Verifying installation...")

    # Test FastAPI dependencies
    fastapi_available = all(
        importlib.util.find_spec(module) is not None
        for module in ["fastapi", "pydantic", "uvicorn"]
    )
    if fastapi_available:
        print("✓ FastAPI dependencies available")
    else:
        print("✗ FastAPI dependencies missing")
        return False

    # Test MCP import (might not be available)
    if importlib.util.find_spec("mcp") is not None:
        print("✓ MCP dependencies available")
    else:
        print("⚠ MCP dependencies not available (install with: pip install mcp)")

    # Test pipeline dependencies
    if importlib.util.find_spec("whisperx") is not None:
        print("✓ WhisperX available")
    else:
        print("⚠ WhisperX not available (install with: pip install whisperx)")

    if importlib.util.find_spec("yt_dlp") is not None:
        print("✓ yt-dlp available")
    else:
        print("⚠ yt-dlp not available (install with: pip install yt-dlp)")

    return True


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("Pixel Voice API & MCP Server Setup Complete (using uv)!")
    print("=" * 70)

    print("\nTo start the services:")
    print("  API Server:  uv run python start_api.py")
    print("  MCP Server:  uv run python start_mcp.py")

    print("\nAlternatively (if virtual environment is activated):")
    print("  API Server:  python start_api.py")
    print("  MCP Server:  python start_mcp.py")

    print("\nOr using Docker:")
    print("  docker-compose up -d")

    print("\nAPI will be available at:")
    print("  http://localhost:8000")
    print("  Documentation: http://localhost:8000/docs")

    print("\nExample usage:")
    print("  uv run python examples/api_client_example.py")
    print("  uv run python examples/mcp_client_example.py")

    print("\nDevelopment commands:")
    print("  Install dev dependencies: uv pip install -e .[dev]")
    print("  Run tests: uv run pytest")
    print("  Format code: uv run black .")
    print("  Type check: uv run mypy pixel_voice/")

    print("\nConfiguration:")
    print("  Edit .env file to customize settings")
    print("  Check pixel_voice/api/config.py for all options")

    print("\nVirtual environment:")
    print("  Activate: source .venv/bin/activate")
    print("  Deactivate: deactivate")

    print("\nFor more information:")
    print("  See README.md for detailed documentation")


def main():
    """Main setup function."""
    print("Pixel Voice API & MCP Server Setup (using uv)")
    print("=" * 50)

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if uv is installed, install if needed
    if not check_uv_installed():
        print("uv not found, installing...")
        if not install_uv():
            print("✗ Failed to install uv")
            print("Please install uv manually: https://github.com/astral-sh/uv")
            sys.exit(1)
        print("✓ uv installed successfully")

    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Creating environment file", create_env_file),
        ("Making scripts executable", make_scripts_executable),
        ("Verifying installation", verify_installation),
    ]

    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"✗ {step_name} failed")
            sys.exit(1)
        print(f"✓ {step_name} completed")

    print_usage_instructions()


if __name__ == "__main__":
    main()
