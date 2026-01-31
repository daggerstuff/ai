#!/bin/bash
# Setup script for Pixel Voice Training Pipeline environment using uv/uvx

# 1. Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# 2. Create a virtual environment using uvx (or python -m venv as fallback)
if command -v uvx &> /dev/null; then
    echo "Creating virtual environment with uvx..."
    uvx venv .venv
else
    echo "uvx not found, using python -m venv as fallback..."
    python3 -m venv .venv
fi

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install Python dependencies with uv
uv pip install --upgrade pip
uv pip install -r requirements/pixel_voice_pipeline.txt

echo "Pixel Voice Training Pipeline environment setup complete (using uv/uvx)."