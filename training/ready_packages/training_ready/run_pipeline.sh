#!/bin/bash
# Wrapper script to run training_ready pipeline with uv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "🚀 Running pipeline with uv..."
    uv run python3 "$SCRIPT_DIR/scripts/prepare_training_data.py" "$@"
else
    echo "⚠️  uv not found, using system python3..."
    python3 "$SCRIPT_DIR/scripts/prepare_training_data.py" "$@"
fi

