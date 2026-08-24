#!/usr/bin/env bash
set -euo pipefail

echo "Cleaning __pycache__ and .pyc"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

echo "Done"
