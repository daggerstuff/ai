#!/bin/bash
# Export script for json_api
# Format: json
# Use case: system_integration

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --export-json-api
