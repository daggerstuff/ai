#!/bin/bash
# Export script for powerpoint
# Format: pptx
# Use case: executive_presentations

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --export-powerpoint
