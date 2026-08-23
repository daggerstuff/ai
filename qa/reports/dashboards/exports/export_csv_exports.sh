#!/bin/bash
# Export script for csv_exports
# Format: csv
# Use case: data_sharing

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --export-csv-exports
