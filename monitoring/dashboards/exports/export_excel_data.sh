#!/bin/bash
# Export script for excel_data
# Format: xlsx
# Use case: data_analysis

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --export-excel-data
