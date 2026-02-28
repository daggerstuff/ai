#!/bin/bash
# Export script for pdf_reports
# Format: pdf
# Use case: formal_reporting

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --export-pdf-reports
