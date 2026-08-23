#!/bin/bash
# Automated monthly_report generation script
cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --generate-monthly-report
