#!/bin/bash
# Automated weekly_report generation script
cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --generate-weekly-report
