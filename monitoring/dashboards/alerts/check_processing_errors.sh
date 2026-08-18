#!/bin/bash
# Alert monitoring script for processing_errors
# Threshold: error_rate > 5%
# Severity: medium

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --check-processing-errors
