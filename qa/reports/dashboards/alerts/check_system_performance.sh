#!/bin/bash
# Alert monitoring script for system_performance
# Threshold: response_time > 10s
# Severity: medium

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --check-system-performance
