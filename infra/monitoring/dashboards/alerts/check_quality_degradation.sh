#!/bin/bash
# Alert monitoring script for quality_degradation
# Threshold: quality_score < 40
# Severity: high

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --check-quality-degradation
