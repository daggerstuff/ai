#!/bin/bash
# Alert monitoring script for data_anomalies
# Threshold: anomaly_score > 0.8
# Severity: low

cd /home/vivi/pixelated/ai
uv run monitoring/operational_dashboard_system.py --check-data-anomalies
