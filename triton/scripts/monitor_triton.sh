#!/bin/bash
# Monitor Triton Inference Server performance and health

set -euo pipefail

HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
ALERT_THRESHOLD="${ALERT_THRESHOLD:-0.8}"

# Create log directory
mkdir -p "$LOG_DIR"
MONITOR_LOG="$LOG_DIR/triton_monitor_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

alert() {
    echo "[ALERT] $*" | tee -a "$MONITOR_LOG"
}

log "=== Triton Monitoring Started ==="
log "HTTP Port: $HTTP_PORT"
log "Metrics Port: $METRICS_PORT"
log "Monitor Interval: ${MONITOR_INTERVAL}s"
log "Alert Threshold: ${ALERT_THRESHOLD}"

# Initialize metrics
declare -A previous_metrics

collect_metrics() {
    local timestamp=$(date +%s)
    
    # Get server status
    local server_status=$(curl -s "http://localhost:$HTTP_PORT/v2/health/ready" 2>/dev/null || echo "unavailable")
    
    # Get model stats
    local model_stats=$(curl -s "http://localhost:$HTTP_PORT/v2/models/pixel/stats" 2>/dev/null || echo "{}")
    
    # Get GPU metrics
    local gpu_util=0
    local gpu_mem=0
    if command -v nvidia-smi &> /dev/null; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | sed 's/%//')
        gpu_mem=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader | head -1 | sed 's/%//')
    fi
    
    # Get Prometheus metrics
    local prometheus_metrics=$(curl -s "http://localhost:$METRICS_PORT/metrics" 2>/dev/null || echo "")
    
    # Extract key metrics
    local request_count=$(echo "$prometheus_metrics" | grep "nv_inference_request_total" | awk '{print $2}' | head -1)
    local inference_count=$(echo "$prometheus_metrics" | grep "nv_inference_exec_count" | awk '{print $2}' | head -1)
    local gpu_used=$(echo "$prometheus_metrics" | grep "nv_gpu_utilization" | awk '{print $2}' | head -1)
    
    log "=== Metrics Snapshot ==="
    log "Timestamp: $timestamp"
    log "Server Status: $server_status"
    log "GPU Utilization: ${gpu_util}%"
    log "GPU Memory: ${gpu_mem}%"
    log "Request Count: ${request_count:-N/A}"
    log "Inference Count: ${inference_count:-N/A}"
    
    # Check thresholds
    if (( ${gpu_util%.*} > ${ALERT_THRESHOLD%.*} * 100 )); then
        alert "High GPU utilization: ${gpu_util}%"
    fi
    
    if (( ${gpu_mem%.*} > ${ALERT_THRESHOLD%.*} * 100 )); then
        alert "High GPU memory usage: ${gpu_mem}%"
    fi
    
    # Store metrics for trending
    echo "$timestamp,$gpu_util,$gpu_mem,$request_count,$inference_count" >> "$LOG_DIR/metrics.csv"
}

# Main monitoring loop
log "Starting continuous monitoring..."
while true; do
    collect_metrics
    sleep "$MONITOR_INTERVAL"
done
