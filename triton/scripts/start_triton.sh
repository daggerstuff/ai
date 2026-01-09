#!/bin/bash
# Start Triton Inference Server with Pixel model

set -euo pipefail

# Configuration
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
METRICS_PORT="${METRICS_PORT:-8002}"
GRPC_PORT="${GRPC_PORT:-8001}"
HTTP_PORT="${HTTP_PORT:-8000}"
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.8}"
TRITON_PROFILE="${TRITON_PROFILE:-production}"

# Logging
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/triton_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== Triton Inference Server Startup ==="
log "Model Repository: $MODEL_REPOSITORY"
log "HTTP Port: $HTTP_PORT"
log "gRPC Port: $GRPC_PORT"
log "Metrics Port: $METRICS_PORT"
log "Log File: $LOG_FILE"

# Validate model repository
if [ ! -d "$MODEL_REPOSITORY/pixel" ]; then
    log "ERROR: Model repository not found at $MODEL_REPOSITORY/pixel"
    exit 1
fi

log "Model repository validated"

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    log "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
        log "  $line"
    done
    GPU_AVAILABLE=true
else
    log "WARNING: No NVIDIA GPU detected, running on CPU"
    GPU_AVAILABLE=false
fi

# Set environment variables
export TRITONSERVER_LOG_LEVEL=INFO
export TRITON_METRICS_ENABLED=1
export TRITON_EAGER_MODE=true
export TRITON_ENABLE_GPU_METRICS=true

# GPU memory configuration
if [ "$GPU_AVAILABLE" = true ]; then
    export CUDA_VISIBLE_DEVICES=0
    export TRITON_GPU_MEMORY_FRACTION=$GPU_MEMORY_FRACTION
    log "GPU memory fraction set to $GPU_MEMORY_FRACTION"
fi

# Performance tuning based on profile
case "$TRITON_PROFILE" in
    production)
        log "Using production profile"
        TRITON_ARGS="--max-batch-size=32"
        TRITON_ARGS="$TRITON_ARGS --model-control-mode=poll"
        TRITON_ARGS="$TRITON_ARGS --model-repository-poll-secs=5"
        ;;
    development)
        log "Using development profile"
        TRITON_ARGS="--max-batch-size=8"
        TRITON_ARGS="$TRITON_ARGS --model-control-mode=none"
        ;;
    performance)
        log "Using performance profiling profile"
        TRITON_ARGS="--max-batch-size=64"
        TRITON_ARGS="$TRITON_ARGS --model-control-mode=poll"
        TRITON_ARGS="$TRITON_ARGS --trace-file=/workspace/logs/trace.log"
        ;;
    *)
        log "ERROR: Unknown profile: $TRITON_PROFILE"
        exit 1
        ;;
esac

# Common arguments
TRITON_ARGS="$TRITON_ARGS --model-repository=$MODEL_REPOSITORY"
TRITON_ARGS="$TRITON_ARGS --http-port=$HTTP_PORT"
TRITON_ARGS="$TRITON_ARGS --grpc-port=$GRPC_PORT"
TRITON_ARGS="$TRITON_ARGS --metrics-port=$METRICS_PORT"
TRITON_ARGS="$TRITON_ARGS --strict-model-config=false"
TRITON_ARGS="$TRITON_ARGS --log-verbose=true"
TRITON_ARGS="$TRITON_ARGS --allow-metrics=true"

# Signal handlers for graceful shutdown
cleanup() {
    log "Shutting down Triton server..."
    kill $TRITON_PID || true
    wait $TRITON_PID 2>/dev/null || true
    log "Triton server stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start Triton server
log "Starting Triton server with arguments: $TRITON_ARGS"
tritonserver $TRITON_ARGS > >(tee -a "$LOG_FILE") 2>&1 &
TRITON_PID=$!

log "Triton process started with PID: $TRITON_PID"

# Wait for server to be ready
log "Waiting for server to become ready..."
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null 2>&1; then
        log "Triton server is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    log "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log "ERROR: Server failed to become ready after $MAX_RETRIES attempts"
    kill $TRITON_PID || true
    exit 1
fi

# Log server info
log "Server Information:"
curl -s "http://localhost:$HTTP_PORT/v2" | python3 -m json.tool >> "$LOG_FILE" 2>&1 || true

# Monitor models
log "Loading models..."
sleep 2
curl -s "http://localhost:$HTTP_PORT/v2/models" | python3 -m json.tool | tee -a "$LOG_FILE"

# Show metrics endpoint
log "Metrics available at: http://localhost:$METRICS_PORT/metrics"

# Keep running
log "Triton server running. Press Ctrl+C to stop."
wait $TRITON_PID
