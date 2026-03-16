#!/bin/bash
# Triton health check script

set -euo pipefail

HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"

# Check HTTP port
if ! timeout "$TIMEOUT" curl -s "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null 2>&1; then
    echo "ERROR: HTTP health check failed"
    exit 1
fi

# Check gRPC port
if ! timeout "$TIMEOUT" curl -s "http://localhost:$GRPC_PORT/v2/health/ready" > /dev/null 2>&1; then
    echo "WARN: gRPC health check failed (may not be critical)"
fi

# Check metrics endpoint
if ! timeout "$TIMEOUT" curl -s "http://localhost:$METRICS_PORT/metrics" > /dev/null 2>&1; then
    echo "WARN: Metrics endpoint not responding"
fi

# Check model availability
MODELS=$(curl -s "http://localhost:$HTTP_PORT/v2/models" 2>/dev/null || echo "")
if echo "$MODELS" | grep -q "pixel"; then
    echo "OK: Pixel model is available"
else
    echo "ERROR: Pixel model not found in model repository"
    exit 1
fi

echo "OK: Triton health check passed"
exit 0
