#!/bin/bash
# OVH AI Deploy Inference Deployment Script
# Deploys Pixelated Empathy inference service to OVH AI Deploy

set -euo pipefail

# Configuration
OVH_REGION="${OVH_REGION:-US-EAST-VA}"
CHECKPOINT_BUCKET="${CHECKPOINT_BUCKET:-pixelated-checkpoints}"
GPU_MODEL="${GPU_MODEL:-L4}"  # L4 is good for inference
REPLICAS="${REPLICAS:-2}"

# Default image
DEFAULT_IMAGE="pixelated-inference:latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prereqs() {
    if ! command -v ovhai &> /dev/null; then
        log_error "ovhai CLI not found"
        exit 1
    fi
    
    if ! ovhai me &> /dev/null; then
        log_error "Not authenticated. Run: ovhai login"
        exit 1
    fi
}

# Build and push inference image
build_image() {
    local registry="${1:-}"
    
    if [ -z "$registry" ]; then
        registry=$(ovhai registry list --json 2>/dev/null | jq -r '.[0].url' || echo "")
        if [ -z "$registry" ]; then
            log_error "No registry found"
            exit 1
        fi
    fi
    
    local image_name="$registry/pixelated-inference:latest"
    
    log_info "Building inference image..."
    cd /home/vivi/pixelated/ai
    docker build -f docker/ovh/Dockerfile.inference -t pixelated-inference:latest .
    
    log_info "Logging into OVH registry..."
    ovhai registry login
    
    log_info "Pushing image..."
    docker tag pixelated-inference:latest "$image_name"
    docker push "$image_name"
    
    log_success "Image pushed: $image_name"
    echo "$image_name"
}

# Deploy inference app
deploy_app() {
    local image="${1:-$DEFAULT_IMAGE}"
    local app_name="${2:-pixelated-inference}"
    
    log_info "Deploying inference app: $app_name"
    log_info "  Image: $image"
    log_info "  GPU: $GPU_MODEL"
    log_info "  Replicas: $REPLICAS"
    
    # Check if app exists
    if ovhai app get "$app_name" &> /dev/null; then
        log_info "App exists, updating..."
        ovhai app stop "$app_name" || true
        sleep 5
    fi
    
    # Deploy
    ovhai app create \
        --name "$app_name" \
        --gpu 1 \
        --gpu-model "$GPU_MODEL" \
        --cpu 4 \
        --memory 16Gi \
        --replicas "$REPLICAS" \
        --volume "$CHECKPOINT_BUCKET@$OVH_REGION:/models:ro" \
        --env MODEL_DIR=/models \
        --env MAX_LENGTH=2048 \
        --env DEFAULT_MAX_TOKENS=512 \
        --port 8080 \
        "$image"
    
    log_success "App deployed: $app_name"
    
    # Wait for app to be ready
    log_info "Waiting for app to be ready..."
    sleep 30
    
    # Get app URL
    local app_url=$(ovhai app get "$app_name" --json | jq -r '.url' || echo "")
    if [ -n "$app_url" ]; then
        log_success "App URL: $app_url"
        echo ""
        log_info "Test with:"
        echo "  curl $app_url/health"
    fi
}

# Scale app
scale_app() {
    local app_name="${1:-pixelated-inference}"
    local replicas="${2:-2}"
    
    log_info "Scaling $app_name to $replicas replicas..."
    ovhai app scale "$app_name" --replicas "$replicas"
    log_success "Scaled to $replicas replicas"
}

# Get app status
status_app() {
    local app_name="${1:-pixelated-inference}"
    
    log_info "Getting status of $app_name..."
    ovhai app get "$app_name"
}

# Stop app
stop_app() {
    local app_name="${1:-pixelated-inference}"
    
    log_info "Stopping $app_name..."
    ovhai app stop "$app_name"
    log_success "App stopped"
}

# Delete app
delete_app() {
    local app_name="${1:-pixelated-inference}"
    
    log_warn "Deleting $app_name..."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ovhai app delete "$app_name"
        log_success "App deleted"
    else
        log_info "Cancelled"
    fi
}

# Show usage
usage() {
    echo "OVH AI Deploy Inference Deployment"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build [registry]    Build and push inference image"
    echo "  deploy [image] [name]  Deploy inference app"
    echo "  scale [name] [replicas]  Scale app replicas"
    echo "  status [name]       Get app status"
    echo "  stop [name]         Stop app"
    echo "  delete [name]       Delete app"
    echo ""
    echo "Environment Variables:"
    echo "  GPU_MODEL           GPU model (default: L4)"
    echo "  REPLICAS            Number of replicas (default: 2)"
    echo "  CHECKPOINT_BUCKET   Checkpoint bucket name"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build and push image"
    echo "  $0 deploy                   # Deploy with defaults"
    echo "  $0 scale pixelated-inference 3  # Scale to 3 replicas"
}

main() {
    check_prereqs
    
    case "${1:-}" in
        build)
            build_image "${2:-}"
            ;;
        deploy)
            deploy_app "${2:-}" "${3:-pixelated-inference}"
            ;;
        scale)
            scale_app "${2:-pixelated-inference}" "${3:-2}"
            ;;
        status)
            status_app "${2:-pixelated-inference}"
            ;;
        stop)
            stop_app "${2:-pixelated-inference}"
            ;;
        delete)
            delete_app "${2:-pixelated-inference}"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"

