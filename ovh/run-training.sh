#!/bin/bash
# OVH AI Training - Job Runner
# Helper script to build images and launch training jobs

set -euo pipefail

# Configuration
OVH_REGION="${OVH_REGION:-US-EAST-VA}"
DATA_BUCKET="${DATA_BUCKET:-pixelated-training-data}"
CHECKPOINT_BUCKET="${CHECKPOINT_BUCKET:-pixelated-checkpoints}"
PROJECT_ROOT="/home/vivi/pixelated"
IMAGE_NAME="${IMAGE_NAME:-pixelated-training}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

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

# Check prerequisites
check_prereqs() {
    if ! command -v ovhai &> /dev/null; then
        log_error "ovhai CLI not found. Install with:"
        echo "  curl -sSL https://cli.us-east-va.ai.cloud.ovh.us/install.sh | bash"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
}

# Build Docker image
build_image() {
    log_info "Building training Docker image: $IMAGE_NAME:$IMAGE_TAG"
    
    cd "$PROJECT_ROOT"
    
    docker build \
        -f ai/ovh/Dockerfile.training \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
    
    log_success "Image built: $IMAGE_NAME:$IMAGE_TAG"
}

# Push image to OVH registry
push_image() {
    log_info "Pushing image to OVH registry..."
    
    # Login to OVH registry
    ovhai registry login
    
    # Tag for OVH registry
    OVH_REGISTRY=$(ovhai registry url)
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$OVH_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    
    # Push
    docker push "$OVH_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    
    log_success "Image pushed to: $OVH_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
}

# Run training job
run_job() {
    local stage="${1:-all}"
    local flavor="${2:-gpu_nvidia_l40s_1}"
    
    log_info "Launching training job for stage: $stage"
    log_info "GPU flavor: $flavor"
    
    # Build command based on stage
    local cmd="python ai/ovh/train_ovh.py --stage $stage --data-dir /data --checkpoint-dir /checkpoints"
    
    # Add resume flag for later stages
    case $stage in
        reasoning)
            cmd="$cmd --resume-from /checkpoints/foundation/final"
            ;;
        voice)
            cmd="$cmd --resume-from /checkpoints/reasoning/final"
            ;;
    esac
    
    # Get OVH registry URL
    OVH_REGISTRY=$(ovhai registry url)
    
    # Environment variables
    local env_args=""
    if [ -n "${WANDB_API_KEY:-}" ]; then
        env_args="$env_args --env WANDB_API_KEY=$WANDB_API_KEY"
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        env_args="$env_args --env HF_TOKEN=$HF_TOKEN"
    fi
    env_args="$env_args --env PYTHONUNBUFFERED=1"
    env_args="$env_args --env TRANSFORMERS_CACHE=/data/.cache/transformers"
    env_args="$env_args --env HF_HOME=/data/.cache/huggingface"
    
    # Launch job
    ovhai job run \
        --name "wayfarer-sft-$stage-$(date +%Y%m%d-%H%M)" \
        --flavor "$flavor" \
        --volume "$DATA_BUCKET@$OVH_REGION:/data:RO:cache" \
        --volume "$CHECKPOINT_BUCKET@$OVH_REGION:/checkpoints:RW" \
        $env_args \
        "$OVH_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" \
        -- bash -c "$cmd"
    
    log_success "Job submitted! Monitor with: ovhai job list"
}

# List jobs
list_jobs() {
    log_info "Listing OVH AI Training jobs..."
    ovhai job list
}

# Show job logs
show_logs() {
    local job_id="${1:-}"
    
    if [ -z "$job_id" ]; then
        log_error "Job ID required. Usage: $0 logs <job-id>"
        exit 1
    fi
    
    log_info "Showing logs for job: $job_id"
    ovhai job logs -f "$job_id"
}

# Stop job
stop_job() {
    local job_id="${1:-}"
    
    if [ -z "$job_id" ]; then
        log_error "Job ID required. Usage: $0 stop <job-id>"
        exit 1
    fi
    
    log_info "Stopping job: $job_id"
    ovhai job stop "$job_id"
    log_success "Job stopped"
}

# Download checkpoints
download_checkpoints() {
    local target="${1:-./checkpoints}"
    
    log_info "Downloading checkpoints to: $target"
    mkdir -p "$target"
    
    ovhai data pull "$CHECKPOINT_BUCKET" "$target/"
    log_success "Checkpoints downloaded"
}

# Show usage
usage() {
    echo "OVH AI Training - Job Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build                 Build Docker training image"
    echo "  push                  Push image to OVH registry"
    echo "  run [stage] [flavor]  Launch training job"
    echo "                        stage: all, foundation, reasoning, voice (default: all)"
    echo "                        flavor: gpu_nvidia_l40s_1, gpu_nvidia_h100_1 (default: l40s)"
    echo "  list                  List running/completed jobs"
    echo "  logs <job-id>         Show job logs (follow mode)"
    echo "  stop <job-id>         Stop a running job"
    echo "  download [dir]        Download checkpoints (default: ./checkpoints)"
    echo ""
    echo "Workflow:"
    echo "  1. $0 build           # Build Docker image"
    echo "  2. $0 push            # Push to OVH registry"
    echo "  3. ./sync-datasets.sh upload  # Upload training data"
    echo "  4. $0 run             # Launch training (all stages)"
    echo "  5. $0 list            # Monitor progress"
    echo "  6. $0 download        # Get checkpoints after completion"
    echo ""
    echo "Environment Variables:"
    echo "  WANDB_API_KEY         Weights & Biases API key"
    echo "  HF_TOKEN              HuggingFace token"
    echo "  OVH_REGION            OVH region (default: US-EAST-VA)"
    echo "  DATA_BUCKET           Data bucket (default: pixelated-training-data)"
    echo "  CHECKPOINT_BUCKET     Checkpoint bucket (default: pixelated-checkpoints)"
    echo "  IMAGE_NAME            Docker image name (default: pixelated-training)"
    echo "  IMAGE_TAG             Docker image tag (default: latest)"
}

# Main
main() {
    check_prereqs
    
    case "${1:-}" in
        build)
            build_image
            ;;
        push)
            push_image
            ;;
        run)
            run_job "${2:-all}" "${3:-gpu_nvidia_l40s_1}"
            ;;
        list)
            list_jobs
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        stop)
            stop_job "${2:-}"
            ;;
        download)
            download_checkpoints "${2:-./checkpoints}"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
