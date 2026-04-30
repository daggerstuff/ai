#!/bin/bash
# HETZNER AI Training - Job Runner
# Helper script to build images and launch training jobs

set -euo pipefail
HETZNER_AI_CLI="${HETZNER_AI_CLI:-ovhai}"

# Configuration
HETZNER_REGION="${HETZNER_REGION:-hel1}"
DATA_BUCKET="${DATA_BUCKET:-pixel-data}"
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
    if ! command -v ${HETZNER_AI_CLI} &> /dev/null; then
        log_error "${HETZNER_AI_CLI} CLI not found. Install with:"
        echo "  curl -sSL https://docs.hetzner.com/cloud/ai/cli/install.sh | bash"
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
        -f ai/training/ready_packages/platforms/hetzner/Dockerfile.training \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
    
    log_success "Image built: $IMAGE_NAME:$IMAGE_TAG"
}

# Push image to HETZNER registry
push_image() {
    log_info "Pushing image to HETZNER registry..."
    
    # Login to HETZNER registry
    ${HETZNER_AI_CLI} registry login
    
    # Tag for HETZNER registry
    # Registry URL
    # If HETZNER_REGISTRY is not set, try auto-detect from CLI; require explicit configuration fallback
    if [ -z "${HETZNER_REGISTRY:-}" ]; then
         if command -v ${HETZNER_AI_CLI} &> /dev/null && ${HETZNER_AI_CLI} registry url &> /dev/null; then
             HETZNER_REGISTRY=$(${HETZNER_AI_CLI} registry url)
         else
             log_error "Could not auto-detect registry. Set HETZNER_REGISTRY explicitly."
             log_error "Example: export HETZNER_REGISTRY='registry.hel1.hetzner.cloud/<tenant-id>'"
             exit 1
         fi
    fi
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$HETZNER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    
    # Push
    docker push "$HETZNER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    
    log_success "Image pushed to: $HETZNER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
}

# Run training job
run_job() {
    local stage="${1:-all}"
    local flavor="${2:-l40s-1-gpu}"
    
    log_info "Launching training job for stage: $stage"
    log_info "GPU flavor: $flavor"
    
    # Build command based on stage
    local cmd="python ai/training/ready_packages/platforms/hetzner/train_hetzner_canonical.py --stage $stage --data-dir /data --checkpoint-dir /checkpoints"
    
    # Add resume flag for later stages
    case $stage in
        reasoning)
            cmd="$cmd --resume-from /checkpoints/foundation/final"
            ;;
        voice)
            cmd="$cmd --resume-from /checkpoints/reasoning/final"
            ;;
    esac
    
    # Get HETZNER registry URL
    if [ -z "${HETZNER_REGISTRY:-}" ]; then
         if command -v ${HETZNER_AI_CLI} &> /dev/null && ${HETZNER_AI_CLI} registry url &> /dev/null; then
             HETZNER_REGISTRY=$(${HETZNER_AI_CLI} registry url)
         else
             log_error "Could not auto-detect registry. Set HETZNER_REGISTRY explicitly before running."
             log_error "Example: export HETZNER_REGISTRY='registry.hel1.hetzner.cloud/<tenant-id>'"
             exit 1
         fi
    fi
    
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
    ${HETZNER_AI_CLI} job run \
        --name "wayfarer-sft-$stage-$(date +%Y%m%d-%H%M)" \
        --flavor "$flavor" \
        --volume "$DATA_BUCKET@$HETZNER_REGION:/data:RO:cache" \
        --volume "$CHECKPOINT_BUCKET@$HETZNER_REGION:/checkpoints:RW" \
        $env_args \
        "$HETZNER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" \
        -- bash -c "$cmd"
    
    log_success "Job submitted! Monitor with: ${HETZNER_AI_CLI} job list"
}

# List jobs
list_jobs() {
    log_info "Listing HETZNER AI Training jobs..."
    ${HETZNER_AI_CLI} job list
}

# Show job logs
show_logs() {
    local job_id="${1:-}"
    
    if [ -z "$job_id" ]; then
        log_error "Job ID required. Usage: $0 logs <job-id>"
        exit 1
    fi
    
    log_info "Showing logs for job: $job_id"
    ${HETZNER_AI_CLI} job logs -f "$job_id"
}

# Stop job
stop_job() {
    local job_id="${1:-}"
    
    if [ -z "$job_id" ]; then
        log_error "Job ID required. Usage: $0 stop <job-id>"
        exit 1
    fi
    
    log_info "Stopping job: $job_id"
    ${HETZNER_AI_CLI} job stop "$job_id"
    log_success "Job stopped"
}

# Download checkpoints
download_checkpoints() {
    local target="${1:-./checkpoints}"
    
    log_info "Downloading checkpoints to: $target"
    mkdir -p "$target"
    
    ${HETZNER_AI_CLI} data pull "$CHECKPOINT_BUCKET" "$target/"
    log_success "Checkpoints downloaded"
}

# Show usage
usage() {
    echo "HETZNER AI Training - Job Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build                 Build Docker training image"
    echo "  push                  Push image to HETZNER registry"
    echo "  run [stage] [flavor]  Launch training job"
    echo "                        stage: all, foundation, reasoning, voice (default: all)"
    echo "                        flavor: l40s-1-gpu, l4-1-gpu (default: l40s-1-gpu)"
    echo "  list                  List running/completed jobs"
    echo "  logs <job-id>         Show job logs (follow mode)"
    echo "  stop <job-id>         Stop a running job"
    echo "  download [dir]        Download checkpoints (default: ./checkpoints)"
    echo ""
    echo "Workflow:"
    echo "  1. $0 build           # Build Docker image"
    echo "  2. $0 push            # Push to HETZNER registry"
    echo "  3. ./sync-datasets.sh upload  # Upload training data"
    echo "  4. $0 run             # Launch training (all stages)"
    echo "  5. $0 list            # Monitor progress"
    echo "  6. $0 download        # Get checkpoints after completion"
    echo ""
    echo "Environment Variables:"
    echo "  WANDB_API_KEY         Weights & Biases API key"
    echo "  HF_TOKEN              HuggingFace token"
    echo "  HETZNER_REGION            HETZNER region (default: hel1)"
    echo "  DATA_BUCKET           Data bucket (default: pixel-data)"
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
            run_job "${2:-all}" "${3:-l40s-1-gpu}"
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
