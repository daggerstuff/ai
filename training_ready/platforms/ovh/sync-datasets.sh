#!/bin/bash
# OVH Object Storage Dataset Sync Script
# Syncs training datasets to OVH Object Storage for AI Training
#
# Dataset Sources:
# - Local acquired: ai/data/acquired_datasets/ (CoT reasoning, mental health counseling)
# - Tim Fletcher voice: ai/data/tim_fletcher_voice/
# - Lightning H100 exports: ai/data/lightning_h100/
# - Google Drive (if mounted): /mnt/gdrive/datasets/

set -euo pipefail

# Configuration
OVH_REGION="${OVH_REGION:-US-EAST-VA}"
DATA_BUCKET="${DATA_BUCKET:-pixelated-training-data}"
CHECKPOINT_BUCKET="${CHECKPOINT_BUCKET:-pixelated-checkpoints}"
AI_DIR="/home/vivi/pixelated/ai"
GDRIVE_MOUNT="${GDRIVE_MOUNT:-/mnt/gdrive/datasets}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_ovhai() {
    if ! command -v ovhai &> /dev/null; then
        log_error "ovhai CLI not found. Please install it first:"
        echo "  curl -sSL https://cli.us-east-va.ai.cloud.ovh.us/install.sh | bash"
        exit 1
    fi
}

check_auth() {
    log_info "Checking OVH authentication..."
    if ! ovhai me &> /dev/null; then
        log_error "Not authenticated to OVH. Please run: ovhai login"
        exit 1
    fi
    log_success "Authenticated to OVH AI Platform"
}

ensure_bucket() {
    local bucket_name=$1
    log_info "Checking bucket: $bucket_name"
    if ovhai bucket list "$bucket_name" &> /dev/null; then
        log_success "Bucket '$bucket_name' exists"
    else
        log_warn "Bucket '$bucket_name' may need to be created via OVH Control Panel"
        log_warn "Go to: Public Cloud > Object Storage > Create Container"
    fi
}

# Upload datasets following the current pipeline structure
upload_datasets() {
    log_info "Uploading datasets to OVH Object Storage..."
    log_info "Target bucket: $DATA_BUCKET@$OVH_REGION"
    
    # ===========================================
    # Core Acquired Datasets (Stage 1 & 2 data)
    # ===========================================
    if [ -d "$AI_DIR/data/acquired_datasets" ]; then
        log_info "📦 Uploading acquired datasets (CoT reasoning + mental health counseling)..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/data/acquired_datasets/" --prefix "acquired/" || log_warn "Failed to upload acquired datasets"
    fi
    
    # ===========================================
    # Lightning H100 Training Data
    # ===========================================
    if [ -d "$AI_DIR/data/lightning_h100" ]; then
        log_info "📦 Uploading Lightning H100 training data..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/data/lightning_h100/" --prefix "lightning/" || log_warn "Failed to upload lightning data"
    fi
    
    # ===========================================
    # Tim Fletcher Voice Data (Stage 3)
    # ===========================================
    if [ -d "$AI_DIR/data/tim_fletcher_voice" ]; then
        log_info "📦 Uploading Tim Fletcher voice data..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/data/tim_fletcher_voice/" --prefix "voice/" || log_warn "Failed to upload voice data"
    fi
    
    # ===========================================
    # Pixel Voice Pipeline Data
    # ===========================================
    if [ -d "$AI_DIR/pipelines/pixel_voice" ]; then
        # Upload only the generated data, not the pipeline code
        if [ -d "$AI_DIR/pipelines/pixel_voice/data" ]; then
            log_info "📦 Uploading Pixel Voice generated data..."
            ovhai data push "$DATA_BUCKET" "$AI_DIR/pipelines/pixel_voice/data/" --prefix "pixel_voice/" || log_warn "Failed to upload pixel voice data"
        fi
    fi
    
    # ===========================================
    # Edge Case Pipeline Data
    # ===========================================
    if [ -d "$AI_DIR/pipelines/edge_case_pipeline_standalone" ]; then
        # Look for generated training data
        for data_file in "$AI_DIR/pipelines/edge_case_pipeline_standalone"/*.json*; do
            if [ -f "$data_file" ]; then
                log_info "📦 Uploading edge case data: $(basename "$data_file")..."
                ovhai data push "$DATA_BUCKET" "$data_file" --prefix "edge_cases/" || log_warn "Failed to upload $data_file"
            fi
        done
    fi
    
    # ===========================================
    # Dual Persona Training Data
    # ===========================================
    if [ -d "$AI_DIR/pipelines/dual_persona_training" ]; then
        log_info "📦 Uploading dual persona training data..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/pipelines/dual_persona_training/" --prefix "dual_persona/" || log_warn "Failed to upload dual persona data"
    fi
    
    # ===========================================
    # Dataset Registry (important for training)
    # ===========================================
    if [ -f "$AI_DIR/data/dataset_registry.json" ]; then
        log_info "📦 Uploading dataset registry..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/data/dataset_registry.json" --prefix "config/" || log_warn "Failed to upload registry"
    fi
    
    # ===========================================
    # Training Configuration
    # ===========================================
    if [ -f "$AI_DIR/lightning/moe_training_config.json" ]; then
        log_info "📦 Uploading MoE training config..."
        ovhai data push "$DATA_BUCKET" "$AI_DIR/lightning/moe_training_config.json" --prefix "config/" || log_warn "Failed to upload MoE config"
    fi
    
    # ===========================================
    # Google Drive datasets (if mounted)
    # ===========================================
    if [ -d "$GDRIVE_MOUNT" ] && [ "$(ls -A "$GDRIVE_MOUNT" 2>/dev/null)" ]; then
        log_info "📦 Google Drive mount detected at $GDRIVE_MOUNT"
        read -p "Upload Google Drive datasets? This may take a while. (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Uploading Google Drive datasets..."
            
            # Try canonical structure first, fallback to legacy
            if [ -d "$GDRIVE_MOUNT/cot_reasoning" ]; then
                log_info "  Uploading CoT reasoning (canonical structure)..."
                ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/cot_reasoning/" --prefix "gdrive/cot_reasoning/" || log_warn "Failed to upload CoT reasoning"
            else
                # Fallback to legacy structure
                for dataset in "CoT_Reasoning_Clinical_Diagnosis_Mental_Health" "CoT_Heartbreak_and_Breakups_downloaded.json" "CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json"; do
                    if [ -d "$GDRIVE_MOUNT/$dataset" ] || [ -f "$GDRIVE_MOUNT/$dataset" ]; then
                        log_info "  Uploading $dataset (legacy)..."
                        ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/$dataset" --prefix "gdrive/cot_reasoning/" || log_warn "Failed to upload $dataset"
                    fi
                done
            fi
            
            if [ -d "$GDRIVE_MOUNT/professional_therapeutic" ]; then
                log_info "  Uploading professional therapeutic (canonical structure)..."
                ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/professional_therapeutic/" --prefix "gdrive/professional_therapeutic/" || log_warn "Failed to upload professional therapeutic"
            else
                # Fallback to legacy structure
                for dataset in "Psych8k" "mental_health_counseling_conversations" "therapist-sft-format" "SoulChat2.0"; do
                    if [ -d "$GDRIVE_MOUNT/$dataset" ]; then
                        log_info "  Uploading $dataset (legacy)..."
                        ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/$dataset/" --prefix "gdrive/professional_therapeutic/" || log_warn "Failed to upload $dataset"
                    fi
                done
            fi
            
            # Upload priority datasets (try both canonical and legacy names)
            if [ -d "$GDRIVE_MOUNT/priority" ]; then
                log_info "  Uploading priority datasets (canonical structure)..."
                ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/priority/" --prefix "gdrive/priority/" || log_warn "Failed to upload priority datasets"
            elif [ -d "$GDRIVE_MOUNT/datasets-wendy" ]; then
                log_info "  Uploading priority datasets (legacy: datasets-wendy)..."
                ovhai data push "$DATA_BUCKET" "$GDRIVE_MOUNT/datasets-wendy/" --prefix "gdrive/priority/" || log_warn "Failed to upload priority datasets"
            fi
        else
            log_info "Skipping Google Drive datasets"
        fi
    else
        log_warn "Google Drive not mounted at $GDRIVE_MOUNT - skipping gdrive datasets"
    fi
    
    log_success "Dataset upload completed!"
    echo ""
    log_info "Dataset structure on OVH:"
    echo "  s3://$DATA_BUCKET/"
    echo "  ├── acquired/              # CoT reasoning + mental health counseling"
    echo "  ├── lightning/             # Lightning H100 expert data"
    echo "  ├── voice/                 # Tim Fletcher voice data"
    echo "  ├── pixel_voice/           # Pixel Voice pipeline output"
    echo "  ├── edge_cases/            # Edge case scenarios"
    echo "  ├── dual_persona/          # Dual persona training"
    echo "  ├── gdrive/                # Google Drive datasets (if uploaded)"
    echo "  └── config/                # Training configurations"
}

# Download datasets from OVH
download_datasets() {
    local target_dir="${1:-/data}"
    log_info "Downloading datasets from OVH Object Storage to $target_dir..."
    
    mkdir -p "$target_dir"
    
    ovhai data pull "$DATA_BUCKET" "$target_dir/" || {
        log_error "Failed to download datasets"
        exit 1
    }
    
    log_success "Dataset download completed!"
}

# List bucket contents
list_contents() {
    log_info "Listing contents of $DATA_BUCKET..."
    ovhai bucket list "$DATA_BUCKET"
    
    echo ""
    log_info "Listing contents of $CHECKPOINT_BUCKET..."
    ovhai bucket list "$CHECKPOINT_BUCKET" 2>/dev/null || log_warn "Checkpoint bucket may not exist yet"
}

# Download checkpoints
download_checkpoints() {
    local target_dir="${1:-/checkpoints}"
    log_info "Downloading checkpoints from OVH..."
    
    mkdir -p "$target_dir"
    
    ovhai data pull "$CHECKPOINT_BUCKET" "$target_dir/" || {
        log_error "Failed to download checkpoints"
        exit 1
    }
    
    log_success "Checkpoint download completed!"
}

# Upload checkpoints
upload_checkpoints() {
    local source_dir="${1:-/checkpoints}"
    log_info "Uploading checkpoints to OVH..."
    
    if [ ! -d "$source_dir" ]; then
        log_error "Checkpoint directory not found: $source_dir"
        exit 1
    fi
    
    ovhai data push "$CHECKPOINT_BUCKET" "$source_dir/" || {
        log_error "Failed to upload checkpoints"
        exit 1
    }
    
    log_success "Checkpoint upload completed!"
}

# Show dataset inventory
show_inventory() {
    log_info "📊 Local Dataset Inventory"
    echo ""
    
    echo "=== Acquired Datasets ==="
    if [ -f "$AI_DIR/data/acquired_datasets/cot_reasoning.json" ]; then
        size=$(du -h "$AI_DIR/data/acquired_datasets/cot_reasoning.json" | cut -f1)
        echo "  ✅ CoT Reasoning: $size"
    fi
    if [ -f "$AI_DIR/data/acquired_datasets/mental_health_counseling.json" ]; then
        size=$(du -h "$AI_DIR/data/acquired_datasets/mental_health_counseling.json" | cut -f1)
        echo "  ✅ Mental Health Counseling: $size"
    fi
    
    echo ""
    echo "=== Lightning H100 Data ==="
    if [ -d "$AI_DIR/data/lightning_h100" ]; then
        count=$(ls -1 "$AI_DIR/data/lightning_h100"/*.json 2>/dev/null | wc -l)
        echo "  ✅ Expert files: $count"
    fi
    
    echo ""
    echo "=== Tim Fletcher Voice ==="
    if [ -f "$AI_DIR/data/tim_fletcher_voice/tim_fletcher_voice_profile.json" ]; then
        echo "  ✅ Voice profile: exists"
    fi
    
    echo ""
    echo "=== Google Drive Mount ==="
    if [ -d "$GDRIVE_MOUNT" ] && [ "$(ls -A "$GDRIVE_MOUNT" 2>/dev/null)" ]; then
        echo "  ✅ Mounted at $GDRIVE_MOUNT"
        echo "  📁 Available datasets:"
        ls -1 "$GDRIVE_MOUNT" 2>/dev/null | head -10 | sed 's/^/     /'
    else
        echo "  ❌ Not mounted"
    fi
}

# Show usage
usage() {
    echo "OVH Object Storage Dataset Sync Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  upload              Upload datasets to OVH Object Storage"
    echo "  download [dir]      Download datasets from OVH (default: /data)"
    echo "  list                List bucket contents"
    echo "  inventory           Show local dataset inventory"
    echo "  upload-checkpoints [dir]  Upload checkpoints (default: /checkpoints)"
    echo "  download-checkpoints [dir] Download checkpoints (default: /checkpoints)"
    echo "  check               Check authentication and bucket status"
    echo ""
    echo "Environment Variables:"
    echo "  OVH_REGION          OVH region (default: US-EAST-VA)"
    echo "  DATA_BUCKET         Data bucket name (default: pixelated-training-data)"
    echo "  CHECKPOINT_BUCKET   Checkpoint bucket name (default: pixelated-checkpoints)"
    echo "  GDRIVE_MOUNT        Google Drive mount point (default: /mnt/gdrive/datasets)"
    echo ""
    echo "Examples:"
    echo "  $0 inventory                 # Show local dataset inventory"
    echo "  $0 upload                    # Upload all local datasets"
    echo "  $0 download /path/to/data    # Download to specific directory"
    echo "  $0 list                      # List bucket contents"
}

# Main
main() {
    check_ovhai
    
    case "${1:-}" in
        upload)
            check_auth
            ensure_bucket "$DATA_BUCKET"
            upload_datasets
            ;;
        download)
            check_auth
            download_datasets "${2:-/data}"
            ;;
        list)
            check_auth
            list_contents
            ;;
        inventory)
            show_inventory
            ;;
        upload-checkpoints)
            check_auth
            ensure_bucket "$CHECKPOINT_BUCKET"
            upload_checkpoints "${2:-/checkpoints}"
            ;;
        download-checkpoints)
            check_auth
            download_checkpoints "${2:-/checkpoints}"
            ;;
        check)
            check_auth
            ensure_bucket "$DATA_BUCKET"
            ensure_bucket "$CHECKPOINT_BUCKET"
            log_success "All checks passed!"
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
