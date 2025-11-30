#!/bin/bash
#
# Local-to-VPS Transfer Script
# Transfers missing local datasets to VPS (146.71.78.184)
#
# Usage: Run this script from the project root
#   ./ai/dataset_pipeline/scripts/transfer_to_vps.sh

set -euo pipefail

# Configuration
VPS_USER="vivi"
VPS_HOST="146.71.78.184"
SSH_KEY="$HOME/.ssh/planet"
VPS_DEST_DIR="~/pixelated-datasets/raw/local"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/transfer.log"
ERROR_LOG="$PROJECT_ROOT/transfer_errors.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$ERROR_LOG"
}

log_transfer() {
    echo -e "${BLUE}[TRANSFER]${NC} $1" | tee -a "$LOG_FILE"
}

# Check SSH connection
check_ssh() {
    log_info "Checking SSH connection to VPS..."
    
    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found: $SSH_KEY"
        exit 1
    fi
    
    chmod 600 "$SSH_KEY"
    
    if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'SSH connection successful'" &>/dev/null; then
        log_error "Cannot connect to VPS. Check SSH key and host."
        exit 1
    fi
    
    log_info "SSH connection verified"
}

# Transfer function using rsync
transfer_directory() {
    local source_path="$1"
    local dest_name="$2"
    local description="$3"
    
    if [ ! -d "$source_path" ]; then
        log_warn "Source directory not found: $source_path (skipping)"
        return
    fi
    
    log_transfer "Transferring: $description"
    log_transfer "Source: $source_path"
    log_transfer "Destination: $VPS_DEST_DIR/$dest_name"
    
    # Use rsync with progress, compression, and resume capability
    rsync -avzP \
        --progress \
        --partial \
        --partial-dir=.rsync-partial \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$source_path/" \
        "$VPS_USER@$VPS_HOST:$VPS_DEST_DIR/$dest_name/" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "✓ Successfully transferred: $description"
    else
        log_error "✗ Failed to transfer: $description"
        return 1
    fi
}

# Transfer single file
transfer_file() {
    local source_path="$1"
    local dest_name="$2"
    local description="$3"
    
    if [ ! -f "$source_path" ]; then
        log_warn "Source file not found: $source_path (skipping)"
        return
    fi
    
    log_transfer "Transferring: $description"
    log_transfer "Source: $source_path"
    log_transfer "Destination: $VPS_DEST_DIR/$dest_name"
    
    # Create destination directory on VPS
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" \
        "mkdir -p $(dirname $VPS_DEST_DIR/$dest_name)"
    
    # Transfer file
    rsync -avzP \
        --progress \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$source_path" \
        "$VPS_USER@$VPS_HOST:$VPS_DEST_DIR/$dest_name" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "✓ Successfully transferred: $description"
    else
        log_error "✗ Failed to transfer: $description"
        return 1
    fi
}

# Main transfer function
transfer_datasets() {
    log_info "Starting local-to-VPS dataset transfer..."
    log_warn "Note: This will be slow. Monitor progress in $LOG_FILE"
    
    cd "$PROJECT_ROOT"
    
    # Edge case generator outputs
    transfer_directory \
        "ai/lightning/ghost/edge_case_pipeline_standalone" \
        "edge_case_generator" \
        "Edge Case Generator Outputs"
    
    # Tim Fletcher transcripts
    transfer_directory \
        "ai/training_data_consolidated/transcripts" \
        "tim_fletcher_transcripts" \
        "Tim Fletcher YouTube Transcripts (42 files)"
    
    # Tim Fletcher voice profile
    transfer_directory \
        "ai/data/tim_fletcher_voice" \
        "tim_fletcher_voice" \
        "Tim Fletcher Voice Profile"
    
    # Priority datasets
    transfer_directory \
        "ai/lightning/ghost/datasets/priority_1" \
        "priority_datasets/priority_1" \
        "Priority 1 Dataset"
    
    transfer_directory \
        "ai/lightning/ghost/datasets/priority_2" \
        "priority_datasets/priority_2" \
        "Priority 2 Dataset"
    
    transfer_directory \
        "ai/lightning/ghost/datasets/priority_3" \
        "priority_datasets/priority_3" \
        "Priority 3 Dataset"
    
    # Processed phases
    transfer_directory \
        "ai/lightning/pixelated-training/processed" \
        "processed_phases" \
        "Processed Training Phases (Phase 1-3)"
    
    # Consolidated data (LARGE - may take a long time)
    log_warn "WARNING: Transferring consolidated data (7.2GB). This will take a LONG time!"
    read -p "Continue with consolidated data transfer? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        transfer_directory \
            "ai/training_data_consolidated" \
            "consolidated_data" \
            "Consolidated Training Data (7.2GB)"
    else
        log_warn "Skipping consolidated data transfer"
    fi
    
    # CoT datasets
    transfer_directory \
        "ai/dataset_pipeline/cot_datasets" \
        "cot_datasets" \
        "CoT Reasoning Datasets"
    
    # Psychology books (optional)
    log_warn "Transferring psychology books (126MB)..."
    transfer_directory \
        "ai/lightning/ghost/books" \
        "books" \
        "Psychology Books (PDFs)"
    
    # DSM-V PDF
    if [ -f "ai/DSMV.pdf" ]; then
        transfer_file \
            "ai/DSMV.pdf" \
            "dsmv/DSMV.pdf" \
            "DSM-V PDF"
    fi
    
    log_info "Local-to-VPS transfer complete!"
}

# Generate transfer summary
generate_summary() {
    log_info "Generating transfer summary..."
    
    python3 << EOF
import json
from datetime import datetime

summary = {
    "transfer_completed": datetime.now().isoformat(),
    "vps_host": "$VPS_HOST",
    "destination": "$VPS_DEST_DIR",
    "log_file": "$LOG_FILE",
    "error_log": "$ERROR_LOG"
}

# Read log file to count transfers
try:
    with open("$LOG_FILE", "r") as f:
        log_content = f.read()
        successful = log_content.count("✓ Successfully transferred")
        failed = log_content.count("✗ Failed to transfer")
        summary["transfers_successful"] = successful
        summary["transfers_failed"] = failed
except FileNotFoundError:
    summary["transfers_successful"] = 0
    summary["transfers_failed"] = 0

output_file = "$PROJECT_ROOT/transfer_summary.json"
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Transfer summary: {output_file}")
EOF

    log_info "Transfer summary generated"
}

# Main execution
main() {
    log_info "Starting local-to-VPS dataset transfer..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "VPS: $VPS_USER@$VPS_HOST"
    
    # Check SSH connection
    check_ssh
    
    # Create destination directory on VPS
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" \
        "mkdir -p $VPS_DEST_DIR"
    
    # Transfer datasets
    transfer_datasets
    
    # Generate summary
    generate_summary
    
    log_info "All transfers complete!"
    log_info "Check logs: $LOG_FILE"
    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        log_warn "Errors occurred. Check: $ERROR_LOG"
    fi
}

# Run main function
main

