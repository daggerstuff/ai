#!/bin/bash
# Google Drive Dataset Download Script
# Run this ON the remote server to download datasets with fast bandwidth
#
# Prerequisites on remote server:
#   pip install gdown rclone
#
# Usage:
#   scp ai/ovh/gdrive-download.sh vivi@146.71.78.184:~/
#   ssh -i ~/.ssh/planet vivi@146.71.78.184
#   chmod +x gdrive-download.sh && ./gdrive-download.sh setup
#   ./gdrive-download.sh download-all

set -euo pipefail

# Configuration
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"  # rclone remote name

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

# ===========================================
# Google Drive Dataset URLs/IDs
# From dataset_registry.json
# ===========================================

# CoT Reasoning Datasets
declare -A COT_DATASETS=(
    ["CoT_Clinical_Diagnosis_Mental_Health"]="FOLDER_ID_HERE"
    ["CoT_Heartbreak_and_Breakups"]="FOLDER_ID_HERE"
    ["CoT_Neurodivergent_vs_Neurotypical"]="FOLDER_ID_HERE"
    ["CoT_Mens_Mental_Health"]="FOLDER_ID_HERE"
    ["CoT_Cultural_Nuances"]="FOLDER_ID_HERE"
    ["CoT_Philosophical_Understanding"]="FOLDER_ID_HERE"
    ["CoT_Temporal_Reasoning"]="FOLDER_ID_HERE"
)

# Professional Therapeutic Datasets
declare -A PROFESSIONAL_DATASETS=(
    ["mental_health_counseling_conversations"]="FOLDER_ID_HERE"
    ["SoulChat2.0"]="FOLDER_ID_HERE"
    ["counsel_chat"]="FOLDER_ID_HERE"
    ["LLAMA3_Mental_Counseling"]="FOLDER_ID_HERE"
    ["therapist_sft"]="FOLDER_ID_HERE"
    ["neuro_qa_sft"]="FOLDER_ID_HERE"
    ["Psych8k"]="FOLDER_ID_HERE"
)

# Priority Datasets (datasets-wendy)
declare -A PRIORITY_DATASETS=(
    ["priority_1_FINAL"]="FOLDER_ID_HERE"
    ["priority_2_FINAL"]="FOLDER_ID_HERE"
    ["priority_3_FINAL"]="FOLDER_ID_HERE"
)

# ===========================================
# Setup Functions
# ===========================================

install_deps() {
    log_header "Installing Dependencies"
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found. Please install Python first."
        exit 1
    fi
    
    # Install gdown for direct file downloads
    log_info "Installing gdown..."
    pip3 install --user gdown
    
    # Install rclone for folder sync
    log_info "Installing rclone..."
    if ! command -v rclone &> /dev/null; then
        curl https://rclone.org/install.sh | sudo bash
    fi
    
    log_success "Dependencies installed!"
}

setup_rclone() {
    log_header "Setting up rclone for Google Drive"
    
    if ! command -v rclone &> /dev/null; then
        log_error "rclone not installed. Run: $0 install-deps"
        exit 1
    fi
    
    # Check if remote already exists
    if rclone listremotes | grep -q "^${GDRIVE_REMOTE}:$"; then
        log_success "rclone remote '$GDRIVE_REMOTE' already configured"
        return
    fi
    
    log_info "Launching rclone config..."
    echo ""
    echo "Follow these steps:"
    echo "  1. Choose 'n' for new remote"
    echo "  2. Name it: gdrive"
    echo "  3. Choose 'drive' (Google Drive)"
    echo "  4. Leave client_id and client_secret blank"
    echo "  5. Choose scope: 1 (Full access)"
    echo "  6. Leave root_folder_id blank"
    echo "  7. Leave service_account_file blank"
    echo "  8. Choose 'n' for advanced config"
    echo "  9. Choose 'n' for auto config (since we're on remote server)"
    echo "  10. Copy the URL to your browser, authenticate, paste the code"
    echo ""
    
    rclone config
    
    log_success "rclone configured!"
}

# ===========================================
# Download Functions
# ===========================================

download_with_gdown() {
    local file_id="$1"
    local output_path="$2"
    local name="$3"
    
    log_info "Downloading $name..."
    
    mkdir -p "$(dirname "$output_path")"
    
    # Try folder download first, then file
    if gdown --folder "https://drive.google.com/drive/folders/$file_id" -O "$output_path" 2>/dev/null; then
        log_success "Downloaded folder: $name"
    elif gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path" 2>/dev/null; then
        log_success "Downloaded file: $name"
    else
        log_warn "Failed to download $name (ID: $file_id)"
        return 1
    fi
}

download_with_rclone() {
    local remote_path="$1"
    local local_path="$2"
    local name="$3"
    
    log_info "Syncing $name with rclone..."
    
    mkdir -p "$local_path"
    
    rclone copy "${GDRIVE_REMOTE}:${remote_path}" "$local_path" \
        --progress \
        --transfers=4 \
        --checkers=8 \
        --contimeout=60s \
        --timeout=300s \
        --retries=3 \
        --low-level-retries=10 \
        || {
            log_warn "Failed to sync $name"
            return 1
        }
    
    log_success "Synced: $name"
}

# Download all datasets from a shared Google Drive folder path
download_gdrive_folder() {
    local gdrive_path="$1"
    local local_dir="$2"
    
    log_header "Downloading from: $gdrive_path"
    
    mkdir -p "$local_dir"
    
    rclone copy "${GDRIVE_REMOTE}:${gdrive_path}" "$local_dir" \
        --progress \
        --transfers=4 \
        --checkers=8 \
        --contimeout=60s \
        --timeout=300s \
        --retries=3 \
        --low-level-retries=10
    
    log_success "Downloaded to: $local_dir"
}

# ===========================================
# Main Download Commands
# ===========================================

download_cot_datasets() {
    log_header "Downloading CoT Reasoning Datasets"
    
    local target_dir="$DATASETS_DIR/cot_reasoning"
    mkdir -p "$target_dir"
    
    # Try canonical structure first (if reorganized)
    if rclone lsd "${GDRIVE_REMOTE}:datasets/cot_reasoning" &>/dev/null; then
        log_info "Using canonical structure: datasets/cot_reasoning/"
        download_gdrive_folder "datasets/cot_reasoning" "$target_dir"
    else
        # Fallback to old structure
        log_info "Using legacy structure - downloading individual CoT datasets"
        # If you have a shared folder with all CoT datasets
        # download_gdrive_folder "datasets/CoT_Reasoning" "$target_dir"
        
        # Or download individual datasets
        for name in "${!COT_DATASETS[@]}"; do
            local folder_id="${COT_DATASETS[$name]}"
            if [ "$folder_id" != "FOLDER_ID_HERE" ]; then
                download_with_gdown "$folder_id" "$target_dir/$name" "$name"
            else
                log_warn "Skipping $name - no folder ID configured"
            fi
        done
    fi
    
    log_success "CoT datasets downloaded to: $target_dir"
}

download_professional_datasets() {
    log_header "Downloading Professional Therapeutic Datasets"
    
    local target_dir="$DATASETS_DIR/professional_therapeutic"
    mkdir -p "$target_dir"
    
    # Try canonical structure first (if reorganized)
    if rclone lsd "${GDRIVE_REMOTE}:datasets/professional_therapeutic" &>/dev/null; then
        log_info "Using canonical structure: datasets/professional_therapeutic/"
        download_gdrive_folder "datasets/professional_therapeutic" "$target_dir"
    else
        # Fallback to old structure
        log_info "Using legacy structure - downloading individual professional datasets"
        for name in "${!PROFESSIONAL_DATASETS[@]}"; do
            local folder_id="${PROFESSIONAL_DATASETS[$name]}"
            if [ "$folder_id" != "FOLDER_ID_HERE" ]; then
                download_with_gdown "$folder_id" "$target_dir/$name" "$name"
            else
                log_warn "Skipping $name - no folder ID configured"
            fi
        done
    fi
    
    log_success "Professional datasets downloaded to: $target_dir"
}

download_priority_datasets() {
    log_header "Downloading Priority Datasets"
    
    local target_dir="$DATASETS_DIR/priority"
    mkdir -p "$target_dir"
    
    # Try canonical structure first (priority/ renamed from datasets-wendy/)
    if rclone lsd "${GDRIVE_REMOTE}:datasets/priority" &>/dev/null; then
        log_info "Using canonical structure: datasets/priority/"
        download_gdrive_folder "datasets/priority" "$target_dir"
    elif rclone lsd "${GDRIVE_REMOTE}:datasets/datasets-wendy" &>/dev/null; then
        # Fallback to old structure
        log_info "Using legacy structure: datasets/datasets-wendy/"
        download_gdrive_folder "datasets/datasets-wendy" "$target_dir"
    else
        # Fallback to individual downloads
        log_info "Downloading individual priority datasets"
        for name in "${!PRIORITY_DATASETS[@]}"; do
            local folder_id="${PRIORITY_DATASETS[$name]}"
            if [ "$folder_id" != "FOLDER_ID_HERE" ]; then
                download_with_gdown "$folder_id" "$target_dir/$name.jsonl" "$name"
            else
                log_warn "Skipping $name - no folder ID configured"
            fi
        done
    fi
    
    log_success "Priority datasets downloaded to: $target_dir"
}

download_all() {
    log_header "Downloading All Datasets"
    
    download_cot_datasets
    download_professional_datasets
    download_priority_datasets
    
    log_success "All datasets downloaded!"
    show_inventory
}

# Interactive download using rclone's folder path
download_interactive() {
    log_header "Interactive Download Mode"
    
    echo "Enter the Google Drive folder path (e.g., 'datasets' or 'My Drive/datasets'):"
    read -r gdrive_path
    
    echo "Enter local destination directory (default: $DATASETS_DIR):"
    read -r local_dir
    local_dir="${local_dir:-$DATASETS_DIR}"
    
    download_gdrive_folder "$gdrive_path" "$local_dir"
}

# List Google Drive contents
list_gdrive() {
    local path="${1:-.}"
    
    log_info "Listing Google Drive: $path"
    rclone lsd "${GDRIVE_REMOTE}:${path}" 2>/dev/null || {
        log_error "Failed to list. Make sure rclone is configured."
        exit 1
    }
}

# ===========================================
# Inventory & Status
# ===========================================

show_inventory() {
    log_header "Dataset Inventory"
    
    if [ ! -d "$DATASETS_DIR" ]; then
        log_warn "Datasets directory not found: $DATASETS_DIR"
        return
    fi
    
    echo "Location: $DATASETS_DIR"
    echo ""
    
    # Show size of each subdirectory
    for dir in "$DATASETS_DIR"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            local count=$(find "$dir" -type f | wc -l)
            echo "  📁 $name: $size ($count files)"
        fi
    done
    
    echo ""
    local total_size=$(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1)
    echo "Total: $total_size"
}

check_status() {
    log_header "System Status"
    
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Disk Space:"
    df -h "$HOME" | tail -1
    echo ""
    
    echo "Dependencies:"
    command -v gdown &> /dev/null && echo "  ✅ gdown" || echo "  ❌ gdown"
    command -v rclone &> /dev/null && echo "  ✅ rclone" || echo "  ❌ rclone"
    
    echo ""
    echo "rclone remotes:"
    rclone listremotes 2>/dev/null || echo "  (not configured)"
    
    echo ""
    echo "Datasets directory: $DATASETS_DIR"
    [ -d "$DATASETS_DIR" ] && echo "  ✅ exists" || echo "  ❌ not found"
}

# ===========================================
# Usage
# ===========================================

usage() {
    echo "Google Drive Dataset Download Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Setup Commands:"
    echo "  setup             Full setup (install deps + configure rclone)"
    echo "  install-deps      Install gdown and rclone"
    echo "  configure-rclone  Configure rclone for Google Drive"
    echo ""
    echo "Download Commands:"
    echo "  download-all      Download all configured datasets"
    echo "  download-cot      Download CoT reasoning datasets"
    echo "  download-pro      Download professional therapeutic datasets"
    echo "  download-priority Download priority datasets (datasets-wendy)"
    echo "  download-interactive  Interactive download mode"
    echo ""
    echo "Utility Commands:"
    echo "  list [path]       List Google Drive contents"
    echo "  inventory         Show downloaded dataset inventory"
    echo "  status            Check system and dependencies"
    echo ""
    echo "Environment Variables:"
    echo "  DATASETS_DIR      Download directory (default: ~/datasets)"
    echo "  GDRIVE_REMOTE     rclone remote name (default: gdrive)"
    echo ""
    echo "Quick Start:"
    echo "  1. $0 setup                    # Install and configure"
    echo "  2. $0 list                     # Browse Google Drive"
    echo "  3. $0 download-interactive     # Download specific folder"
    echo ""
    echo "Or edit the FOLDER_ID values in this script and run:"
    echo "  $0 download-all"
}

# ===========================================
# Main
# ===========================================

main() {
    case "${1:-}" in
        setup)
            install_deps
            setup_rclone
            mkdir -p "$DATASETS_DIR"
            log_success "Setup complete! Run '$0 list' to browse Google Drive"
            ;;
        install-deps)
            install_deps
            ;;
        configure-rclone|config)
            setup_rclone
            ;;
        download-all)
            download_all
            ;;
        download-cot)
            download_cot_datasets
            ;;
        download-pro|download-professional)
            download_professional_datasets
            ;;
        download-priority)
            download_priority_datasets
            ;;
        download-interactive|interactive)
            download_interactive
            ;;
        list|ls)
            list_gdrive "${2:-.}"
            ;;
        inventory|inv)
            show_inventory
            ;;
        status)
            check_status
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            if [ -z "${1:-}" ]; then
                usage
            else
                log_error "Unknown command: $1"
                usage
                exit 1
            fi
            ;;
    esac
}

main "$@"

