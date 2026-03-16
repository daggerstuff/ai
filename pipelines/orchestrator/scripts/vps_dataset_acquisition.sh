#!/bin/bash
#
# VPS Dataset Acquisition Script
# Downloads all datasets on VPS (146.71.78.184) for fast bulk acquisition
#
# Usage: Run this script directly on the VPS (no persistent SSH connection needed)
#   
#   Option 1: Interactive mode (will continue if SSH disconnects)
#     ssh -i ~/.ssh/planet vivi@146.71.78.184
#     cd ~/pixelated-datasets
#     nohup ./vps_dataset_acquisition.sh > download_nohup.out 2>&1 &
#     exit  # You can disconnect now
#
#   Option 2: Background daemon mode
#     ssh -i ~/.ssh/planet vivi@146.71.78.184 "cd ~/pixelated-datasets && nohup ./vps_dataset_acquisition.sh --daemon > /dev/null 2>&1 &"
#
#   Option 3: Using screen/tmux (recommended)
#     ssh -i ~/.ssh/planet vivi@146.71.78.184
#     screen -S dataset-download
#     cd ~/pixelated-datasets && ./vps_dataset_acquisition.sh
#     # Press Ctrl+A then D to detach. Reattach with: screen -r dataset-download
#
#   Check status:
#     ssh -i ~/.ssh/planet vivi@146.71.78.184 "tail -f ~/pixelated-datasets/download.log"

set -euo pipefail

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/pixelated-datasets}"
RAW_DIR="$WORKSPACE_DIR/raw"
LOG_FILE="$WORKSPACE_DIR/download.log"
ERROR_LOG="$WORKSPACE_DIR/error.log"
PID_FILE="$WORKSPACE_DIR/download.pid"
STATUS_FILE="$WORKSPACE_DIR/download.status"

# Background execution flag
DAEMON_MODE=false
if [[ "${1:-}" == "--daemon" ]]; then
    DAEMON_MODE=true
    exec 0</dev/null
    exec 1>>"$LOG_FILE"
    exec 2>>"$ERROR_LOG"
fi

# Create workspace directory structure if it doesn't exist
mkdir -p "$WORKSPACE_DIR"/{raw/{huggingface,kaggle,gdrive,local},processed,exports,inventory}
cd "$WORKSPACE_DIR" || exit 1

# If script is in a different location, copy itself to workspace for easier access
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$SCRIPT_DIR" != "$WORKSPACE_DIR" ] && [ ! -f "$WORKSPACE_DIR/vps_dataset_acquisition.sh" ]; then
    if [ -f "${BASH_SOURCE[0]}" ]; then
        cp "${BASH_SOURCE[0]}" "$WORKSPACE_DIR/vps_dataset_acquisition.sh"
        chmod +x "$WORKSPACE_DIR/vps_dataset_acquisition.sh"
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    local msg="[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    if [ "$DAEMON_MODE" = true ]; then
        echo "$msg" >> "$LOG_FILE"
    else
        echo -e "${GREEN}$msg${NC}" | tee -a "$LOG_FILE"
    fi
    update_status "INFO: $1"
}

log_warn() {
    local msg="[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    if [ "$DAEMON_MODE" = true ]; then
        echo "$msg" >> "$LOG_FILE"
    else
        echo -e "${YELLOW}$msg${NC}" | tee -a "$LOG_FILE"
    fi
    update_status "WARN: $1"
}

log_error() {
    local msg="[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    if [ "$DAEMON_MODE" = true ]; then
        echo "$msg" >> "$ERROR_LOG"
    else
        echo -e "${RED}$msg${NC}" | tee -a "$ERROR_LOG"
    fi
    update_status "ERROR: $1"
}

# Status tracking
update_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')|$1" >> "$STATUS_FILE"
    # Keep only last 100 status updates
    tail -n 100 "$STATUS_FILE" > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

# Save PID and start info
save_pid() {
    echo $$ > "$PID_FILE"
    update_status "Started: PID $$"
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        update_status "Completed successfully"
        log_info "Dataset acquisition completed successfully"
    else
        update_status "Failed with exit code $exit_code"
        log_error "Dataset acquisition failed with exit code $exit_code"
    fi
    rm -f "$PID_FILE"
    exit $exit_code
}

trap cleanup EXIT

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python and pip
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check required Python packages
    python3 << EOF
import sys
required = ['huggingface_hub', 'kaggle']
missing = []
for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        log_warn "Installing required Python packages..."
        pip3 install --user huggingface_hub[cli] datasets kaggle awscli boto3
    fi
    
    # Check for HuggingFace token
    if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.huggingface/token" ]; then
        log_warn "HuggingFace token not found. Set HF_TOKEN env var or run 'huggingface-cli login'"
    fi
    
    # Check for Kaggle credentials
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        log_warn "Kaggle credentials not found. Place kaggle.json in ~/.kaggle/ and chmod 600"
    fi
    
    log_info "Prerequisites check complete"
}

# Create directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    mkdir -p "$RAW_DIR"/{huggingface,kaggle,gdrive,local}
    mkdir -p "$WORKSPACE_DIR"/{processed,exports,inventory}
    log_info "Directory structure created"
}

# Download HuggingFace datasets
download_huggingface() {
    log_info "Starting HuggingFace dataset downloads..."
    
    local hf_dir="$RAW_DIR/huggingface"
    
    # Embedded dataset list (self-contained, no external JSON needed)
    python3 << 'PYTHON_EOF'
import os
import subprocess
import sys
import time

# Dataset definitions embedded in script
hf_datasets = [
    {"id": "iqrakiran/customized-mental-health-snli2", "dest": "customized-mental-health-snli2"},
    {"id": "typosonlr/MentalHealthPreProcessed", "dest": "MentalHealthPreProcessed"},
    {"id": "ShreyaR/DepressionDetection", "dest": "DepressionDetection"},
    {"id": "mlx-community/Human-Like-DPO", "dest": "Human-Like-DPO"},
    {"id": "flammenai/character-roleplay-DPO", "dest": "character-roleplay-DPO"},
    {"id": "PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT", "dest": "unalignment_toxic-dpo"},
    {"id": "Amod/mental_health_counseling_conversations", "dest": "mental_health_counseling_conversations"},
    {"id": "thu-coai/esconv", "dest": "esconv"},
    {"id": "nbertagnolli/counsel-chat", "dest": "counsel-chat"},
    {"id": "facebook/empathetic_dialogues", "dest": "empathetic_dialogues"},
]

base_dir = os.path.expanduser("~/pixelated-datasets/raw/huggingface")
failed = []
successful = []

for ds in hf_datasets:
    repo_id = ds['id']
    dest = os.path.join(base_dir, ds['dest'])
    
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Destination: {dest}")
    print(f"{'='*60}")
    
    os.makedirs(dest, exist_ok=True)
    
    try:
        # Try huggingface-cli first
        cmd = ['huggingface-cli', 'download', repo_id, '--local-dir', dest]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        print(f"✓ Successfully downloaded {repo_id}")
        successful.append(repo_id)
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout downloading {repo_id}")
        failed.append(repo_id)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {repo_id}")
        print(f"Error: {e.stderr}")
        failed.append(repo_id)
    except Exception as e:
        print(f"✗ Error downloading {repo_id}: {e}")
        failed.append(repo_id)
    
    # Small delay between downloads to avoid rate limiting
    time.sleep(2)

print(f"\n{'='*60}")
print(f"Download Summary:")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
print(f"{'='*60}")

if failed:
    print(f"\nFailed downloads: {', '.join(failed)}")
    # Don't exit with error - continue with other downloads
else:
    print("\nAll HuggingFace downloads completed successfully!")
PYTHON_EOF

    log_info "HuggingFace downloads complete"
}

# Download Kaggle datasets
download_kaggle() {
    log_info "Starting Kaggle dataset downloads..."
    
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        log_error "Kaggle credentials not found. Skipping Kaggle downloads."
        return
    fi
    
    chmod 600 "$HOME/.kaggle/kaggle.json"
    
    local kaggle_dir="$RAW_DIR/kaggle"
    
    # Download NLP for Mental Health TF-IDF archive
    log_info "Downloading Kaggle dataset: rickyzou/nlp-for-mental-health-text-data"
    
    mkdir -p "$kaggle_dir/nlp-mental-health-tfidf"
    
    if [ "$DAEMON_MODE" = true ]; then
        kaggle kernels output rickyzou/nlp-for-mental-health-text-data \
            -p "$kaggle_dir/nlp-mental-health-tfidf/" \
            >> "$LOG_FILE" 2>&1
    else
        kaggle kernels output rickyzou/nlp-for-mental-health-text-data \
            -p "$kaggle_dir/nlp-mental-health-tfidf/" \
            2>&1 | tee -a "$LOG_FILE"
    fi
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "Kaggle downloads complete"
    else
        log_warn "Kaggle download may have failed. Check log."
    fi
}

# Sync Google Drive datasets
sync_google_drive() {
    log_info "Starting Google Drive dataset sync..."
    
    if ! command -v rclone &> /dev/null; then
        log_warn "rclone not found. Install with: curl https://rclone.org/install.sh | sudo bash"
        log_warn "Or use gdrive CLI. Skipping Google Drive sync."
        return
    fi
    
    # Check for rclone config
    if [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
        log_warn "rclone not configured. Run 'rclone config' first."
        log_warn "Skipping Google Drive sync."
        return
    fi
    
    local gdrive_dir="$RAW_DIR/gdrive"
    
    log_info "Syncing Google Drive datasets..."
    log_warn "Note: This requires rclone remote named 'gdrive' to be configured"
    
    # Sync datasets
    if [ "$DAEMON_MODE" = true ]; then
        rclone copy gdrive:datasets/therapist-sft-format "$gdrive_dir/therapist-sft-format" >> "$LOG_FILE" 2>&1
        rclone copy gdrive:datasets/SoulChat2.0 "$gdrive_dir/SoulChat2.0" >> "$LOG_FILE" 2>&1
        rclone copy gdrive:datasets/counsel-chat "$gdrive_dir/counsel-chat" >> "$LOG_FILE" 2>&1
        rclone copy gdrive:datasets/Psych8k "$gdrive_dir/Psych8k" >> "$LOG_FILE" 2>&1
        rclone copy gdrive:datasets/neuro_qa_SFT_Trainer "$gdrive_dir/neuro_qa_SFT_Trainer" >> "$LOG_FILE" 2>&1
        rclone copy gdrive:datasets "$gdrive_dir/cot" --include "CoT_*" >> "$LOG_FILE" 2>&1
    else
        rclone copy gdrive:datasets/therapist-sft-format "$gdrive_dir/therapist-sft-format" -P 2>&1 | tee -a "$LOG_FILE"
        rclone copy gdrive:datasets/SoulChat2.0 "$gdrive_dir/SoulChat2.0" -P 2>&1 | tee -a "$LOG_FILE"
        rclone copy gdrive:datasets/counsel-chat "$gdrive_dir/counsel-chat" -P 2>&1 | tee -a "$LOG_FILE"
        rclone copy gdrive:datasets/Psych8k "$gdrive_dir/Psych8k" -P 2>&1 | tee -a "$LOG_FILE"
        rclone copy gdrive:datasets/neuro_qa_SFT_Trainer "$gdrive_dir/neuro_qa_SFT_Trainer" -P 2>&1 | tee -a "$LOG_FILE"
        rclone copy gdrive:datasets "$gdrive_dir/cot" --include "CoT_*" -P 2>&1 | tee -a "$LOG_FILE"
    fi
    
    log_info "Google Drive sync complete"
}

# Generate checksums and inventory
generate_inventory() {
    log_info "Generating VPS inventory manifest..."
    
    python3 << 'INVENTORY_EOF'
import json
import hashlib
import os
import datetime
from pathlib import Path

def calculate_checksum(filepath):
    """Calculate SHA256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_dir_size(path):
    """Get total size of directory in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

base_dir_str = os.path.expanduser("~/pixelated-datasets/raw")
workspace_dir_str = os.path.expanduser("~/pixelated-datasets")

inventory = {
    "vps_host": os.uname().nodename,
    "generated": datetime.datetime.now().isoformat(),
    "base_dir": workspace_dir_str,
    "datasets": {}
}

base_dir = Path(base_dir_str)
total_size = 0

for source_dir in ['huggingface', 'kaggle', 'gdrive']:
    source_path = base_dir / source_dir
    if not source_path.exists():
        continue
    
    inventory["datasets"][source_dir] = {
        "path": str(source_path),
        "size_bytes": get_dir_size(source_path),
        "items": []
    }
    
    for item_path in source_path.iterdir():
        if item_path.is_dir():
            size = get_dir_size(item_path)
            total_size += size
            
            files = []
            for file_path in item_path.rglob('*'):
                if file_path.is_file():
                    try:
                        checksum = calculate_checksum(file_path)
                        files.append({
                            "path": str(file_path.relative_to(source_path)),
                            "size_bytes": file_path.stat().st_size,
                            "sha256": checksum
                        })
                    except Exception as e:
                        print(f"Warning: Could not checksum {file_path}: {e}")
            
            inventory["datasets"][source_dir]["items"].append({
                "name": item_path.name,
                "size_bytes": size,
                "file_count": len(files),
                "files": files[:10]  # Limit to first 10 files for manifest size
            })

inventory["total_size_bytes"] = total_size
inventory["total_size_gb"] = round(total_size / (1024**3), 2)

output_file = os.path.join(workspace_dir_str, "inventory", "VPS_INVENTORY.json")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(inventory, f, indent=2)

print(f"✓ Inventory generated: {output_file}")
print(f"Total datasets size: {inventory['total_size_gb']} GB")
INVENTORY_EOF

    log_info "VPS inventory manifest generated: $WORKSPACE_DIR/inventory/VPS_INVENTORY.json"
}

# Main execution
main() {
    # Save PID for status tracking
    save_pid
    
    log_info "Starting VPS dataset acquisition..."
    log_info "Workspace: $WORKSPACE_DIR"
    log_info "Log file: $LOG_FILE"
    log_info "PID: $$"
    
    if [ "$DAEMON_MODE" = true ]; then
        log_info "Running in daemon mode"
    fi
    
    # Change to workspace directory
    cd "$WORKSPACE_DIR" || exit 1
    
    # Check prerequisites
    check_prerequisites
    
    # Setup directories
    setup_directories
    
    # Download datasets
    update_status "Starting HuggingFace downloads"
    download_huggingface
    
    update_status "Starting Kaggle downloads"
    download_kaggle
    
    update_status "Starting Google Drive sync"
    sync_google_drive
    
    # Generate inventory
    update_status "Generating inventory manifest"
    generate_inventory
    
    log_info "Dataset acquisition complete!"
    log_info "Check logs: $LOG_FILE"
    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        log_warn "Errors occurred. Check: $ERROR_LOG"
    fi
    
    update_status "All downloads complete"
}

# Status check function (can be called separately)
check_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Process is running (PID: $pid)"
            echo "Last status updates:"
            tail -n 10 "$STATUS_FILE" 2>/dev/null || echo "No status updates yet"
            echo ""
            echo "Recent log entries:"
            tail -n 5 "$LOG_FILE" 2>/dev/null || echo "No log entries yet"
        else
            echo "Process is not running (PID file exists but process is dead)"
            rm -f "$PID_FILE"
        fi
    else
        echo "No download process is currently running"
    fi
}

# Show help
show_help() {
    cat << EOF
VPS Dataset Acquisition Script

Usage:
  ./vps_dataset_acquisition.sh [options]

Options:
  --daemon          Run in background daemon mode
  --status          Check current download status
  --help            Show this help message

Examples:

  # Run interactively (will continue if SSH disconnects)
  nohup ./vps_dataset_acquisition.sh > download_nohup.out 2>&1 &
  
  # Run as daemon
  ./vps_dataset_acquisition.sh --daemon
  
  # Check status
  ./vps_dataset_acquisition.sh --status
  
  # Using screen (recommended)
  screen -S dataset-download
  ./vps_dataset_acquisition.sh
  # Press Ctrl+A then D to detach
  
  # Reattach to screen
  screen -r dataset-download

Logs:
  Main log: $LOG_FILE
  Error log: $ERROR_LOG
  Status: $STATUS_FILE

EOF
}

# Handle command line arguments
if [ "${1:-}" == "--status" ]; then
    check_status
    exit 0
elif [ "${1:-}" == "--help" ] || [ "${1:-}" == "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main

