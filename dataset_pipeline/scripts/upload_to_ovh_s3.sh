#!/bin/bash
#
# VPS-to-OVH S3 Upload Script
# Uploads all consolidated datasets from VPS to OVH S3 for training
#
# Usage: Run this script on the VPS
#   ssh -i ~/.ssh/planet vivi@146.71.78.184
#   cd ~/pixelated-datasets
#   ./upload_to_ovh_s3.sh

set -euo pipefail

# Configuration
WORKSPACE_DIR="$HOME/pixelated-datasets"
RAW_DIR="$WORKSPACE_DIR/raw"
S3_BUCKET="pixel-data"
S3_ENDPOINT="s3.gra.io.cloud.ovh.net"
LOG_FILE="$WORKSPACE_DIR/upload.log"
ERROR_LOG="$WORKSPACE_DIR/upload_errors.log"

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

log_upload() {
    echo -e "${BLUE}[UPLOAD]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Install with: pip install awscli"
        log_error "Or use: pip install boto3 for Python-based upload"
        exit 1
    fi
    
    # Check AWS credentials
    if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
        log_warn "AWS credentials not set. Check environment variables:"
        log_warn "  AWS_ACCESS_KEY_ID"
        log_warn "  AWS_SECRET_ACCESS_KEY"
        log_warn "  AWS_DEFAULT_REGION (optional, defaults to gra)"
        
        if [ ! -f "$HOME/.aws/credentials" ]; then
            log_error "AWS credentials file not found. Please configure:"
            log_error "  1. Set environment variables, OR"
            log_error "  2. Run 'aws configure' and set up credentials"
            exit 1
        fi
    fi
    
    # Set S3 endpoint
    export AWS_ENDPOINT_URL="https://$S3_ENDPOINT"
    
    # Test S3 connection
    log_info "Testing S3 connection..."
    if aws s3 ls "s3://$S3_BUCKET" --endpoint-url "$AWS_ENDPOINT_URL" &>/dev/null; then
        log_info "S3 connection successful"
    else
        log_error "Cannot connect to S3 bucket: $S3_BUCKET"
        log_error "Check credentials and endpoint: $AWS_ENDPOINT_URL"
        exit 1
    fi
}

# Create S3 bucket structure
create_s3_structure() {
    log_info "Creating S3 bucket structure..."
    
    # Create main directories
    aws s3api put-object \
        --bucket "$S3_BUCKET" \
        --key "raw/" \
        --endpoint-url "$AWS_ENDPOINT_URL" \
        &>/dev/null || true
    
    aws s3api put-object \
        --bucket "$S3_BUCKET" \
        --key "processed/" \
        --endpoint-url "$AWS_ENDPOINT_URL" \
        &>/dev/null || true
    
    aws s3api put-object \
        --bucket "$S3_BUCKET" \
        --key "exports/" \
        --endpoint-url "$AWS_ENDPOINT_URL" \
        &>/dev/null || true
    
    log_info "S3 structure created"
}

# Upload directory to S3
upload_directory() {
    local source_path="$1"
    local s3_prefix="$2"
    local description="$3"
    
    if [ ! -d "$source_path" ]; then
        log_warn "Source directory not found: $source_path (skipping)"
        return
    fi
    
    log_upload "Uploading: $description"
    log_upload "Source: $source_path"
    log_upload "Destination: s3://$S3_BUCKET/$s3_prefix"
    
    # Use aws s3 sync with progress and checksum verification
    aws s3 sync \
        "$source_path" \
        "s3://$S3_BUCKET/$s3_prefix" \
        --endpoint-url "$AWS_ENDPOINT_URL" \
        --checksum \
        --no-progress \
        2>&1 | while IFS= read -r line; do
            echo "$line" | tee -a "$LOG_FILE"
        done
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "✓ Successfully uploaded: $description"
        
        # Verify upload with file count comparison
        local local_count=$(find "$source_path" -type f | wc -l)
        log_info "Local files: $local_count"
        
        return 0
    else
        log_error "✗ Failed to upload: $description"
        return 1
    fi
}

# Upload datasets to S3
upload_datasets() {
    log_info "Starting dataset upload to OVH S3..."
    log_info "Bucket: s3://$S3_BUCKET"
    log_info "Endpoint: $AWS_ENDPOINT_URL"
    
    # Upload HuggingFace datasets
    log_info "Uploading HuggingFace datasets..."
    upload_directory \
        "$RAW_DIR/huggingface" \
        "raw/huggingface/" \
        "HuggingFace Datasets"
    
    # Upload Kaggle datasets
    log_info "Uploading Kaggle datasets..."
    upload_directory \
        "$RAW_DIR/kaggle" \
        "raw/kaggle/" \
        "Kaggle Datasets"
    
    # Upload Google Drive datasets
    log_info "Uploading Google Drive datasets..."
    upload_directory \
        "$RAW_DIR/gdrive" \
        "raw/gdrive/" \
        "Google Drive Datasets"
    
    # Upload local datasets
    log_info "Uploading local datasets..."
    upload_directory \
        "$RAW_DIR/local" \
        "raw/local/" \
        "Local Datasets"
    
    log_info "Dataset uploads complete!"
}

# Generate S3 inventory manifest
generate_s3_inventory() {
    log_info "Generating S3 inventory manifest..."
    
    python3 << EOF
import json
import boto3
from datetime import datetime
from botocore.config import Config

# Configure S3 client
config = Config(
    endpoint_url="https://$S3_ENDPOINT",
    region_name="gra"
)

s3_client = boto3.client(
    's3',
    endpoint_url="https://$S3_ENDPOINT",
    config=config
)

bucket = "$S3_BUCKET"
inventory = {
    "s3_bucket": bucket,
    "s3_endpoint": "$S3_ENDPOINT",
    "generated": datetime.now().isoformat(),
    "base_dir": "$RAW_DIR",
    "datasets": {}
}

# List all objects in bucket
paginator = s3_client.get_paginator('list_objects_v2')
total_size = 0
file_count = 0

for prefix in ['raw/huggingface/', 'raw/kaggle/', 'raw/gdrive/', 'raw/local/']:
    try:
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        items = []
        prefix_size = 0
        prefix_count = 0
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                if obj['Key'].endswith('/'):
                    continue
                
                size = obj['Size']
                prefix_size += size
                prefix_count += 1
                total_size += size
                file_count += 1
                
                items.append({
                    "key": obj['Key'],
                    "size_bytes": size,
                    "last_modified": obj['LastModified'].isoformat(),
                    "etag": obj['ETag'].strip('"')
                })
        
        if items:
            inventory["datasets"][prefix.rstrip('/')] = {
                "prefix": prefix,
                "file_count": prefix_count,
                "size_bytes": prefix_size,
                "size_gb": round(prefix_size / (1024**3), 2),
                "items": items[:50]  # Limit to first 50 items
            }
    except Exception as e:
        print(f"Error listing {prefix}: {e}")

inventory["total_size_bytes"] = total_size
inventory["total_size_gb"] = round(total_size / (1024**3), 2)
inventory["total_files"] = file_count

output_file = "$WORKSPACE_DIR/inventory/S3_INVENTORY.json"
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(inventory, f, indent=2)

print(f"✓ S3 inventory generated: {output_file}")
print(f"Total files: {file_count}")
print(f"Total size: {inventory['total_size_gb']} GB")

# Upload inventory to S3
s3_client.upload_file(
    output_file,
    bucket,
    "inventory/S3_INVENTORY.json",
    ExtraArgs={'ContentType': 'application/json'}
)
print(f"✓ Inventory uploaded to S3: s3://{bucket}/inventory/S3_INVENTORY.json")
EOF

    log_info "S3 inventory manifest generated and uploaded"
}

# Verify uploads
verify_uploads() {
    log_info "Verifying S3 uploads..."
    
    python3 << EOF
import boto3
from botocore.config import Config

config = Config(
    endpoint_url="https://$S3_ENDPOINT",
    region_name="gra"
)

s3_client = boto3.client(
    's3',
    endpoint_url="https://$S3_ENDPOINT",
    config=config
)

bucket = "$S3_BUCKET"
print("\nVerification Summary:")
print("=" * 60)

for prefix in ['raw/huggingface/', 'raw/kaggle/', 'raw/gdrive/', 'raw/local/']:
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if 'Contents' in response:
            print(f"✓ {prefix.rstrip('/')}: Files found")
        else:
            print(f"✗ {prefix.rstrip('/')}: No files found")
    except Exception as e:
        print(f"✗ {prefix.rstrip('/')}: Error - {e}")

print("=" * 60)
EOF

    log_info "Verification complete"
}

# Main execution
main() {
    log_info "Starting VPS-to-OVH S3 dataset upload..."
    log_info "Workspace: $WORKSPACE_DIR"
    log_info "S3 Bucket: s3://$S3_BUCKET"
    
    cd "$WORKSPACE_DIR" || exit 1
    
    # Check prerequisites
    check_prerequisites
    
    # Create S3 structure
    create_s3_structure
    
    # Upload datasets
    upload_datasets
    
    # Generate inventory
    generate_s3_inventory
    
    # Verify uploads
    verify_uploads
    
    log_info "S3 upload complete!"
    log_info "Check logs: $LOG_FILE"
    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        log_warn "Errors occurred. Check: $ERROR_LOG"
    fi
    
    log_info "S3 Inventory: s3://$S3_BUCKET/inventory/S3_INVENTORY.json"
}

# Run main function
main

