#!/bin/bash
# HETZNER Object Storage Sync Script
# Uploads processed datasets to HETZNER S3 for training (using native ${HETZNER_AI_CLI} CLI)
#
# Prerequisites:
# 1. ${HETZNER_AI_CLI} login (authenticate first)
# 2. Run from local machine OR remote server with ${HETZNER_AI_CLI} installed
#
# Usage:
#   ./sync_datasets.sh upload    # Sync generated datasets to HETZNER
#   ./sync_datasets.sh status    # Check bucket contents
#   ./sync_datasets.sh create    # Ensure bucket exists
#

set -e
HETZNER_AI_CLI="${HETZNER_AI_CLI:-ovhai}"

# Configuration
BUCKET_NAME="pixel-data"
REGION="hel1" # Updated per user instructions (was GRA, now hel1)
GENERATED_DATA_DIR="ai/training_ready/data/generated"
REMOTE_ALIAS="hetzner_s3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_auth() {
    log "Checking HETZNER authentication..."
    if ! ${HETZNER_AI_CLI} datastore list &>/dev/null; then
        log "ERROR: Not authenticated. Run '${HETZNER_AI_CLI} login' first"
        exit 1
    fi
    log "✓ Authenticated"
}

create_bucket() {
    log "Ensuring bucket '$BUCKET_NAME' exists in $REGION..."
    if ${HETZNER_AI_CLI} bucket list "$REGION" 2>/dev/null | grep -q "$BUCKET_NAME"; then
        log "✓ Bucket exists"
    else
        ${HETZNER_AI_CLI} bucket create "$BUCKET_NAME" "$REGION" || {
             log "WARNING: Creation returned non-zero, checking if it exists anyway..."
        }
        # Double check
        if ${HETZNER_AI_CLI} bucket list "$REGION" | grep -q "$BUCKET_NAME"; then
             log "✓ Bucket confirmed"
        else
             log "❌ Failed to create/verify bucket"
             exit 1
        fi
    fi
}

upload_datasets() {
    log "Syncing datasets from $GENERATED_DATA_DIR to $BUCKET_NAME..."
    
    if [ ! -d "$GENERATED_DATA_DIR" ]; then
        log "ERROR: Data directory $GENERATED_DATA_DIR not found!"
        exit 1
    fi

    # 1. YouTube Transcripts
    if [ -d "$GENERATED_DATA_DIR/youtube_transcripts" ]; then
        log "Uploading YouTube transcripts..."
        ${HETZNER_AI_CLI} bucket object upload "$BUCKET_NAME@$REGION" "$GENERATED_DATA_DIR/youtube_transcripts" \
            --prefix "datasets/youtube_transcripts/" \
            --recursive
    fi

    # 2. Academic Research
    if [ -d "$GENERATED_DATA_DIR/academic_research" ]; then
        log "Uploading Academic Research..."
        ${HETZNER_AI_CLI} bucket object upload "$BUCKET_NAME@$REGION" "$GENERATED_DATA_DIR/academic_research" \
            --prefix "datasets/academic_research/" \
            --recursive
    fi

    # 3. Therapeutic Books
    if [ -d "$GENERATED_DATA_DIR/therapeutic_books" ]; then
        log "Uploading Therapeutic Books..."
        ${HETZNER_AI_CLI} bucket object upload "$BUCKET_NAME@$REGION" "$GENERATED_DATA_DIR/therapeutic_books" \
            --prefix "datasets/therapeutic_books/" \
            --recursive
    fi
    
    # 4. NeMo Synthetic (if exists)
    if [ -d "$GENERATED_DATA_DIR/nemo_synthetic" ]; then
        log "Uploading NeMo synthetic data..."
        ${HETZNER_AI_CLI} bucket object upload "$BUCKET_NAME@$REGION" "$GENERATED_DATA_DIR/nemo_synthetic" \
            --prefix "datasets/nemo_synthetic/" \
            --recursive
    fi

    log "✓ Upload complete"
}

check_status() {
    log "Checking bucket contents..."
    ${HETZNER_AI_CLI} bucket object list "$BUCKET_NAME@$REGION" --output json | python3 -c '
import sys, json
try:
    data = json.load(sys.stdin)
    print(f"Total objects: {len(data)}")
    for item in data[:10]:
        print(f" - {item.get(\"name\", \"unknown\")}")
except Exception as e:
    print("Error parsing output:", e)
'
}

show_usage() {
    echo "HETZNER Dataset Sync Tool"
    echo "Usage: $0 {upload|status|create}"
}

# Main
case "${1:-}" in
    upload)
        check_auth
        create_bucket
        upload_datasets
        ;;
    status)
        check_auth
        check_status
        ;;
    create)
        check_auth
        create_bucket
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
