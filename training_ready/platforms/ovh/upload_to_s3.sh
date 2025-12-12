#!/bin/bash
# OVH Object Storage Upload Script
# Uploads processed datasets to OVH S3 for training transfer
#
# Prerequisites:
# 1. ovhai login (authenticate first)
# 2. Run from local machine OR remote server with ovhai installed
#
# Usage:
#   ./upload_to_s3.sh create    # Create bucket
#   ./upload_to_s3.sh upload    # Upload datasets
#   ./upload_to_s3.sh status    # Check upload status
#   ./upload_to_s3.sh all       # Create + Upload

set -e

# Configuration
BUCKET_NAME="pixelated-training-datasets"
REGION="GRA"  # OVH region (GRA, BHS, SBG, etc.)
LOCAL_DATA_DIR="${HOME}/datasets/consolidated"
REMOTE_SERVER="vivi@146.71.78.184"
SSH_KEY="${HOME}/.ssh/planet"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_auth() {
    log "Checking OVH authentication..."
    if ! ovhai datastore list &>/dev/null; then
        log "ERROR: Not authenticated. Run 'ovhai login' first"
        log ""
        log "To login:"
        log "  ovhai login -u YOUR_USERNAME --password-from-stdin"
        log ""
        log "Or with token:"
        log "  ovhai login --token YOUR_TOKEN"
        exit 1
    fi
    log "✓ Authenticated"
}

create_bucket() {
    log "Creating bucket: $BUCKET_NAME in region $REGION"
    
    # Check if bucket exists
    if ovhai bucket list "$REGION" 2>/dev/null | grep -q "$BUCKET_NAME"; then
        log "✓ Bucket already exists"
    else
        ovhai bucket create "$BUCKET_NAME" "$REGION" || {
            log "Note: Bucket may already exist or creation requires different permissions"
        }
        log "✓ Bucket created/verified"
    fi
}

upload_from_remote() {
    log "Uploading datasets from remote server to OVH S3..."
    
    # Create upload script for remote execution
    UPLOAD_SCRIPT=$(cat << 'REMOTE_EOF'
#!/bin/bash
# This runs on the remote server

BUCKET_NAME="pixelated-training-datasets"
REGION="GRA"
DATA_DIR="${HOME}/datasets/consolidated"

# Check if ovhai is installed on remote
if ! command -v ovhai &>/dev/null; then
    echo "Installing ovhai on remote..."
    curl -sSL https://cli.gra.ai.cloud.ovh.net/install.sh | bash
    export PATH="$HOME/bin:$PATH"
fi

echo "Uploading from $DATA_DIR to $BUCKET_NAME..."

# Upload each category
for category in cot_reasoning professional processed misc; do
    if [ -d "$DATA_DIR/$category" ]; then
        echo "Uploading $category..."
        ovhai bucket object upload "$BUCKET_NAME@$REGION" "$DATA_DIR/$category" \
            --prefix "raw/$category/" \
            --recursive || echo "Failed to upload $category"
    fi
done

# Upload ChatML formatted data if it exists
if [ -d "$DATA_DIR/chatml_formatted" ]; then
    echo "Uploading ChatML formatted data..."
    ovhai bucket object upload "$BUCKET_NAME@$REGION" "$DATA_DIR/chatml_formatted" \
        --prefix "chatml/" \
        --recursive || echo "Failed to upload chatml_formatted"
fi

echo "Upload complete!"
REMOTE_EOF
)

    # Execute on remote
    ssh -i "$SSH_KEY" "$REMOTE_SERVER" "$UPLOAD_SCRIPT"
}

upload_local() {
    log "Uploading from local directory..."
    
    if [ ! -d "$LOCAL_DATA_DIR" ]; then
        log "Local data directory not found: $LOCAL_DATA_DIR"
        log "Trying remote upload instead..."
        upload_from_remote
        return
    fi
    
    for category in cot_reasoning professional processed misc chatml_formatted; do
        if [ -d "$LOCAL_DATA_DIR/$category" ]; then
            log "Uploading $category..."
            ovhai bucket object upload "$BUCKET_NAME@$REGION" "$LOCAL_DATA_DIR/$category" \
                --prefix "$category/" \
                --recursive
        fi
    done
}

check_status() {
    log "Checking bucket status..."
    
    ovhai bucket list "$REGION" | grep -E "NAME|$BUCKET_NAME" || echo "Bucket not found"
    
    log ""
    log "Bucket contents:"
    ovhai bucket object list "$BUCKET_NAME@$REGION" 2>/dev/null | head -30 || echo "Cannot list objects"
}

show_usage() {
    echo "OVH Object Storage Upload Script"
    echo ""
    echo "Usage: $0 {create|upload|upload-remote|status|all}"
    echo ""
    echo "Commands:"
    echo "  create        - Create the S3 bucket"
    echo "  upload        - Upload datasets from local"
    echo "  upload-remote - Upload datasets from remote server"
    echo "  status        - Check bucket and upload status"
    echo "  all           - Create bucket and upload"
    echo ""
    echo "Prerequisites:"
    echo "  1. Run 'ovhai login' to authenticate"
    echo "  2. Ensure datasets are in ~/datasets/consolidated/"
}

# Main
case "${1:-}" in
    create)
        check_auth
        create_bucket
        ;;
    upload)
        check_auth
        upload_local
        ;;
    upload-remote)
        check_auth
        upload_from_remote
        ;;
    status)
        check_auth
        check_status
        ;;
    all)
        check_auth
        create_bucket
        upload_local
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

log "Done!"

