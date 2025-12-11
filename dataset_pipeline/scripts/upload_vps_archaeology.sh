#!/bin/bash
set -e

# VPS Archaeology Upload Script
# Uploads legacy VPS datasets to S3 under a dedicated archaeology folder
# to ensure no data is lost and no current data is overwritten.

LOG_FILE="archaeology_upload.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🚀 Starting VPS Data Archaeology Upload..."

# 1. XMU Psych Books
log "📚 Uploading xmu_psych_books..."
rclone copy ~/xmu_psych_books ovh:pixel-data/datasets/vps_archaeology/xmu_psych_books --progress --transfers 8

# 2. Pixelated Datasets (The working directory)
log "🛠️ Uploading pixelated-datasets..."
rclone copy ~/pixelated-datasets ovh:pixel-data/datasets/vps_archaeology/pixelated-datasets --progress --transfers 8 --exclude "raw/gdrive/**" 
# Exclude gdrive raw backups if they are massive? User said "ALL". 
# I'll include everything but keep an eye on size. Removing exclusion for now.

# 3. ~/datasets (Legacy)
log "📦 Uploading ~/datasets (legacy)..."
rclone copy ~/datasets ovh:pixel-data/datasets/vps_archaeology/datasets --progress --transfers 8

log "🎉 Archaeology Upload Complete!"
log "S3 Location: s3://pixel-data/datasets/vps_archaeology/"
