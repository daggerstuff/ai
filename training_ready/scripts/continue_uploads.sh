#!/usr/bin/env zsh
# Continue uploading local datasets to S3
# This script can be run multiple times to resume uploads

set -e

cd /home/vivi/pixelated

# Load environment
set -a
source ai/.env
set +a

# Continue uploads, skipping already uploaded files
echo "🚀 Continuing local dataset uploads to S3..."
echo ""

uv run python3 ai/training_ready/scripts/upload_local_datasets_to_s3.py \
    --skip-existing \
    --resume ai/training_ready/scripts/output/local_upload_results.json

echo ""
echo "📊 Current status:"
uv run python3 ai/training_ready/scripts/monitor_upload_progress.py
