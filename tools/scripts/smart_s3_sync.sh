#!/bin/bash
# Sync S3 backup from Google Drive to DO Spaces with deduplication
# Excludes archive/ and intermediate _processed files

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REDIS_AUDIT="${PROJECT_ROOT}/scripts/check-redis-hardening.sh"

if ! "$REDIS_AUDIT"; then
  echo "Redis hardening audit failed"
  exit 1
fi

SOURCE="gdrive:backups/S3-Complete"
DEST="BackupStorageS3:pixel-data"
LOG="/tmp/rclone_smart_sync.log"

echo "========================================"
echo "SMART S3 SYNC WITH DEDUPLICATION"
echo "========================================"
echo "Source: $SOURCE"
echo "Dest: $DEST"
echo "Log: $LOG"
echo ""

# Kill any existing rclone
pkill -f "rclone sync.*S3-Complete" || true
sleep 2

# Create exclusion file
cat > /tmp/rclone_exclude.txt <<'EOF'
# Exclude archive (94 GiB of old backups)
archive/**

# Exclude intermediate processing versions (keep only final)
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed_processing.jsonl
*_processed_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed_processed.jsonl
*_processed_processed_processed_processed.jsonl
*_processed_processed_processed.jsonl
*_processed_processed.jsonl
EOF

echo "### Calculating transfer size ###"
rclone size "$SOURCE" \
  --exclude-from /tmp/rclone_exclude.txt \
  --json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"count\"]:,} files, {d[\"bytes\"]/1024**3:.1f} GiB')"

echo ""
echo "### Starting sync ###"
nohup rclone sync "$SOURCE" "$DEST" \
  --exclude-from /tmp/rclone_exclude.txt \
  --transfers 8 \
  --checkers 16 \
  --contimeout 60s \
  --timeout 300s \
  --retries 3 \
  --low-level-retries 10 \
  --stats 30s \
  --stats-one-line \
  --log-file "$LOG" \
  > /tmp/rclone_stdout.log 2>&1 &

RCLONE_PID=$!
echo "Rclone PID: $RCLONE_PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG"
echo "  tail -f /tmp/rclone_stdout.log"
echo "  ps aux | grep rclone"
