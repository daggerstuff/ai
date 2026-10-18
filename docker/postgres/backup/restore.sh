#!/bin/bash
set -euo pipefail

# --- Configuration ---
PG_DATA_DIR="${PGDATA:-/var/lib/postgresql/data}"
BACKUP_ROOT_DIR="${BACKUP_DIR:-/backups/postgres}"
BASE_BACKUP_DIR="${BACKUP_ROOT_DIR}/base"
WAL_ARCHIVE_DIR="${BACKUP_ROOT_DIR}/wal"
RECOVERY_TARGET_TIME="${RECOVERY_TARGET_TIME:-}" # Optional: ISO 8601 format, e.g. "2024-01-01 12:00:00 UTC"

# --- Stop PostgreSQL ---
# This script assumes it's running with privileges to control PostgreSQL service.
echo "Stopping PostgreSQL..."
# Use pg_ctl to stop the server if it's running
pg_ctl -D "${PG_DATA_DIR}" -o "-c config_file=/etc/postgresql/postgresql.conf" stop || echo "PostgreSQL not running or failed to stop."

# --- Find latest backup ---
LATEST_BACKUP=$(find "${BASE_BACKUP_DIR}" -type f -name "base_*.tar.gz" | sort -r | head -n 1)
if [ -z "${LATEST_BACKUP}" ]; then
    echo "No base backups found. Cannot restore."
    exit 1
fi
echo "Restoring from latest backup: ${LATEST_BACKUP}"

# --- Clean data directory and restore ---
rm -rf "${PG_DATA_DIR:?}"/*
echo "Restoring base backup..."
# Extract to parent of data dir, then move contents
BACKUP_EXTRACTED_DIR_NAME=$(tar -tzf "${LATEST_BACKUP}" | head -1 | cut -f1 -d"/")
tar -xzvf "${LATEST_BACKUP}" -C "$(dirname "${PG_DATA_DIR}")"
mv "$(dirname "${PG_DATA_DIR}")/${BACKUP_EXTRACTED_DIR_NAME}"/* "${PG_DATA_DIR}/"
rmdir "$(dirname "${PG_DATA_DIR}")/${BACKUP_EXTRACTED_DIR_NAME}"

# --- Configure for PITR ---
echo "Configuring for Point-in-Time Recovery..."
touch "${PG_DATA_DIR}/recovery.signal" # For PG12+

# Create postgresql.auto.conf with restore_command
# For PG12+, recovery settings are in postgresql.conf or postgresql.auto.conf
cat > "${PG_DATA_DIR}/postgresql.auto.conf" <<EOF
restore_command = 'cp ${WAL_ARCHIVE_DIR}/%f %p'
EOF

if [ -n "${RECOVERY_TARGET_TIME}" ]; then
    echo "Recovery target time set to: ${RECOVERY_TARGET_TIME}"
    cat >> "${PG_DATA_DIR}/postgresql.auto.conf" <<EOF
recovery_target_time = '${RECOVERY_TARGET_TIME}'
recovery_target_action = 'promote'
EOF
fi

# Ensure correct permissions
chown -R postgres:postgres "${PG_DATA_DIR}"
chmod 700 "${PG_DATA_DIR}"

# --- Start PostgreSQL ---
echo "Starting PostgreSQL for recovery..."
pg_ctl -D "${PG_DATA_DIR}" -o "-c config_file=/etc/postgresql/postgresql.conf" start

echo "Restore process initiated. Monitor PostgreSQL logs for recovery progress."
echo "Once recovery is complete, PostgreSQL will promote itself and be available for connections."
