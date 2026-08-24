#!/bin/bash
set -euo pipefail

# --- Configuration ---
BACKUP_ROOT_DIR="${BACKUP_DIR:-/backups/postgres}"
BASE_BACKUP_DIR="${BACKUP_ROOT_DIR}/base"
WAL_ARCHIVE_DIR="${BACKUP_ROOT_DIR}/wal"
BASE_BACKUP_RETENTION_DAYS="${BASE_BACKUP_RETENTION_DAYS:-7}"
# WARNING: Time-based WAL cleanup is not foolproof. A safer method uses pg_archivecleanup.
# This is a simpler "good enough" approach. Make sure this retention is longer than base backup retention.
WAL_ARCHIVE_RETENTION_DAYS="${WAL_ARCHIVE_RETENTION_DAYS:-8}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

# --- Create directories ---
mkdir -p "${BASE_BACKUP_DIR}"
mkdir -p "${WAL_ARCHIVE_DIR}"

# --- Create base backup ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="base_${TIMESTAMP}"
BACKUP_PATH_TEMP="${BASE_BACKUP_DIR}/${BACKUP_NAME}"

echo "Starting base backup to ${BACKUP_PATH_TEMP}..."

# Assumes .pgpass is configured for password-less auth, hence the -w flag.
pg_basebackup \
    -h "${POSTGRES_HOST}" \
    -p "${POSTGRES_PORT}" \
    -U "${POSTGRES_USER}" \
    -D "${BACKUP_PATH_TEMP}" \
    -F p \
    -X stream \
    -P \
    -v \
    -w

echo "Base backup successful."
echo "Compressing backup..."
tar -czvf "${BACKUP_PATH_TEMP}.tar.gz" -C "${BASE_BACKUP_DIR}" "${BACKUP_NAME}"
rm -rf "${BACKUP_PATH_TEMP}"
echo "Backup compressed to ${BACKUP_PATH_TEMP}.tar.gz"

# --- Cleanup old backups ---
echo "Cleaning up old base backups (older than ${BASE_BACKUP_RETENTION_DAYS} days)..."
find "${BASE_BACKUP_DIR}" -type f -name "base_*.tar.gz" -mtime "+${BASE_BACKUP_RETENTION_DAYS}" -print -delete

echo "Cleaning up old WALs (older than ${WAL_ARCHIVE_RETENTION_DAYS} days)..."
find "${WAL_ARCHIVE_DIR}" -type f -mtime "+${WAL_ARCHIVE_RETENTION_DAYS}" -print -delete

echo "Automated backup finished successfully."
