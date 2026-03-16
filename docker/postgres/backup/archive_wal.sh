#!/bin/bash
set -euo pipefail

# This script is intended to be used as the `archive_command` in postgresql.conf
# e.g., archive_command = '/path/to/this/archive_wal.sh %p %f'

# %p is the path of the file to archive
# %f is the filename only
WAL_FILE_PATH=$1
WAL_FILE_NAME=$2

WAL_ARCHIVE_DIR="${WAL_ARCHIVE_DIR:-/backups/postgres/wal}"

# Ensure the archive directory exists
mkdir -p "${WAL_ARCHIVE_DIR}"

echo "Archiving WAL file: ${WAL_FILE_NAME} to ${WAL_ARCHIVE_DIR}"
cp "${WAL_FILE_PATH}" "${WAL_ARCHIVE_DIR}/${WAL_FILE_NAME}"
