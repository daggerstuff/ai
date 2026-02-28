# Automated PostgreSQL Backup and Point-in-Time Recovery (PITR)

This directory contains scripts to set up an automated backup system for PostgreSQL with Point-in-Time Recovery capabilities. This is intended to fulfill task `1A.1.3`.

## Components

- `backup.sh`: Performs periodic base backups and cleans up old backups and WAL files.
- `restore.sh`: Restores the database from a backup to a specific point in time.
- `archive_wal.sh`: A script to be used by PostgreSQL's `archive_command` to archive WAL files.

## Setup Instructions

### 1. PostgreSQL Configuration for WAL Archiving

To enable PITR, you must configure your PostgreSQL server for continuous WAL archiving. Add or modify these lines in your `postgresql.conf`:

```ini
wal_level = replica
archive_mode = on
archive_command = '/path/to/backup/scripts/archive_wal.sh %p %f' # IMPORTANT: use absolute path
archive_timeout = 60s
