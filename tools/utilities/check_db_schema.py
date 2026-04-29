#!/usr/bin/env python3
"""Check database schema and sample data"""

import logging
import sqlite3
from pathlib import Path


def get_db_connection(db_path):
    """Helper to get a database connection and cursor."""
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()

def check_database():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("db_schema_checker")

    db_path = Path("database/conversations.db")

    if not db_path.exists():
        logger.error(f"❌ Database not found at: {db_path}")
        return

    try:
        log_conversations_summary(db_path, logger)
    except Exception as e:
        logger.error(f"❌ Error checking database: {e}")


def log_conversations_summary(db_path, logger):
    conn, cursor = get_db_connection(db_path)

    # Get table schema
    cursor.execute("PRAGMA table_info(conversations)")
    columns = cursor.fetchall()

    logger.info("📊 Database Schema:")
    logger.info("=" * 50)
    for col in columns:
        logger.info(f"  {col[1]} ({col[2]})")

    # Get sample data
    cursor.execute("SELECT * FROM conversations LIMIT 1")
    if sample := cursor.fetchone():
        logger.info("\n📋 Sample Record:")
        logger.info("=" * 50)
        for i, col in enumerate(columns):
            value = sample[i] if i < len(sample) else "NULL"
            logger.info(f"  {col[1]}: {value}")

    # Get count
    cursor.execute("SELECT COUNT(*) FROM conversations")
    count = cursor.fetchone()[0]
    logger.info(f"\n📈 Total Records: {count:,}")

    conn.close()

if __name__ == "__main__":
    check_database()
