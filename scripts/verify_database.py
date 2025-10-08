#!/usr/bin/env python3
"""
Verify Database Migration
Check the migrated data in PostgreSQL.
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "user": "postgres",
    "password": "postgres",
    "database": "pixelated_empathy"
}

def verify_migration():
    """Verify the migration results."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Count conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        logger.info(f"✅ Total conversations: {conv_count}")
        
        # Count messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        logger.info(f"✅ Total messages: {msg_count}")
        
        # Show conversation sources
        cursor.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source")
        sources = cursor.fetchall()
        logger.info("📊 Conversations by source:")
        for source, count in sources:
            logger.info(f"  - {source}: {count}")
        
        # Show sample conversation
        cursor.execute("""
            SELECT c.id, c.source, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id, c.source
            LIMIT 3
        """)
        samples = cursor.fetchall()
        logger.info("📝 Sample conversations:")
        for conv_id, source, msg_count in samples:
            logger.info(f"  - {conv_id} ({source}): {msg_count} messages")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_migration()
