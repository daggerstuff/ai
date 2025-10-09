#!/usr/bin/env python3
"""
Migrate Only Conversation JSONL Files
Focus on actual conversation data, not reports.
"""

import os
import sys
import json
import logging
from pathlib import Path
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "user": "postgres",
    "password": "postgres",
    "database": "pixelated_empathy"
}

def migrate_jsonl_file(file_path: Path) -> int:
    """Migrate a JSONL conversation file."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        migrated_count = 0
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    conv_data = json.loads(line)
                    
                    # Generate conversation ID
                    conv_id = f"{file_path.stem}_{line_num}"
                    
                    # Get metadata
                    metadata = conv_data.get('metadata', {})
                    source = metadata.get('source_dataset', file_path.stem)
                    tier = metadata.get('tier', 1)
                    category = metadata.get('category', 'therapeutic')
                    quality_score = metadata.get('quality_score', 0.0)
                    
                    # Insert conversation
                    cursor.execute("""
                        INSERT INTO conversations (id, source, tier, category, quality_score)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (conv_id, source, f"TIER_{tier}", category, quality_score))
                    
                    # Get conversation messages
                    messages = conv_data.get('conversation', [])
                    
                    for i, msg in enumerate(messages):
                        msg_id = f"{conv_id}_msg_{i}"
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        
                        cursor.execute("""
                            INSERT INTO messages (id, conversation_id, role, content, word_count)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, (msg_id, conv_id, role, content, len(content.split())))
                    
                    migrated_count += 1
                    
                    if migrated_count % 100 == 0:
                        conn.commit()
                        logger.info(f"Migrated {migrated_count} conversations from {file_path.name}")
                    
                    if migrated_count >= 500:  # Limit for testing
                        logger.info(f"Stopping at 500 conversations for {file_path.name}")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                    continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Migrated {migrated_count} conversations from {file_path}")
        return migrated_count
        
    except Exception as e:
        logger.error(f"❌ Failed to migrate {file_path}: {e}")
        return 0

def find_conversation_jsonl_files() -> list:
    """Find JSONL files with actual conversations."""
    jsonl_files = []
    
    # Look for conversation JSONL files
    data_dir = Path("data/processed")
    if data_dir.exists():
        for jsonl_file in data_dir.rglob("*conversations.jsonl"):
            jsonl_files.append(jsonl_file)
    
    return jsonl_files

def main():
    """Main migration function."""
    logger.info("🚀 MIGRATING CONVERSATION JSONL FILES TO POSTGRESQL")
    
    # Find conversation files
    jsonl_files = find_conversation_jsonl_files()
    
    if not jsonl_files:
        logger.error("No conversation JSONL files found!")
        return False
    
    logger.info(f"Found {len(jsonl_files)} conversation JSONL files:")
    for f in jsonl_files:
        logger.info(f"  - {f}")
    
    # Migrate each file
    total_migrated = 0
    for file_path in jsonl_files:
        logger.info(f"\n📁 Processing: {file_path}")
        migrated = migrate_jsonl_file(file_path)
        total_migrated += migrated
    
    logger.info(f"\n✅ MIGRATION COMPLETED!")
    logger.info(f"Total conversations migrated: {total_migrated}")
    
    # Verify results
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source ORDER BY COUNT(*) DESC")
        sources = cursor.fetchall()
        
        logger.info(f"\n📊 FINAL STATISTICS:")
        logger.info(f"Total conversations: {conv_count}")
        logger.info(f"Total messages: {msg_count}")
        logger.info(f"Average messages per conversation: {msg_count/conv_count if conv_count > 0 else 0:.1f}")
        
        logger.info(f"\n📈 CONVERSATIONS BY SOURCE:")
        for source, count in sources:
            logger.info(f"  - {source}: {count}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
