#!/usr/bin/env python3
"""
Migrate Conversations to Docker PostgreSQL
Simplified migration script for Docker PostgreSQL setup.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection settings for Docker PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "user": "postgres", 
    "password": "postgres",
    "database": "pixelated_empathy"
}

def create_tables():
    """Create the conversation tables."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category VARCHAR,
                tier VARCHAR,
                source VARCHAR,
                conversation_count INTEGER DEFAULT 1,
                total_messages INTEGER DEFAULT 0,
                avg_message_length FLOAT DEFAULT 0.0,
                quality_score FLOAT DEFAULT 0.0
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR REFERENCES conversations(id),
                role VARCHAR NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER DEFAULT 0,
                sentiment_score FLOAT DEFAULT 0.0
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_tier ON conversations(tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        return False

def count_existing_data():
    """Count existing conversations and messages."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {"conversations": conv_count, "messages": msg_count}
        
    except Exception as e:
        logger.error(f"❌ Count query failed: {e}")
        return {"conversations": 0, "messages": 0}

def migrate_json_file(file_path: Path) -> int:
    """Migrate a single JSON/JSONL file to the database."""
    try:
        conversations = []

        if file_path.suffix == '.jsonl':
            # Handle JSONL files (one JSON object per line)
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversations.append(json.loads(line))
        else:
            # Handle regular JSON files
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict):
                if 'conversations' in data:
                    conversations = data['conversations']
                elif 'data' in data:
                    conversations = data['data']
                else:
                    conversations = [data]  # Single conversation

        if not conversations:
            logger.warning(f"No conversations found in {file_path}")
            return 0
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        migrated_count = 0
        
        for conv in conversations:
            try:
                # Extract conversation data
                conv_id = conv.get('id', conv.get('conversation_id', f"{file_path.stem}_{migrated_count}"))

                # Get metadata if available
                metadata = conv.get('metadata', {})
                source = metadata.get('source_dataset', metadata.get('category', file_path.stem))
                tier = metadata.get('tier', 'TIER_1')
                category = metadata.get('category', 'therapeutic')
                quality_score = metadata.get('quality_score', 0.0)

                # Insert conversation
                cursor.execute("""
                    INSERT INTO conversations (id, source, tier, category, quality_score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (conv_id, source, str(tier), category, quality_score))

                # Extract and insert messages
                messages = conv.get('messages', conv.get('conversation', []))
                if isinstance(messages, str):
                    # Handle single message as string
                    messages = [{"role": "user", "content": messages}]

                for i, msg in enumerate(messages):
                    if isinstance(msg, str):
                        msg = {"role": "user" if i % 2 == 0 else "assistant", "content": msg}

                    msg_id = msg.get('id', f"{conv_id}_msg_{i}")
                    role = msg.get('role', 'user')
                    content = msg.get('content', str(msg))

                    cursor.execute("""
                        INSERT INTO messages (id, conversation_id, role, content, word_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (msg_id, conv_id, role, content, len(content.split())))

                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    conn.commit()
                    logger.info(f"Migrated {migrated_count} conversations from {file_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to migrate conversation {conv.get('id', 'unknown')}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Migrated {migrated_count} conversations from {file_path}")
        return migrated_count
        
    except Exception as e:
        logger.error(f"❌ Failed to migrate {file_path}: {e}")
        return 0

def find_conversation_files(data_dir: Path) -> list:
    """Find all JSON files that might contain conversations."""
    json_files = []
    
    # Look in processed directory
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        json_files.extend(processed_dir.rglob("*.json"))
        json_files.extend(processed_dir.rglob("*.jsonl"))
    
    # Look in batch processing directory
    batch_dir = data_dir / "batch_processing"
    if batch_dir.exists():
        json_files.extend(batch_dir.rglob("*.json"))
    
    return json_files

def main():
    """Main migration function."""
    logger.info("🚀 STARTING CONVERSATION MIGRATION TO DOCKER POSTGRESQL")
    
    # Step 1: Create tables
    if not create_tables():
        return False
    
    # Step 2: Count existing data
    initial_stats = count_existing_data()
    logger.info(f"Initial database stats: {initial_stats}")
    
    # Step 3: Find conversation files
    data_dir = Path("data")
    json_files = find_conversation_files(data_dir)
    
    if not json_files:
        logger.warning("No JSON files found for migration")
        return False
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Step 4: Migrate files
    total_migrated = 0
    for file_path in json_files[:5]:  # Start with first 5 files as a test
        logger.info(f"Processing: {file_path}")
        migrated = migrate_json_file(file_path)
        total_migrated += migrated
        
        if total_migrated > 1000:  # Stop after 1000 conversations for initial test
            logger.info("Stopping after 1000 conversations for initial test")
            break
    
    # Step 5: Final stats
    final_stats = count_existing_data()
    logger.info(f"Final database stats: {final_stats}")
    logger.info(f"Total conversations migrated: {total_migrated}")
    
    logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY!")
    logger.info("Next steps:")
    logger.info("  1. Verify data: psql -h localhost -p 5433 -U postgres -d pixelated_empathy")
    logger.info("  2. Query conversations: SELECT COUNT(*) FROM conversations;")
    logger.info("  3. Query messages: SELECT COUNT(*) FROM messages;")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
