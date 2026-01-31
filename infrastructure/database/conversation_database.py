#!/usr/bin/env python3
"""
Database Integration for Conversations - Task 5.4.3.2

Implements database integration for processed conversations:
- Connection management and pooling
- CRUD operations for conversations
- Batch insertion and updates
- Transaction management
- Performance optimization
"""

import json
import uuid
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from contextlib import contextmanager
import logging

# Enterprise imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))
from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_error_handling import handle_error, with_retry

# Schema imports
from conversation_schema import ConversationSchema, ConversationTier, ProcessingStatus

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    database_path: str
    connection_pool_size: int = 10
    timeout_seconds: int = 30
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    cache_size_mb: int = 64

class ConversationDatabase:
    """Database integration for conversation storage and management."""
    
    def __init__(self, db_config: DatabaseConfig = None):
        self.config = get_config()
        self.logger = get_logger("conversation_database")
        
        # Database configuration
        self.db_config = db_config or DatabaseConfig(
            database_path="/home/vivi/pixelated/ai/database/conversations.db",
            connection_pool_size=10,
            timeout_seconds=30
        )
        
        # Connection management
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self._local = threading.local()
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"Conversation database initialized: {self.db_config.database_path}")
    
    def _initialize_database(self):
        """Initialize database with schema and optimizations."""
        
        # Ensure database directory exists
        Path(self.db_config.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create initial connection and schema
        with self._get_connection() as conn:
            # Enable optimizations
            if self.db_config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            
            if self.db_config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")
            
            conn.execute(f"PRAGMA cache_size=-{self.db_config.cache_size_mb * 1024}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # Create tables
            self._create_tables(conn)
            
            # Create indexes
            self._create_indexes(conn)
            
            conn.commit()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        
        # Main conversations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                dataset_source TEXT NOT NULL,
                tier TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                conversations_json TEXT NOT NULL,
                turn_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                character_count INTEGER DEFAULT 0,
                language TEXT DEFAULT 'en',
                processing_status TEXT DEFAULT 'raw',
                processed_at TIMESTAMP,
                processed_by TEXT,
                processing_version TEXT,
                original_filename TEXT,
                file_index INTEGER,
                batch_id TEXT,
                version INTEGER DEFAULT 1,
                previous_version_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT
            )
        """)
        
        # Quality metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_quality (
                quality_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                overall_quality REAL DEFAULT 0.0,
                therapeutic_accuracy REAL DEFAULT 0.0,
                clinical_compliance REAL DEFAULT 0.0,
                safety_score REAL DEFAULT 0.0,
                conversation_coherence REAL DEFAULT 0.0,
                emotional_authenticity REAL DEFAULT 0.0,
                validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validator_version TEXT,
                validation_notes TEXT,
                quality_metadata_json TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )
        """)
        
        # Tags table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_tags (
                tag_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                tag_type TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )
        """)
        
        # Dataset sources table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_sources (
                source_id TEXT PRIMARY KEY,
                source_name TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                description TEXT,
                total_conversations INTEGER DEFAULT 0,
                processed_conversations INTEGER DEFAULT 0,
                average_quality REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_metadata_json TEXT
            )
        """)
        
        # Processing batches table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_batches (
                batch_id TEXT PRIMARY KEY,
                batch_name TEXT NOT NULL,
                batch_type TEXT NOT NULL,
                source_dataset TEXT,
                total_conversations INTEGER DEFAULT 0,
                processed_conversations INTEGER DEFAULT 0,
                failed_conversations INTEGER DEFAULT 0,
                processing_status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                processing_time_seconds INTEGER DEFAULT 0,
                batch_metadata_json TEXT
            )
        """)
        
        # Search index table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_search (
                search_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                searchable_content TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )
        """)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for performance."""
        
        indexes = [
            # Conversations indexes
            "CREATE INDEX IF NOT EXISTS idx_conversations_dataset_source ON conversations(dataset_source)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_tier ON conversations(tier)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_batch_id ON conversations(batch_id)",
            
            # Quality indexes
            "CREATE INDEX IF NOT EXISTS idx_quality_conversation_id ON conversation_quality(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_quality_overall ON conversation_quality(overall_quality)",
            "CREATE INDEX IF NOT EXISTS idx_quality_therapeutic ON conversation_quality(therapeutic_accuracy)",
            
            # Tags indexes
            "CREATE INDEX IF NOT EXISTS idx_tags_conversation_id ON conversation_tags(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_tags_type_value ON conversation_tags(tag_type, tag_value)",
            
            # Search index
            "CREATE INDEX IF NOT EXISTS idx_search_conversation_id ON conversation_search(conversation_id)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        
        conn = None
        try:
            # Try to get connection from pool
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = sqlite3.connect(
                        self.db_config.database_path,
                        timeout=self.db_config.timeout_seconds,
                        check_same_thread=False
                    )
                    conn.row_factory = sqlite3.Row
            
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                # Return connection to pool
                with self.pool_lock:
                    if len(self.connection_pool) < self.db_config.connection_pool_size:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
    
    @with_retry(component="conversation_database")
    def insert_conversation(self, conversation: ConversationSchema) -> bool:
        """Insert a new conversation into the database."""
        
        try:
            with self._get_connection() as conn:
                # Insert main conversation record
                conn.execute("""
                    INSERT INTO conversations (
                        conversation_id, dataset_source, tier, title, summary,
                        conversations_json, turn_count, word_count, character_count,
                        language, processing_status, processed_at, processed_by,
                        processing_version, original_filename, file_index, batch_id,
                        version, previous_version_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation.conversation_id,
                    conversation.dataset_source,
                    conversation.tier.value,
                    conversation.title,
                    conversation.summary,
                    json.dumps(conversation.conversations),
                    conversation.turn_count,
                    conversation.word_count,
                    conversation.character_count,
                    conversation.language,
                    conversation.processing_status.value,
                    conversation.processed_at,
                    conversation.processed_by,
                    conversation.processing_version,
                    conversation.original_filename,
                    conversation.file_index,
                    conversation.batch_id,
                    conversation.version,
                    conversation.previous_version_id,
                    json.dumps(conversation.metadata)
                ))
                
                # Insert quality metrics if available
                if any([conversation.overall_quality, conversation.therapeutic_accuracy,
                       conversation.clinical_compliance, conversation.safety_score]):
                    self._insert_quality_metrics(conn, conversation)
                
                # Insert tags if available
                if conversation.tags or conversation.categories or conversation.therapeutic_techniques:
                    self._insert_tags(conn, conversation)
                
                conn.commit()
                
            self.logger.debug(f"Inserted conversation: {conversation.conversation_id}")
            return True
            
        except Exception as e:
            handle_error(e, "conversation_database", {
                "operation": "insert_conversation",
                "conversation_id": conversation.conversation_id
            })
            return False
    
    def _insert_quality_metrics(self, conn: sqlite3.Connection, conversation: ConversationSchema):
        """Insert quality metrics for a conversation."""
        
        quality_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO conversation_quality (
                quality_id, conversation_id, overall_quality, therapeutic_accuracy,
                clinical_compliance, safety_score, conversation_coherence,
                emotional_authenticity, validator_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            quality_id,
            conversation.conversation_id,
            conversation.overall_quality,
            conversation.therapeutic_accuracy,
            conversation.clinical_compliance,
            conversation.safety_score,
            conversation.conversation_coherence,
            conversation.emotional_authenticity,
            conversation.processing_version
        ))
    
    def _insert_tags(self, conn: sqlite3.Connection, conversation: ConversationSchema):
        """Insert tags for a conversation."""
        
        # Insert regular tags
        for tag in conversation.tags:
            tag_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO conversation_tags (tag_id, conversation_id, tag_type, tag_value)
                VALUES (?, ?, ?, ?)
            """, (tag_id, conversation.conversation_id, 'tag', tag))
        
        # Insert categories
        for category in conversation.categories:
            tag_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO conversation_tags (tag_id, conversation_id, tag_type, tag_value)
                VALUES (?, ?, ?, ?)
            """, (tag_id, conversation.conversation_id, 'category', category))
        
        # Insert therapeutic techniques
        for technique in conversation.therapeutic_techniques:
            tag_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO conversation_tags (tag_id, conversation_id, tag_type, tag_value)
                VALUES (?, ?, ?, ?)
            """, (tag_id, conversation.conversation_id, 'technique', technique))
    
    @with_retry(component="conversation_database")
    def batch_insert_conversations(self, conversations: List[ConversationSchema], 
                                 batch_size: int = 1000) -> Dict[str, int]:
        """Insert multiple conversations in batches."""
        
        results = {'inserted': 0, 'failed': 0}
        
        try:
            with self._get_connection() as conn:
                # Process in batches
                for i in range(0, len(conversations), batch_size):
                    batch = conversations[i:i + batch_size]
                    
                    try:
                        # Prepare batch data
                        conversation_data = []
                        quality_data = []
                        tag_data = []
                        
                        for conv in batch:
                            # Main conversation data
                            conversation_data.append((
                                conv.conversation_id, conv.dataset_source, conv.tier.value,
                                conv.title, conv.summary, json.dumps(conv.conversations),
                                conv.turn_count, conv.word_count, conv.character_count,
                                conv.language, conv.processing_status.value, conv.processed_at,
                                conv.processed_by, conv.processing_version, conv.original_filename,
                                conv.file_index, conv.batch_id, conv.version,
                                conv.previous_version_id, json.dumps(conv.metadata)
                            ))
                            
                            # Quality data
                            if any([conv.overall_quality, conv.therapeutic_accuracy]):
                                quality_data.append((
                                    str(uuid.uuid4()), conv.conversation_id,
                                    conv.overall_quality, conv.therapeutic_accuracy,
                                    conv.clinical_compliance, conv.safety_score,
                                    conv.conversation_coherence, conv.emotional_authenticity,
                                    conv.processing_version
                                ))
                            
                            # Tag data
                            for tag in conv.tags:
                                tag_data.append((str(uuid.uuid4()), conv.conversation_id, 'tag', tag))
                            for category in conv.categories:
                                tag_data.append((str(uuid.uuid4()), conv.conversation_id, 'category', category))
                            for technique in conv.therapeutic_techniques:
                                tag_data.append((str(uuid.uuid4()), conv.conversation_id, 'technique', technique))
                        
                        # Execute batch inserts
                        conn.executemany("""
                            INSERT INTO conversations (
                                conversation_id, dataset_source, tier, title, summary,
                                conversations_json, turn_count, word_count, character_count,
                                language, processing_status, processed_at, processed_by,
                                processing_version, original_filename, file_index, batch_id,
                                version, previous_version_id, metadata_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, conversation_data)
                        
                        if quality_data:
                            conn.executemany("""
                                INSERT INTO conversation_quality (
                                    quality_id, conversation_id, overall_quality, therapeutic_accuracy,
                                    clinical_compliance, safety_score, conversation_coherence,
                                    emotional_authenticity, validator_version
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, quality_data)
                        
                        if tag_data:
                            conn.executemany("""
                                INSERT INTO conversation_tags (tag_id, conversation_id, tag_type, tag_value)
                                VALUES (?, ?, ?, ?)
                            """, tag_data)
                        
                        conn.commit()
                        results['inserted'] += len(batch)
                        
                        self.logger.debug(f"Batch inserted: {len(batch)} conversations")
                        
                    except Exception as e:
                        conn.rollback()
                        results['failed'] += len(batch)
                        handle_error(e, "conversation_database", {
                            "operation": "batch_insert",
                            "batch_size": len(batch)
                        })
            
            self.logger.info(f"Batch insert complete: {results['inserted']} inserted, {results['failed']} failed")
            return results
            
        except Exception as e:
            handle_error(e, "conversation_database", {"operation": "batch_insert_conversations"})
            return results
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation by ID."""
        
        try:
            with self._get_connection() as conn:
                # Get main conversation data
                cursor = conn.execute("""
                    SELECT * FROM conversations WHERE conversation_id = ?
                """, (conversation_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                conversation = dict(row)
                
                # Parse JSON fields
                conversation['conversations'] = json.loads(conversation['conversations_json'])
                conversation['metadata'] = json.loads(conversation['metadata_json'] or '{}')
                
                # Get quality metrics
                cursor = conn.execute("""
                    SELECT * FROM conversation_quality WHERE conversation_id = ?
                """, (conversation_id,))
                
                quality_row = cursor.fetchone()
                if quality_row:
                    conversation['quality_metrics'] = dict(quality_row)
                
                # Get tags
                cursor = conn.execute("""
                    SELECT tag_type, tag_value FROM conversation_tags WHERE conversation_id = ?
                """, (conversation_id,))
                
                tags = {'tags': [], 'categories': [], 'techniques': []}
                for tag_row in cursor.fetchall():
                    tag_type, tag_value = tag_row
                    if tag_type == 'tag':
                        tags['tags'].append(tag_value)
                    elif tag_type == 'category':
                        tags['categories'].append(tag_value)
                    elif tag_type == 'technique':
                        tags['techniques'].append(tag_value)
                
                conversation.update(tags)
                
                return conversation
                
        except Exception as e:
            handle_error(e, "conversation_database", {
                "operation": "get_conversation",
                "conversation_id": conversation_id
            })
            return None
    
    def search_conversations(self, filters: Dict[str, Any] = None, 
                           limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Search conversations with filters."""
        
        try:
            with self._get_connection() as conn:
                # Build query
                where_clauses = []
                params = []
                
                if filters:
                    if 'dataset_source' in filters:
                        where_clauses.append("dataset_source = ?")
                        params.append(filters['dataset_source'])
                    
                    if 'tier' in filters:
                        where_clauses.append("tier = ?")
                        params.append(filters['tier'])
                    
                    if 'processing_status' in filters:
                        where_clauses.append("processing_status = ?")
                        params.append(filters['processing_status'])
                    
                    if 'min_quality' in filters:
                        where_clauses.append("""
                            conversation_id IN (
                                SELECT conversation_id FROM conversation_quality 
                                WHERE overall_quality >= ?
                            )
                        """)
                        params.append(filters['min_quality'])
                
                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                query = f"""
                    SELECT conversation_id, dataset_source, tier, title, 
                           turn_count, word_count, processing_status, created_at
                    FROM conversations 
                    WHERE {where_sql}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                return results
                
        except Exception as e:
            handle_error(e, "conversation_database", {
                "operation": "search_conversations",
                "filters": filters
            })
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Total conversations
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                stats['total_conversations'] = cursor.fetchone()[0]
                
                # Conversations by tier
                cursor = conn.execute("""
                    SELECT tier, COUNT(*) FROM conversations GROUP BY tier
                """)
                stats['conversations_by_tier'] = dict(cursor.fetchall())
                
                # Conversations by status
                cursor = conn.execute("""
                    SELECT processing_status, COUNT(*) FROM conversations GROUP BY processing_status
                """)
                stats['conversations_by_status'] = dict(cursor.fetchall())
                
                # Average quality metrics
                cursor = conn.execute("""
                    SELECT 
                        AVG(overall_quality) as avg_overall_quality,
                        AVG(therapeutic_accuracy) as avg_therapeutic_accuracy,
                        AVG(safety_score) as avg_safety_score
                    FROM conversation_quality
                """)
                quality_row = cursor.fetchone()
                if quality_row:
                    stats['average_quality_metrics'] = {
                        'overall_quality': quality_row[0] or 0.0,
                        'therapeutic_accuracy': quality_row[1] or 0.0,
                        'safety_score': quality_row[2] or 0.0
                    }
                
                # Database size
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                stats['database_size_mb'] = (page_count * page_size) / (1024 * 1024)
                
                return stats
                
        except Exception as e:
            handle_error(e, "conversation_database", {"operation": "get_database_statistics"})
            return {}
    
    def close(self):
        """Close all database connections."""
        
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
        
        self.logger.info("Database connections closed")

if __name__ == "__main__":
    # Test the database integration
    db = ConversationDatabase()
    
    print("🗄️ DATABASE INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Create test conversation
        test_conversation = ConversationSchema(
            conversation_id=str(uuid.uuid4()),
            dataset_source="test_dataset",
            tier=ConversationTier.PRIORITY_1,
            conversations=[
                {"human": "Hello, how are you?"},
                {"assistant": "I'm doing well, thank you for asking."}
            ],
            overall_quality=0.85,
            therapeutic_accuracy=0.80,
            safety_score=0.95,
            tags=["greeting", "wellness"],
            categories=["social"],
            processing_status=ProcessingStatus.PROCESSED,
            turn_count=2,
            word_count=15
        )
        
        # Test insertion
        success = db.insert_conversation(test_conversation)
        print(f"✅ Conversation inserted: {success}")
        
        # Test retrieval
        retrieved = db.get_conversation(test_conversation.conversation_id)
        print(f"✅ Conversation retrieved: {retrieved is not None}")
        
        # Test search
        results = db.search_conversations({'tier': 'priority_1'}, limit=10)
        print(f"✅ Search results: {len(results)} conversations found")
        
        # Test statistics
        stats = db.get_database_statistics()
        print(f"✅ Database statistics: {stats.get('total_conversations', 0)} total conversations")
        print(f"   Database size: {stats.get('database_size_mb', 0):.2f} MB")
        
    finally:
        db.close()
    
    print("✅ Database integration test complete!")
