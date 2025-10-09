#!/usr/bin/env python3
"""
Database Schema Design - Task 5.4.3.1

Designs comprehensive database schema for conversation storage and management:
- Conversation storage with metadata
- Quality metrics and tracking
- Dataset organization and relationships
- Indexing for performance
- Scalable architecture for 2.59M+ conversations
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

# Enterprise imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))
from enterprise_config import get_config
from enterprise_logging import get_logger

class ConversationTier(Enum):
    """Conversation quality tiers."""
    PRIORITY_1 = "priority_1"
    PRIORITY_2 = "priority_2"
    PRIORITY_3 = "priority_3"
    PROFESSIONAL = "professional"
    COT_REASONING = "cot_reasoning"
    REDDIT = "reddit"
    RESEARCH = "research"

class ProcessingStatus(Enum):
    """Processing status for conversations."""
    RAW = "raw"
    PROCESSED = "processed"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class ConversationSchema:
    """Schema definition for conversation storage."""
    
    # Core conversation data
    conversation_id: str
    dataset_source: str
    tier: ConversationTier
    
    # Conversation content
    conversations: List[Dict[str, str]]  # List of human/assistant exchanges
    title: Optional[str] = None
    summary: Optional[str] = None
    
    # Quality metrics
    overall_quality: float = 0.0
    therapeutic_accuracy: float = 0.0
    clinical_compliance: float = 0.0
    safety_score: float = 0.0
    conversation_coherence: float = 0.0
    emotional_authenticity: float = 0.0
    
    # Processing metadata
    processing_status: ProcessingStatus = ProcessingStatus.RAW
    processed_at: Optional[datetime] = None
    processed_by: Optional[str] = None
    processing_version: str = "5.4.3"
    
    # Dataset metadata
    original_filename: Optional[str] = None
    file_index: Optional[int] = None
    batch_id: Optional[str] = None
    
    # Content analysis
    turn_count: int = 0
    word_count: int = 0
    character_count: int = 0
    language: str = "en"
    
    # Classification tags
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    therapeutic_techniques: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Version control
    version: int = 1
    previous_version_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class DatabaseSchemaDesigner:
    """Designs and manages database schema for conversation storage."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("database_schema")
        
        # Schema definitions
        self.tables = {}
        self.indexes = {}
        self.constraints = {}
        
        self._design_schema()
        self.logger.info("Database schema designer initialized")
    
    def _design_schema(self):
        """Design comprehensive database schema."""
        
        # Main conversations table
        self.tables['conversations'] = {
            'columns': {
                'conversation_id': 'VARCHAR(36) PRIMARY KEY',
                'dataset_source': 'VARCHAR(100) NOT NULL',
                'tier': 'VARCHAR(20) NOT NULL',
                'title': 'TEXT',
                'summary': 'TEXT',
                'conversations_json': 'LONGTEXT NOT NULL',  # JSON array of exchanges
                'turn_count': 'INTEGER DEFAULT 0',
                'word_count': 'INTEGER DEFAULT 0',
                'character_count': 'INTEGER DEFAULT 0',
                'language': 'VARCHAR(10) DEFAULT "en"',
                'processing_status': 'VARCHAR(20) DEFAULT "raw"',
                'processed_at': 'TIMESTAMP NULL',
                'processed_by': 'VARCHAR(100)',
                'processing_version': 'VARCHAR(20)',
                'original_filename': 'VARCHAR(255)',
                'file_index': 'INTEGER',
                'batch_id': 'VARCHAR(36)',
                'version': 'INTEGER DEFAULT 1',
                'previous_version_id': 'VARCHAR(36)',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP',
                'metadata_json': 'JSON'
            },
            'description': 'Main table for storing conversation data'
        }
        
        # Quality metrics table
        self.tables['conversation_quality'] = {
            'columns': {
                'quality_id': 'VARCHAR(36) PRIMARY KEY',
                'conversation_id': 'VARCHAR(36) NOT NULL',
                'overall_quality': 'DECIMAL(5,4) DEFAULT 0.0000',
                'therapeutic_accuracy': 'DECIMAL(5,4) DEFAULT 0.0000',
                'clinical_compliance': 'DECIMAL(5,4) DEFAULT 0.0000',
                'safety_score': 'DECIMAL(5,4) DEFAULT 0.0000',
                'conversation_coherence': 'DECIMAL(5,4) DEFAULT 0.0000',
                'emotional_authenticity': 'DECIMAL(5,4) DEFAULT 0.0000',
                'validation_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'validator_version': 'VARCHAR(20)',
                'validation_notes': 'TEXT',
                'quality_metadata_json': 'JSON'
            },
            'description': 'Quality metrics and validation results'
        }
        
        # Conversation tags table (many-to-many)
        self.tables['conversation_tags'] = {
            'columns': {
                'tag_id': 'VARCHAR(36) PRIMARY KEY',
                'conversation_id': 'VARCHAR(36) NOT NULL',
                'tag_type': 'VARCHAR(50) NOT NULL',  # 'tag', 'category', 'technique'
                'tag_value': 'VARCHAR(100) NOT NULL',
                'confidence': 'DECIMAL(5,4) DEFAULT 1.0000',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'description': 'Tags, categories, and therapeutic techniques'
        }
        
        # Dataset sources table
        self.tables['dataset_sources'] = {
            'columns': {
                'source_id': 'VARCHAR(36) PRIMARY KEY',
                'source_name': 'VARCHAR(100) UNIQUE NOT NULL',
                'source_type': 'VARCHAR(50) NOT NULL',
                'tier': 'VARCHAR(20) NOT NULL',
                'description': 'TEXT',
                'total_conversations': 'INTEGER DEFAULT 0',
                'processed_conversations': 'INTEGER DEFAULT 0',
                'average_quality': 'DECIMAL(5,4) DEFAULT 0.0000',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP',
                'source_metadata_json': 'JSON'
            },
            'description': 'Dataset source information and statistics'
        }
        
        # Processing batches table
        self.tables['processing_batches'] = {
            'columns': {
                'batch_id': 'VARCHAR(36) PRIMARY KEY',
                'batch_name': 'VARCHAR(100) NOT NULL',
                'batch_type': 'VARCHAR(50) NOT NULL',
                'source_dataset': 'VARCHAR(100)',
                'total_conversations': 'INTEGER DEFAULT 0',
                'processed_conversations': 'INTEGER DEFAULT 0',
                'failed_conversations': 'INTEGER DEFAULT 0',
                'processing_status': 'VARCHAR(20) DEFAULT "pending"',
                'started_at': 'TIMESTAMP NULL',
                'completed_at': 'TIMESTAMP NULL',
                'processing_time_seconds': 'INTEGER DEFAULT 0',
                'batch_metadata_json': 'JSON'
            },
            'description': 'Processing batch tracking and statistics'
        }
        
        # Conversation versions table (for version control)
        self.tables['conversation_versions'] = {
            'columns': {
                'version_id': 'VARCHAR(36) PRIMARY KEY',
                'conversation_id': 'VARCHAR(36) NOT NULL',
                'version_number': 'INTEGER NOT NULL',
                'change_type': 'VARCHAR(50) NOT NULL',  # 'created', 'updated', 'quality_updated'
                'change_description': 'TEXT',
                'changed_by': 'VARCHAR(100)',
                'changed_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'previous_data_json': 'LONGTEXT',  # Snapshot of previous state
                'change_metadata_json': 'JSON'
            },
            'description': 'Version control and change tracking'
        }
        
        # Search index table (for full-text search)
        self.tables['conversation_search'] = {
            'columns': {
                'search_id': 'VARCHAR(36) PRIMARY KEY',
                'conversation_id': 'VARCHAR(36) NOT NULL',
                'searchable_content': 'LONGTEXT NOT NULL',
                'content_vector': 'JSON',  # For semantic search
                'last_indexed': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'description': 'Full-text search index and semantic vectors'
        }
        
        # Statistics and analytics table
        self.tables['conversation_statistics'] = {
            'columns': {
                'stat_id': 'VARCHAR(36) PRIMARY KEY',
                'stat_type': 'VARCHAR(50) NOT NULL',  # 'daily', 'weekly', 'monthly', 'dataset'
                'stat_date': 'DATE NOT NULL',
                'dataset_source': 'VARCHAR(100)',
                'tier': 'VARCHAR(20)',
                'total_conversations': 'INTEGER DEFAULT 0',
                'average_quality': 'DECIMAL(5,4) DEFAULT 0.0000',
                'average_word_count': 'DECIMAL(8,2) DEFAULT 0.00',
                'quality_distribution_json': 'JSON',
                'statistics_json': 'JSON',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'description': 'Aggregated statistics and analytics'
        }
        
        # Define indexes for performance
        self._design_indexes()
        
        # Define foreign key constraints
        self._design_constraints()
    
    def _design_indexes(self):
        """Design database indexes for optimal performance."""
        
        # Primary indexes for conversations table
        self.indexes['conversations'] = [
            'CREATE INDEX idx_conversations_dataset_source ON conversations(dataset_source)',
            'CREATE INDEX idx_conversations_tier ON conversations(tier)',
            'CREATE INDEX idx_conversations_processing_status ON conversations(processing_status)',
            'CREATE INDEX idx_conversations_created_at ON conversations(created_at)',
            'CREATE INDEX idx_conversations_updated_at ON conversations(updated_at)',
            'CREATE INDEX idx_conversations_batch_id ON conversations(batch_id)',
            'CREATE INDEX idx_conversations_quality ON conversations(dataset_source, tier)',
            'CREATE INDEX idx_conversations_filename ON conversations(original_filename)',
        ]
        
        # Quality metrics indexes
        self.indexes['conversation_quality'] = [
            'CREATE INDEX idx_quality_conversation_id ON conversation_quality(conversation_id)',
            'CREATE INDEX idx_quality_overall ON conversation_quality(overall_quality)',
            'CREATE INDEX idx_quality_therapeutic ON conversation_quality(therapeutic_accuracy)',
            'CREATE INDEX idx_quality_safety ON conversation_quality(safety_score)',
            'CREATE INDEX idx_quality_validation_date ON conversation_quality(validation_date)',
        ]
        
        # Tags indexes
        self.indexes['conversation_tags'] = [
            'CREATE INDEX idx_tags_conversation_id ON conversation_tags(conversation_id)',
            'CREATE INDEX idx_tags_type_value ON conversation_tags(tag_type, tag_value)',
            'CREATE INDEX idx_tags_value ON conversation_tags(tag_value)',
        ]
        
        # Dataset sources indexes
        self.indexes['dataset_sources'] = [
            'CREATE INDEX idx_sources_name ON dataset_sources(source_name)',
            'CREATE INDEX idx_sources_type ON dataset_sources(source_type)',
            'CREATE INDEX idx_sources_tier ON dataset_sources(tier)',
        ]
        
        # Processing batches indexes
        self.indexes['processing_batches'] = [
            'CREATE INDEX idx_batches_status ON processing_batches(processing_status)',
            'CREATE INDEX idx_batches_started_at ON processing_batches(started_at)',
            'CREATE INDEX idx_batches_source ON processing_batches(source_dataset)',
        ]
        
        # Version control indexes
        self.indexes['conversation_versions'] = [
            'CREATE INDEX idx_versions_conversation_id ON conversation_versions(conversation_id)',
            'CREATE INDEX idx_versions_number ON conversation_versions(conversation_id, version_number)',
            'CREATE INDEX idx_versions_changed_at ON conversation_versions(changed_at)',
        ]
        
        # Search indexes
        self.indexes['conversation_search'] = [
            'CREATE INDEX idx_search_conversation_id ON conversation_search(conversation_id)',
            'CREATE FULLTEXT INDEX idx_search_content ON conversation_search(searchable_content)',
        ]
        
        # Statistics indexes
        self.indexes['conversation_statistics'] = [
            'CREATE INDEX idx_stats_type_date ON conversation_statistics(stat_type, stat_date)',
            'CREATE INDEX idx_stats_source ON conversation_statistics(dataset_source)',
            'CREATE INDEX idx_stats_tier ON conversation_statistics(tier)',
        ]
    
    def _design_constraints(self):
        """Design foreign key constraints and relationships."""
        
        self.constraints = {
            'conversation_quality': [
                'ALTER TABLE conversation_quality ADD CONSTRAINT fk_quality_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE'
            ],
            'conversation_tags': [
                'ALTER TABLE conversation_tags ADD CONSTRAINT fk_tags_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE'
            ],
            'conversation_versions': [
                'ALTER TABLE conversation_versions ADD CONSTRAINT fk_versions_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE'
            ],
            'conversation_search': [
                'ALTER TABLE conversation_search ADD CONSTRAINT fk_search_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE'
            ],
            'processing_batches': [
                'ALTER TABLE processing_batches ADD CONSTRAINT fk_batches_source FOREIGN KEY (source_dataset) REFERENCES dataset_sources(source_name) ON DELETE SET NULL'
            ]
        }
    
    def generate_sql_schema(self, database_type: str = "mysql") -> Dict[str, List[str]]:
        """Generate SQL schema for specified database type."""
        
        sql_statements = {
            'tables': [],
            'indexes': [],
            'constraints': []
        }
        
        # Generate table creation statements
        for table_name, table_def in self.tables.items():
            columns = []
            for col_name, col_def in table_def['columns'].items():
                columns.append(f"    {col_name} {col_def}")
            
            columns_str = ',\n'.join(columns)
            create_table = f"""CREATE TABLE {table_name} (
{columns_str}
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"""
            
            sql_statements['tables'].append(create_table)
        
        # Generate index statements
        for table_name, table_indexes in self.indexes.items():
            sql_statements['indexes'].extend(table_indexes)
        
        # Generate constraint statements
        for table_name, table_constraints in self.constraints.items():
            sql_statements['constraints'].extend(table_constraints)
        
        return sql_statements
    
    def get_schema_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive schema documentation."""
        
        documentation = {
            'overview': {
                'total_tables': len(self.tables),
                'total_indexes': sum(len(indexes) for indexes in self.indexes.values()),
                'total_constraints': sum(len(constraints) for constraints in self.constraints.values()),
                'designed_for': '2.59M+ conversations with enterprise-grade performance'
            },
            'tables': {},
            'relationships': {},
            'performance_considerations': {
                'indexing_strategy': 'Comprehensive indexing for query performance',
                'partitioning': 'Consider partitioning by tier or date for large datasets',
                'caching': 'Implement query result caching for frequently accessed data',
                'archiving': 'Archive old versions and statistics to maintain performance'
            }
        }
        
        # Document each table
        for table_name, table_def in self.tables.items():
            documentation['tables'][table_name] = {
                'description': table_def['description'],
                'columns': len(table_def['columns']),
                'indexes': len(self.indexes.get(table_name, [])),
                'primary_key': [col for col, def_ in table_def['columns'].items() if 'PRIMARY KEY' in def_],
                'foreign_keys': len(self.constraints.get(table_name, []))
            }
        
        # Document relationships
        documentation['relationships'] = {
            'conversations': 'Central table with one-to-many relationships to quality, tags, versions, and search',
            'conversation_quality': 'One-to-one with conversations for quality metrics',
            'conversation_tags': 'Many-to-many relationship for flexible tagging',
            'dataset_sources': 'One-to-many with conversations for source tracking',
            'processing_batches': 'One-to-many with conversations for batch processing',
            'conversation_versions': 'One-to-many with conversations for version control',
            'conversation_search': 'One-to-one with conversations for search indexing',
            'conversation_statistics': 'Aggregated data for analytics and reporting'
        }
        
        return documentation
    
    def estimate_storage_requirements(self, conversation_count: int = 2590000) -> Dict[str, Any]:
        """Estimate storage requirements for the given number of conversations."""
        
        # Average sizes (in bytes)
        avg_conversation_size = 2048  # Average conversation JSON size
        avg_metadata_size = 512      # Average metadata size
        avg_search_content_size = 1024  # Average searchable content size
        
        estimates = {
            'conversations': {
                'rows': conversation_count,
                'avg_row_size_bytes': avg_conversation_size + avg_metadata_size + 200,  # Base columns
                'estimated_size_gb': (conversation_count * (avg_conversation_size + avg_metadata_size + 200)) / (1024**3)
            },
            'conversation_quality': {
                'rows': conversation_count,
                'avg_row_size_bytes': 200,  # Quality metrics
                'estimated_size_gb': (conversation_count * 200) / (1024**3)
            },
            'conversation_tags': {
                'rows': conversation_count * 5,  # Average 5 tags per conversation
                'avg_row_size_bytes': 150,
                'estimated_size_gb': (conversation_count * 5 * 150) / (1024**3)
            },
            'conversation_search': {
                'rows': conversation_count,
                'avg_row_size_bytes': avg_search_content_size + 100,
                'estimated_size_gb': (conversation_count * (avg_search_content_size + 100)) / (1024**3)
            },
            'other_tables': {
                'estimated_size_gb': 0.5  # Dataset sources, batches, versions, statistics
            }
        }
        
        # Calculate totals
        total_data_size = sum(table['estimated_size_gb'] for table in estimates.values() if isinstance(table, dict) and 'estimated_size_gb' in table)
        index_overhead = total_data_size * 0.3  # Estimate 30% overhead for indexes
        total_size = total_data_size + index_overhead
        
        return {
            'conversation_count': conversation_count,
            'table_estimates': estimates,
            'summary': {
                'total_data_size_gb': round(total_data_size, 2),
                'index_overhead_gb': round(index_overhead, 2),
                'total_estimated_size_gb': round(total_size, 2),
                'recommended_storage_gb': round(total_size * 1.5, 2)  # 50% buffer
            }
        }

if __name__ == "__main__":
    # Test the database schema designer
    designer = DatabaseSchemaDesigner()
    
    print("🗄️ DATABASE SCHEMA DESIGN")
    print("=" * 50)
    
    # Generate SQL schema
    sql_schema = designer.generate_sql_schema()
    print(f"✅ Generated schema: {len(sql_schema['tables'])} tables, {len(sql_schema['indexes'])} indexes")
    
    # Get documentation
    docs = designer.get_schema_documentation()
    print(f"📚 Schema documentation: {docs['overview']['total_tables']} tables documented")
    
    # Estimate storage requirements
    storage = designer.estimate_storage_requirements()
    print(f"💾 Storage estimate: {storage['summary']['total_estimated_size_gb']} GB for 2.59M conversations")
    
    # Save schema to file
    schema_file = Path("/home/vivi/pixelated/ai/database/conversation_schema.sql")
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(schema_file, 'w') as f:
        f.write("-- Pixelated AI Conversation Database Schema\n")
        f.write("-- Generated for enterprise-grade conversation storage\n\n")
        
        f.write("-- Table Creation\n")
        for statement in sql_schema['tables']:
            f.write(statement + "\n\n")
        
        f.write("-- Index Creation\n")
        for statement in sql_schema['indexes']:
            f.write(statement + ";\n")
        
        f.write("\n-- Constraint Creation\n")
        for statement in sql_schema['constraints']:
            f.write(statement + ";\n")
    
    print(f"💾 Schema saved to: {schema_file}")
    print("✅ Database schema design complete!")
