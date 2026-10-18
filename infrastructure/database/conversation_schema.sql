-- Pixelated AI Conversation Database Schema
-- Generated for enterprise-grade conversation storage

-- Table Creation
CREATE TABLE conversations (
    conversation_id VARCHAR(36) PRIMARY KEY,
    dataset_source VARCHAR(100) NOT NULL,
    tier VARCHAR(20) NOT NULL,
    title TEXT,
    summary TEXT,
    conversations_json LONGTEXT NOT NULL,
    turn_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    character_count INTEGER DEFAULT 0,
    language VARCHAR(10) DEFAULT "en",
    processing_status VARCHAR(20) DEFAULT "raw",
    processed_at TIMESTAMP NULL,
    processed_by VARCHAR(100),
    processing_version VARCHAR(20),
    original_filename VARCHAR(255),
    file_index INTEGER,
    batch_id VARCHAR(36),
    version INTEGER DEFAULT 1,
    previous_version_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata_json JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE conversation_quality (
    quality_id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    overall_quality DECIMAL(5,4) DEFAULT 0.0000,
    therapeutic_accuracy DECIMAL(5,4) DEFAULT 0.0000,
    clinical_compliance DECIMAL(5,4) DEFAULT 0.0000,
    safety_score DECIMAL(5,4) DEFAULT 0.0000,
    conversation_coherence DECIMAL(5,4) DEFAULT 0.0000,
    emotional_authenticity DECIMAL(5,4) DEFAULT 0.0000,
    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validator_version VARCHAR(20),
    validation_notes TEXT,
    quality_metadata_json JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE conversation_tags (
    tag_id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    tag_type VARCHAR(50) NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) DEFAULT 1.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE dataset_sources (
    source_id VARCHAR(36) PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    tier VARCHAR(20) NOT NULL,
    description TEXT,
    total_conversations INTEGER DEFAULT 0,
    processed_conversations INTEGER DEFAULT 0,
    average_quality DECIMAL(5,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    source_metadata_json JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE processing_batches (
    batch_id VARCHAR(36) PRIMARY KEY,
    batch_name VARCHAR(100) NOT NULL,
    batch_type VARCHAR(50) NOT NULL,
    source_dataset VARCHAR(100),
    total_conversations INTEGER DEFAULT 0,
    processed_conversations INTEGER DEFAULT 0,
    failed_conversations INTEGER DEFAULT 0,
    processing_status VARCHAR(20) DEFAULT "pending",
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    processing_time_seconds INTEGER DEFAULT 0,
    batch_metadata_json JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE conversation_versions (
    version_id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    version_number INTEGER NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    change_description TEXT,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    previous_data_json LONGTEXT,
    change_metadata_json JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE conversation_search (
    search_id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    searchable_content LONGTEXT NOT NULL,
    content_vector JSON,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE conversation_statistics (
    stat_id VARCHAR(36) PRIMARY KEY,
    stat_type VARCHAR(50) NOT NULL,
    stat_date DATE NOT NULL,
    dataset_source VARCHAR(100),
    tier VARCHAR(20),
    total_conversations INTEGER DEFAULT 0,
    average_quality DECIMAL(5,4) DEFAULT 0.0000,
    average_word_count DECIMAL(8,2) DEFAULT 0.00,
    quality_distribution_json JSON,
    statistics_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Index Creation
CREATE INDEX idx_conversations_dataset_source ON conversations(dataset_source);
CREATE INDEX idx_conversations_tier ON conversations(tier);
CREATE INDEX idx_conversations_processing_status ON conversations(processing_status);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at);
CREATE INDEX idx_conversations_batch_id ON conversations(batch_id);
CREATE INDEX idx_conversations_quality ON conversations(dataset_source, tier);
CREATE INDEX idx_conversations_filename ON conversations(original_filename);
CREATE INDEX idx_quality_conversation_id ON conversation_quality(conversation_id);
CREATE INDEX idx_quality_overall ON conversation_quality(overall_quality);
CREATE INDEX idx_quality_therapeutic ON conversation_quality(therapeutic_accuracy);
CREATE INDEX idx_quality_safety ON conversation_quality(safety_score);
CREATE INDEX idx_quality_validation_date ON conversation_quality(validation_date);
CREATE INDEX idx_tags_conversation_id ON conversation_tags(conversation_id);
CREATE INDEX idx_tags_type_value ON conversation_tags(tag_type, tag_value);
CREATE INDEX idx_tags_value ON conversation_tags(tag_value);
CREATE INDEX idx_sources_name ON dataset_sources(source_name);
CREATE INDEX idx_sources_type ON dataset_sources(source_type);
CREATE INDEX idx_sources_tier ON dataset_sources(tier);
CREATE INDEX idx_batches_status ON processing_batches(processing_status);
CREATE INDEX idx_batches_started_at ON processing_batches(started_at);
CREATE INDEX idx_batches_source ON processing_batches(source_dataset);
CREATE INDEX idx_versions_conversation_id ON conversation_versions(conversation_id);
CREATE INDEX idx_versions_number ON conversation_versions(conversation_id, version_number);
CREATE INDEX idx_versions_changed_at ON conversation_versions(changed_at);
CREATE INDEX idx_search_conversation_id ON conversation_search(conversation_id);
CREATE FULLTEXT INDEX idx_search_content ON conversation_search(searchable_content);
CREATE INDEX idx_stats_type_date ON conversation_statistics(stat_type, stat_date);
CREATE INDEX idx_stats_source ON conversation_statistics(dataset_source);
CREATE INDEX idx_stats_tier ON conversation_statistics(tier);

-- Constraint Creation
ALTER TABLE conversation_quality ADD CONSTRAINT fk_quality_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE;
ALTER TABLE conversation_tags ADD CONSTRAINT fk_tags_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE;
ALTER TABLE conversation_versions ADD CONSTRAINT fk_versions_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE;
ALTER TABLE conversation_search ADD CONSTRAINT fk_search_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE;
ALTER TABLE processing_batches ADD CONSTRAINT fk_batches_source FOREIGN KEY (source_dataset) REFERENCES dataset_sources(source_name) ON DELETE SET NULL;
