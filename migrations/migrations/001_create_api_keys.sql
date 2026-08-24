-- Migration: 001_create_api_keys
-- Description: Create SQLite tables for persistent API key storage
-- This migration is designed to be run against the SQLite database at ai/data/conversation_system.db
-- Apply with: sqlite3 ai/data/conversation_system.db < ai/security/migrations/001_create_api_keys.sql

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    permissions TEXT NOT NULL DEFAULT '["read"]',       -- JSON array of PermissionLevel values
    is_active INTEGER NOT NULL DEFAULT 1,                -- 0 = inactive, 1 = active
    created_at TEXT NOT NULL DEFAULT (datetime('now')),  -- ISO-8601 timestamp
    expires_at TEXT,                                     -- ISO-8601 timestamp, NULL = no expiry
    last_used_at TEXT,                                   -- ISO-8601 timestamp, NULL = never used
    last_failed_at TEXT                                  -- ISO-8601 timestamp, NULL = no failures
);

CREATE TABLE IF NOT EXISTS api_key_rate_limits (
    api_key_id TEXT NOT NULL REFERENCES api_keys(key_id) ON DELETE CASCADE,
    window_start TEXT NOT NULL,                          -- ISO-8601 timestamp of rate-limit window start
    request_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_key_rate_limits_window_start ON api_key_rate_limits(window_start);
