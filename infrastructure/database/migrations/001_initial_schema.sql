-- PostgreSQL Schema Migrations for CMS Business Strategy System
-- Database: pixelated-business-strategy
-- Version: 1.0

-- ============================================================================
-- USERS & AUTHENTICATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL DEFAULT 'viewer', -- admin, manager, analyst, viewer
  department VARCHAR(100),
  status VARCHAR(50) DEFAULT 'active', -- active, inactive, suspended
  last_login TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  CHECK (role IN ('admin', 'manager', 'analyst', 'viewer')),
  CHECK (status IN ('active', 'inactive', 'suspended'))
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_status ON users(status);

-- ============================================================================
-- ROLES & PERMISSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(100) UNIQUE NOT NULL,
  description TEXT,
  permissions TEXT[] DEFAULT '{}', -- JSON array of permission strings
  is_system BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS permissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  resource_type VARCHAR(100) NOT NULL, -- document, project, strategy, etc.
  resource_id VARCHAR(255) NOT NULL,
  permission_level VARCHAR(50) NOT NULL, -- view, edit, approve, admin
  granted_by UUID REFERENCES users(id),
  granted_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP,
  
  UNIQUE(user_id, resource_type, resource_id),
  CHECK (permission_level IN ('view', 'edit', 'approve', 'admin'))
);

CREATE INDEX idx_permissions_user_resource ON permissions(user_id, resource_type);
CREATE INDEX idx_permissions_resource ON permissions(resource_type, resource_id);
CREATE INDEX idx_permissions_level ON permissions(permission_level);

-- ============================================================================
-- AUDIT & COMPLIANCE LOGGING
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  action VARCHAR(100) NOT NULL, -- create, update, delete, view, export
  resource_type VARCHAR(100) NOT NULL,
  resource_id VARCHAR(255),
  changes JSONB, -- Track what changed: {field: {old: value, new: value}}
  ip_address INET,
  user_agent VARCHAR(500),
  status VARCHAR(50) DEFAULT 'success', -- success, error
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  
  CHECK (status IN ('success', 'error'))
);

CREATE INDEX idx_audit_user_action ON audit_logs(user_id, action, created_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id, created_at DESC);
CREATE INDEX idx_audit_action ON audit_logs(action, created_at DESC);

-- ============================================================================
-- DATA ACCESS LOGGING (HIPAA Compliance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_access_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  document_id VARCHAR(255) NOT NULL,
  access_type VARCHAR(50) NOT NULL, -- view, download, export
  accessed_at TIMESTAMP DEFAULT NOW(),
  data_sensitivity VARCHAR(50), -- public, internal, confidential, restricted
  duration_seconds INT,
  
  CHECK (access_type IN ('view', 'download', 'export')),
  CHECK (data_sensitivity IN ('public', 'internal', 'confidential', 'restricted'))
);

CREATE INDEX idx_data_access_user ON data_access_logs(user_id, accessed_at DESC);
CREATE INDEX idx_data_access_document ON data_access_logs(document_id, accessed_at DESC);
CREATE INDEX idx_data_access_sensitivity ON data_access_logs(data_sensitivity, accessed_at DESC);

-- ============================================================================
-- WORKFLOW & APPROVAL SYSTEM
-- ============================================================================

CREATE TABLE IF NOT EXISTS approval_workflows (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  resource_type VARCHAR(100) NOT NULL, -- e.g., strategy_plan, business_document
  status VARCHAR(50) DEFAULT 'active', -- active, inactive, deprecated
  
  -- Approval chain structure (JSONB for flexibility)
  steps JSONB NOT NULL, -- [{step_number: 1, required_role: 'manager', ...}]
  
  created_at TIMESTAMP DEFAULT NOW(),
  created_by UUID REFERENCES users(id),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  UNIQUE(name, resource_type),
  CHECK (status IN ('active', 'inactive', 'deprecated'))
);

CREATE INDEX idx_workflow_resource_type ON approval_workflows(resource_type);
CREATE INDEX idx_workflow_status ON approval_workflows(status);

-- Approval Requests
CREATE TABLE IF NOT EXISTS approval_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id UUID REFERENCES approval_workflows(id),
  resource_type VARCHAR(100) NOT NULL,
  resource_id VARCHAR(255) NOT NULL,
  requestor_id UUID NOT NULL REFERENCES users(id),
  
  status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected, withdrawn
  current_step INT DEFAULT 1,
  total_steps INT,
  
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP,
  
  CHECK (status IN ('pending', 'approved', 'rejected', 'withdrawn'))
);

CREATE INDEX idx_approval_status ON approval_requests(status, created_at DESC);
CREATE INDEX idx_approval_resource ON approval_requests(resource_type, resource_id);
CREATE INDEX idx_approval_requestor ON approval_requests(requestor_id, created_at DESC);

-- Approval Steps (audit trail for each step in the workflow)
CREATE TABLE IF NOT EXISTS approval_steps (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  approval_request_id UUID NOT NULL REFERENCES approval_requests(id) ON DELETE CASCADE,
  step_number INT NOT NULL,
  required_role VARCHAR(100) NOT NULL,
  assigned_to UUID REFERENCES users(id),
  status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected
  
  reviewed_at TIMESTAMP,
  reviewed_by UUID REFERENCES users(id),
  review_comments TEXT,
  
  UNIQUE(approval_request_id, step_number),
  CHECK (status IN ('pending', 'approved', 'rejected'))
);

CREATE INDEX idx_approval_steps_request ON approval_steps(approval_request_id);
CREATE INDEX idx_approval_steps_assigned ON approval_steps(assigned_to, status);

-- ============================================================================
-- NOTIFICATIONS & ALERTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type VARCHAR(100) NOT NULL, -- document_shared, approval_request, comment_mentioned
  title VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  resource_type VARCHAR(100),
  resource_id VARCHAR(255),
  action_url VARCHAR(500),
  
  read_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  
  CHECK (type IN ('document_shared', 'approval_request', 'comment_mentioned', 'document_updated'))
);

CREATE INDEX idx_notifications_user_read ON notifications(user_id, read_at, created_at DESC);
CREATE INDEX idx_notifications_type ON notifications(type, created_at DESC);

-- ============================================================================
-- COLLABORATION & COMMENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id VARCHAR(255) NOT NULL,
  author_id UUID NOT NULL REFERENCES users(id),
  content TEXT NOT NULL,
  
  -- Threading for comments
  parent_comment_id UUID REFERENCES comments(id),
  resolved BOOLEAN DEFAULT FALSE,
  resolved_by UUID REFERENCES users(id),
  resolved_at TIMESTAMP,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_comments_document ON comments(document_id, created_at DESC);
CREATE INDEX idx_comments_author ON comments(author_id, created_at DESC);
CREATE INDEX idx_comments_parent ON comments(parent_comment_id);
CREATE INDEX idx_comments_resolved ON comments(resolved, created_at DESC);

-- ============================================================================
-- DOCUMENT SHARING & COLLABORATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_shares (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id VARCHAR(255) NOT NULL,
  shared_by UUID NOT NULL REFERENCES users(id),
  shared_with UUID NOT NULL REFERENCES users(id),
  permission_level VARCHAR(50) NOT NULL, -- view, edit, comment
  shared_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP,
  
  UNIQUE(document_id, shared_with),
  CHECK (permission_level IN ('view', 'edit', 'comment'))
);

CREATE INDEX idx_document_shares_document ON document_shares(document_id);
CREATE INDEX idx_document_shares_with ON document_shares(shared_with, shared_at DESC);

-- ============================================================================
-- ACTIVITY & ENGAGEMENT TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_activity (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id VARCHAR(255) NOT NULL,
  user_id UUID NOT NULL REFERENCES users(id),
  activity_type VARCHAR(100) NOT NULL, -- view, edit, comment, download, export
  metadata JSONB, -- Additional activity data
  created_at TIMESTAMP DEFAULT NOW(),
  
  CHECK (activity_type IN ('view', 'edit', 'comment', 'download', 'export', 'shared'))
);

CREATE INDEX idx_activity_document ON document_activity(document_id, created_at DESC);
CREATE INDEX idx_activity_user ON document_activity(user_id, created_at DESC);
CREATE INDEX idx_activity_type ON document_activity(activity_type, created_at DESC);

-- ============================================================================
-- CONTENT VERSIONING & HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id VARCHAR(255) NOT NULL,
  version_number INT NOT NULL,
  title VARCHAR(255),
  content TEXT,
  created_by UUID NOT NULL REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  change_summary TEXT,
  
  UNIQUE(document_id, version_number)
);

CREATE INDEX idx_versions_document ON document_versions(document_id, version_number DESC);

-- ============================================================================
-- SYSTEM SETTINGS & CONFIGURATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key VARCHAR(255) UNIQUE NOT NULL,
  value JSONB NOT NULL,
  description TEXT,
  is_sensitive BOOLEAN DEFAULT FALSE,
  updated_by UUID REFERENCES users(id),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_settings_key ON system_settings(key);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active users with their roles
CREATE OR REPLACE VIEW v_active_users AS
SELECT 
  id,
  email,
  name,
  role,
  department,
  last_login,
  created_at
FROM users
WHERE status = 'active'
ORDER BY name;

-- User permissions with resource details
CREATE OR REPLACE VIEW v_user_permissions AS
SELECT 
  u.id as user_id,
  u.email,
  p.resource_type,
  p.resource_id,
  p.permission_level,
  p.granted_at,
  p.expires_at
FROM users u
LEFT JOIN permissions p ON u.id = p.user_id
WHERE p.id IS NOT NULL
  AND (p.expires_at IS NULL OR p.expires_at > NOW())
ORDER BY u.email, p.resource_type, p.resource_id;

-- Pending approvals by user
CREATE OR REPLACE VIEW v_pending_approvals AS
SELECT 
  ar.id as approval_request_id,
  ar.resource_type,
  ar.resource_id,
  ast.step_number,
  u.email as assigned_to,
  ar.created_at,
  (ar.created_at + INTERVAL '7 days') as due_date
FROM approval_requests ar
JOIN approval_steps ast ON ar.id = ast.approval_request_id
LEFT JOIN users u ON ast.assigned_to = u.id
WHERE ar.status = 'pending'
  AND ast.status = 'pending'
ORDER BY ar.created_at DESC;

-- Recent audit activity
CREATE OR REPLACE VIEW v_recent_audit_activity AS
SELECT 
  u.email,
  al.action,
  al.resource_type,
  al.resource_id,
  al.status,
  al.created_at
FROM audit_logs al
JOIN users u ON al.user_id = u.id
ORDER BY al.created_at DESC
LIMIT 1000;

-- ============================================================================
-- GRANTS & SECURITY
-- ============================================================================

-- Create read-only role for analysts
CREATE ROLE analyst_role NOLOGIN;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_role;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO analyst_role;

-- Create content manager role
CREATE ROLE content_manager_role NOLOGIN;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO content_manager_role;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO content_manager_role;

-- Create admin role
CREATE ROLE admin_role NOLOGIN;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_role;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin_role;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

INSERT INTO roles (name, description, permissions, is_system) VALUES
  ('admin', 'System Administrator', ARRAY['*'], TRUE),
  ('manager', 'Content Manager', ARRAY['document:*', 'project:*', 'approval:*'], TRUE),
  ('analyst', 'Data Analyst', ARRAY['document:read', 'project:read', 'report:read'], TRUE),
  ('viewer', 'Viewer', ARRAY['document:read'], TRUE)
ON CONFLICT DO NOTHING;

INSERT INTO system_settings (key, value, description) VALUES
  ('cms.version', '"1.0"', 'CMS System Version'),
  ('cms.max_document_size_mb', '100', 'Maximum document upload size in MB'),
  ('cms.approval_timeout_days', '7', 'Approval request timeout in days'),
  ('cms.archive_after_days', '730', 'Auto-archive documents after days of inactivity')
ON CONFLICT DO NOTHING;
