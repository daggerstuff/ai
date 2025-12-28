# CMS Business Strategy Database Schema

**Version**: 1.0  
**Last Updated**: December 27, 2025  
**Status**: Design Phase  

---

## Overview

The Business Strategy Expansion & CMS System requires a multi-database approach:
- **MongoDB Atlas**: Document store for flexible business content, documents, and strategies
- **PostgreSQL (Supabase)**: Relational data for users, permissions, audit logs, and structured workflows
- **Redis**: Caching, real-time collaboration state, and session management

---

## 1. MongoDB Collections (Document Store)

### 1.1 Business Documents & Content

#### `business_documents`
Core collection for all business strategy documents, plans, and content.

```javascript
{
  _id: ObjectId,
  
  // Identity & Metadata
  documentId: String,           // Unique business document identifier
  title: String,                // Document title
  slug: String,                 // URL-friendly identifier
  type: Enum[
    'strategy_plan',
    'market_analysis',
    'competitive_analysis',
    'case_study',
    'pitch_deck',
    'operational_guide',
    'research_report',
    'custom'
  ],
  category: String,             // e.g., "Market Research", "Sales", "Marketing"
  description: String,          // Document description
  
  // Content
  content: {
    markdown: String,           // Full markdown content
    sections: [
      {
        id: String,
        title: String,
        content: String,
        order: Number
      }
    ],
    metadata: {
      wordCount: Number,
      readingTime: Number        // minutes
    }
  },
  
  // Status & Workflow
  status: Enum['draft', 'review', 'approved', 'published', 'archived'],
  version: Number,              // Version number
  revisions: [
    {
      revisionId: String,
      version: Number,
      timestamp: Date,
      author: ObjectId,          // Reference to user
      changes: String,           // Summary of changes
      content: String            // Full content snapshot
    }
  ],
  
  // Permissions & Ownership
  owner: ObjectId,              // Reference to user
  contributors: [ObjectId],     // References to contributing users
  permissions: {
    view: [ObjectId],           // Users who can view
    edit: [ObjectId],           // Users who can edit
    comment: [ObjectId]         // Users who can comment
  },
  
  // Relationships
  linkedDocuments: [String],    // IDs of related documents
  linkedProjects: [ObjectId],   // References to projects
  tags: [String],               // Searchable tags
  
  // Audit Trail
  createdAt: Date,
  updatedAt: Date,
  createdBy: ObjectId,
  updatedBy: ObjectId,
  lastReviewedAt: Date,
  lastReviewedBy: ObjectId,
  
  // SEO & Discovery
  seo: {
    metaTitle: String,
    metaDescription: String,
    keywords: [String]
  },
  
  // Attachments & Media
  attachments: [
    {
      id: String,
      filename: String,
      url: String,               // S3/Azure Blob URL
      mimeType: String,
      size: Number,
      uploadedAt: Date,
      uploadedBy: ObjectId
    }
  ]
}
```

### 1.2 Projects & Initiatives

#### `projects`
Business strategy projects and initiatives.

```javascript
{
  _id: ObjectId,
  projectId: String,            // Unique identifier
  name: String,                 // Project name
  description: String,
  status: Enum['planning', 'active', 'on_hold', 'completed', 'cancelled'],
  
  // Timeline
  startDate: Date,
  targetCompletionDate: Date,
  actualCompletionDate: Date,
  
  // Leadership
  owner: ObjectId,              // Project owner (reference to user)
  stakeholders: [
    {
      userId: ObjectId,
      role: String,             // e.g., "Lead", "Contributor", "Reviewer"
      joinedAt: Date
    }
  ],
  
  // Scope & Objectives
  objectives: [
    {
      id: String,
      description: String,
      successCriteria: [String],
      priority: Enum['critical', 'high', 'medium', 'low'],
      status: Enum['not_started', 'in_progress', 'completed', 'blocked']
    }
  ],
  
  // Linked Content
  linkedDocuments: [String],    // Document IDs
  linkedStrategies: [ObjectId], // Strategy IDs
  relatedProjects: [ObjectId],  // Project IDs
  
  // Metadata
  budget: Number,
  allocatedResources: [String],
  riskAssessment: String,
  
  // Timestamps
  createdAt: Date,
  updatedAt: Date,
  createdBy: ObjectId
}
```

### 1.3 Market Research & Analytics

#### `market_research`
Market analysis data, trends, competitive intelligence.

```javascript
{
  _id: ObjectId,
  researchId: String,
  title: String,
  type: Enum['market_analysis', 'competitor_analysis', 'trend_research', 'customer_research'],
  
  // Research Data
  findings: [
    {
      id: String,
      title: String,
      description: String,
      impact: Enum['high', 'medium', 'low'],
      evidence: [String],       // References or quotes
      implications: String      // How this affects strategy
    }
  ],
  
  // Target Market Data
  targetMarkets: [
    {
      segment: String,          // e.g., "Academic Institutions"
      size: Number,             // Market size in dollars or units
      growth_rate: Number,      // Annual growth %
      key_players: [String],
      opportunities: [String],
      threats: [String]
    }
  ],
  
  // Competitive Landscape
  competitors: [
    {
      id: String,
      name: String,
      strengths: [String],
      weaknesses: [String],
      market_share: Number,
      pricing_strategy: String,
      unique_selling_proposition: String
    }
  ],
  
  // Timeline
  researchDate: Date,
  nextReviewDate: Date,
  sources: [
    {
      name: String,
      url: String,
      accessedDate: Date,
      credibility: Enum['high', 'medium', 'low']
    }
  ],
  
  // Metadata
  author: ObjectId,
  status: Enum['draft', 'validated', 'published'],
  createdAt: Date,
  updatedAt: Date
}
```

### 1.4 Strategic Plans & Roadmaps

#### `strategic_plans`
High-level strategic plans and roadmaps.

```javascript
{
  _id: ObjectId,
  planId: String,
  title: String,                // e.g., "2025-2026 Growth Strategy"
  planType: Enum['annual', 'quarterly', 'multi_year', 'product', 'market'],
  
  // Timeline
  fiscalYear: Number,
  quarter: Number,              // 1-4, null for annual
  startDate: Date,
  endDate: Date,
  
  // Strategic Elements
  vision: String,               // Long-term vision
  mission: String,              // Current mission
  keyObjectives: [
    {
      id: String,
      title: String,
      description: String,
      keyResults: [
        {
          id: String,
          description: String,
          target: Number,
          actual: Number,
          unit: String,           // %, $, customers, etc.
          dueDate: Date,
          status: Enum['on_track', 'at_risk', 'off_track', 'completed']
        }
      ]
    }
  ],
  
  // Initiatives & Projects
  initiatives: [ObjectId],      // References to projects
  
  // Resource Allocation
  budgetAllocation: {
    total: Number,
    byFunction: {
      sales: Number,
      marketing: Number,
      operations: Number,
      technology: Number,
      other: Number
    }
  },
  
  // Risk Management
  risks: [
    {
      id: String,
      description: String,
      probability: Enum['high', 'medium', 'low'],
      impact: Enum['high', 'medium', 'low'],
      mitigation_strategy: String,
      owner: ObjectId
    }
  ],
  
  // Success Metrics
  kpis: [
    {
      id: String,
      name: String,
      target: Number,
      unit: String,
      measurement_frequency: String,
      owner: ObjectId,
      current_value: Number,
      last_updated: Date
    }
  ],
  
  // Review & Approval
  owner: ObjectId,
  approvers: [ObjectId],
  status: Enum['draft', 'under_review', 'approved', 'executing', 'completed', 'archived'],
  approvalDate: Date,
  
  // Audit Trail
  createdAt: Date,
  updatedAt: Date,
  createdBy: ObjectId
}
```

### 1.5 Sales & Partnerships

#### `sales_opportunities`
Sales pipeline and opportunity tracking.

```javascript
{
  _id: ObjectId,
  opportunityId: String,
  account: String,              // Customer/Institution name
  
  // Opportunity Details
  title: String,
  description: String,
  value: Number,                // Deal value
  currency: String,             // USD, EUR, etc.
  stage: Enum[
    'prospect',
    'qualified_lead',
    'proposal',
    'negotiation',
    'won',
    'lost',
    'stalled'
  ],
  
  // Timeline
  createdDate: Date,
  expectedCloseDate: Date,
  actualCloseDate: Date,
  
  // People & Relationships
  owner: ObjectId,              // Sales rep
  accountManager: ObjectId,
  contacts: [
    {
      name: String,
      email: String,
      phone: String,
      title: String,
      department: String,
      lastContact: Date
    }
  ],
  
  // Activity & Engagement
  nextAction: String,
  nextActionDate: Date,
  activities: [
    {
      type: String,             // call, email, meeting, proposal
      date: Date,
      notes: String,
      participant: ObjectId
    }
  ],
  
  // Metadata
  source: Enum['inbound', 'outbound', 'referral', 'event', 'partnership'],
  priority: Enum['high', 'medium', 'low'],
  probability: Number,          // 0-100%
  
  // Documents
  linkedDocuments: [String],    // Proposals, contracts, etc.
  
  // Audit Trail
  createdAt: Date,
  updatedAt: Date,
  createdBy: ObjectId
}
```

### 1.6 Content Hub & Knowledge Base

#### `knowledge_articles`
Reusable content for thought leadership, guides, best practices.

```javascript
{
  _id: ObjectId,
  articleId: String,
  title: String,
  slug: String,
  category: String,
  
  // Content
  content: String,              // Markdown
  summary: String,
  author: ObjectId,
  
  // Metadata
  publishedDate: Date,
  updatedDate: Date,
  status: Enum['draft', 'published', 'archived'],
  featured: Boolean,
  
  // Engagement
  views: Number,
  shares: Number,
  likes: Number,
  
  // SEO
  seo: {
    metaTitle: String,
    metaDescription: String,
    keywords: [String]
  },
  
  // Relationships
  tags: [String],
  relatedArticles: [String],
  linkedResources: [String],
  
  // Audit
  createdAt: Date,
  createdBy: ObjectId
}
```

---

## 2. PostgreSQL Tables (Relational Data)

### 2.1 Users & Permissions

#### `users`
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL, -- admin, manager, analyst, viewer
  department VARCHAR(100),
  status VARCHAR(50) DEFAULT 'active',
  last_login TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `permissions`
```sql
CREATE TABLE permissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  resource_type VARCHAR(100) NOT NULL, -- document, project, strategy, etc.
  resource_id VARCHAR(255) NOT NULL,
  permission_level VARCHAR(50) NOT NULL, -- view, edit, approve, admin
  granted_by UUID REFERENCES users(id),
  granted_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP,
  
  UNIQUE(user_id, resource_type, resource_id)
);
```

#### `roles`
```sql
CREATE TABLE roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(100) UNIQUE NOT NULL,
  description TEXT,
  permissions TEXT[] DEFAULT '{}', -- JSON array of permission strings
  created_at TIMESTAMP DEFAULT NOW()
);
```

### 2.2 Audit & Compliance

#### `audit_logs`
```sql
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  action VARCHAR(100) NOT NULL, -- create, update, delete, view, export
  resource_type VARCHAR(100) NOT NULL,
  resource_id VARCHAR(255),
  changes JSONB, -- Track what changed
  ip_address INET,
  user_agent VARCHAR(500),
  status VARCHAR(50) DEFAULT 'success',
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  
  INDEX idx_user_action_created (user_id, action, created_at),
  INDEX idx_resource_created (resource_type, resource_id, created_at)
);
```

#### `data_access_logs`
```sql
CREATE TABLE data_access_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  document_id VARCHAR(255) NOT NULL,
  access_type VARCHAR(50) NOT NULL, -- view, download, export
  accessed_at TIMESTAMP DEFAULT NOW(),
  data_sensitivity VARCHAR(50), -- public, internal, confidential, restricted
  
  INDEX idx_user_access (user_id, accessed_at)
);
```

### 2.3 Workflow & Approvals

#### `approval_workflows`
```sql
CREATE TABLE approval_workflows (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  resource_type VARCHAR(100) NOT NULL, -- e.g., strategy_plan, business_document
  status VARCHAR(50) DEFAULT 'active',
  
  -- Approval chain
  steps JSONB NOT NULL, -- Array of approval steps with required roles
  
  created_at TIMESTAMP DEFAULT NOW(),
  created_by UUID REFERENCES users(id)
);

CREATE TABLE approval_requests (
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
  
  INDEX idx_status_created (status, created_at)
);

CREATE TABLE approval_steps (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  approval_request_id UUID NOT NULL REFERENCES approval_requests(id) ON DELETE CASCADE,
  step_number INT NOT NULL,
  required_role VARCHAR(100) NOT NULL,
  assigned_to UUID REFERENCES users(id),
  status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected
  
  reviewed_at TIMESTAMP,
  reviewed_by UUID REFERENCES users(id),
  review_comments TEXT,
  
  UNIQUE(approval_request_id, step_number)
);
```

### 2.4 Notifications & Communication

#### `notifications`
```sql
CREATE TABLE notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type VARCHAR(100) NOT NULL, -- document_shared, approval_request, comment_mentioned
  title VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  resource_type VARCHAR(100),
  resource_id VARCHAR(255),
  
  read_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  
  INDEX idx_user_read_created (user_id, read_at, created_at)
);
```

### 2.5 Collaboration

#### `comments`
```sql
CREATE TABLE comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id VARCHAR(255) NOT NULL,
  author_id UUID NOT NULL REFERENCES users(id),
  content TEXT NOT NULL,
  
  -- Threading
  parent_comment_id UUID REFERENCES comments(id),
  resolved BOOLEAN DEFAULT FALSE,
  resolved_by UUID REFERENCES users(id),
  resolved_at TIMESTAMP,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  INDEX idx_document_created (document_id, created_at),
  INDEX idx_author_created (author_id, created_at)
);
```

---

## 3. Redis Keys (Cache & Real-time)

### 3.1 Caching Strategy

```
# Document Cache (1 hour TTL)
doc:{documentId} → JSON

# User Session Cache (24 hour TTL)
session:{sessionId} → User session data

# Real-time Collaboration State
collab:{documentId}:users → Set of active editing users
collab:{documentId}:cursors → Cursor positions by user
collab:{documentId}:changes → Pending changes queue

# Feature Flags & Configuration
feature:{featureName} → Boolean

# Rate Limiting
ratelimit:{userId}:{action} → Count

# Analytics Cache (1 hour TTL)
stats:{type}:{period} → Aggregated metrics

# Notifications Queue
notifications:{userId} → Queue of pending notifications

# Lock Management
lock:{documentId}:{userId} → Timestamp (mutex for concurrent edits)
```

---

## 4. Data Relationships & Access Patterns

### 4.1 Key Query Patterns

**Fetch User's Accessible Documents**
```javascript
// MongoDB
db.business_documents.find({
  $or: [
    { owner: userId },
    { "permissions.view": userId },
    { "permissions.edit": userId }
  ]
})
```

**Get Project with All Linked Documents**
```javascript
// MongoDB - aggregation pipeline
db.projects.aggregate([
  { $match: { projectId: projectId } },
  { $lookup: {
    from: "business_documents",
    localField: "linkedDocuments",
    foreignField: "documentId",
    as: "documents"
  }}
])
```

**Fetch Approval Chain for Strategic Plan**
```sql
-- PostgreSQL
SELECT 
  ar.id, ar.status, ar.current_step,
  step.step_number, step.required_role, step.status,
  u.name, u.email
FROM approval_requests ar
JOIN approval_steps step ON ar.id = step.approval_request_id
LEFT JOIN users u ON step.reviewed_by = u.id
WHERE ar.resource_id = $1 AND ar.resource_type = 'strategic_plan'
ORDER BY step.step_number;
```

---

## 5. Implementation Roadmap

### Phase 1 (Week 1-2): Foundation
- [ ] Create MongoDB collections with indexes
- [ ] Create PostgreSQL schema and relationships
- [ ] Set up Redis connection and key structure
- [ ] Implement connection pooling and error handling

### Phase 2 (Week 3-4): DAL & Repositories
- [ ] Create MongoDB data access layer
- [ ] Create PostgreSQL ORM/query builders
- [ ] Implement caching layer
- [ ] Set up migration system

### Phase 3 (Week 5-6): APIs & Services
- [ ] REST API endpoints for documents
- [ ] Project management APIs
- [ ] Workflow & approval APIs
- [ ] Search and filtering services

### Phase 4 (Week 7-8): Testing & Optimization
- [ ] Integration tests
- [ ] Performance testing
- [ ] Index optimization
- [ ] Load testing

---

*Last Updated: December 27, 2025*

