// MongoDB Collection Schemas for CMS Business Strategy System
// Database: pixelated-business-strategy
// Version: 1.0
// Companions: 001_initial_schema.sql (PostgreSQL), cms_redis_config.py (Redis)

// ============================================================================
// BUSINESS DOCUMENTS
// ============================================================================

db.createCollection("business_documents", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["documentId", "title", "slug", "type", "status", "owner", "createdAt", "updatedAt"],
      properties: {
        documentId: { bsonType: "string" },
        title: { bsonType: "string" },
        slug: { bsonType: "string" },
        type: {
          enum: [
            "strategy_plan", "market_analysis", "competitive_analysis",
            "case_study", "pitch_deck", "operational_guide",
            "research_report", "custom"
          ]
        },
        category: { bsonType: "string" },
        description: { bsonType: "string" },
        content: {
          bsonType: "object",
          properties: {
            markdown: { bsonType: "string" },
            sections: {
              bsonType: "array",
              items: {
                bsonType: "object",
                required: ["id", "title", "content", "order"],
                properties: {
                  id: { bsonType: "string" },
                  title: { bsonType: "string" },
                  content: { bsonType: "string" },
                  order: { bsonType: "int" }
                }
              }
            },
            metadata: {
              bsonType: "object",
              properties: {
                wordCount: { bsonType: "int" },
                readingTime: { bsonType: "double" }
              }
            }
          }
        },
        status: {
          enum: ["draft", "review", "approved", "published", "archived"]
        },
        version: { bsonType: "int" },
        revisions: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["revisionId", "version", "timestamp", "author"],
            properties: {
              revisionId: { bsonType: "string" },
              version: { bsonType: "int" },
              timestamp: { bsonType: "date" },
              author: { bsonType: "objectId" },
              changes: { bsonType: "string" },
              content: { bsonType: "string" }
            }
          }
        },
        owner: { bsonType: "objectId" },
        contributors: {
          bsonType: "array",
          items: { bsonType: "objectId" }
        },
        permissions: {
          bsonType: "object",
          properties: {
            view: { bsonType: "array", items: { bsonType: "objectId" } },
            edit: { bsonType: "array", items: { bsonType: "objectId" } },
            comment: { bsonType: "array", items: { bsonType: "objectId" } }
          }
        },
        linkedDocuments: { bsonType: "array", items: { bsonType: "string" } },
        linkedProjects: { bsonType: "array", items: { bsonType: "objectId" } },
        tags: { bsonType: "array", items: { bsonType: "string" } },
        createdAt: { bsonType: "date" },
        updatedAt: { bsonType: "date" },
        createdBy: { bsonType: "objectId" },
        updatedBy: { bsonType: "objectId" },
        lastReviewedAt: { bsonType: "date" },
        lastReviewedBy: { bsonType: "objectId" },
        seo: {
          bsonType: "object",
          properties: {
            metaTitle: { bsonType: "string" },
            metaDescription: { bsonType: "string" },
            keywords: { bsonType: "array", items: { bsonType: "string" } }
          }
        },
        attachments: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "filename", "url", "mimeType", "size", "uploadedAt", "uploadedBy"],
            properties: {
              id: { bsonType: "string" },
              filename: { bsonType: "string" },
              url: { bsonType: "string" },
              mimeType: { bsonType: "string" },
              size: { bsonType: "int" },
              uploadedAt: { bsonType: "date" },
              uploadedBy: { bsonType: "objectId" }
            }
          }
        }
      }
    }
  }
});

db.business_documents.createIndex({ documentId: 1 }, { unique: true });
db.business_documents.createIndex({ slug: 1 }, { unique: true });
db.business_documents.createIndex({ type: 1, status: 1 });
db.business_documents.createIndex({ owner: 1, status: 1 });
db.business_documents.createIndex({ "permissions.view": 1 });
db.business_documents.createIndex({ "permissions.edit": 1 });
db.business_documents.createIndex({ tags: 1 });
db.business_documents.createIndex({ category: 1 });
db.business_documents.createIndex({ updatedAt: -1 });
db.business_documents.createIndex({ createdAt: -1 });
db.business_documents.createIndex({
  title: "text",
  description: "text",
  "content.markdown": "text",
  "seo.keywords": "text",
  tags: "text"
}, {
  name: "document_text_search",
  weights: {
    title: 10,
    "seo.keywords": 5,
    tags: 3,
    description: 2,
    "content.markdown": 1
  }
});

// ============================================================================
// PROJECTS
// ============================================================================

db.createCollection("projects", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["projectId", "name", "status", "owner", "createdAt", "updatedAt"],
      properties: {
        projectId: { bsonType: "string" },
        name: { bsonType: "string" },
        description: { bsonType: "string" },
        status: {
          enum: ["planning", "active", "on_hold", "completed", "cancelled"]
        },
        startDate: { bsonType: "date" },
        targetCompletionDate: { bsonType: "date" },
        actualCompletionDate: { bsonType: "date" },
        owner: { bsonType: "objectId" },
        stakeholders: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["userId", "role", "joinedAt"],
            properties: {
              userId: { bsonType: "objectId" },
              role: { bsonType: "string" },
              joinedAt: { bsonType: "date" }
            }
          }
        },
        objectives: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "description"],
            properties: {
              id: { bsonType: "string" },
              description: { bsonType: "string" },
              successCriteria: { bsonType: "array", items: { bsonType: "string" } },
              priority: { enum: ["critical", "high", "medium", "low"] },
              status: { enum: ["not_started", "in_progress", "completed", "blocked"] }
            }
          }
        },
        linkedDocuments: { bsonType: "array", items: { bsonType: "string" } },
        linkedStrategies: { bsonType: "array", items: { bsonType: "objectId" } },
        relatedProjects: { bsonType: "array", items: { bsonType: "objectId" } },
        budget: { bsonType: "double" },
        allocatedResources: { bsonType: "array", items: { bsonType: "string" } },
        riskAssessment: { bsonType: "string" },
        createdAt: { bsonType: "date" },
        updatedAt: { bsonType: "date" },
        createdBy: { bsonType: "objectId" }
      }
    }
  }
});

db.projects.createIndex({ projectId: 1 }, { unique: true });
db.projects.createIndex({ status: 1, startDate: -1 });
db.projects.createIndex({ owner: 1, status: 1 });
db.projects.createIndex({ "stakeholders.userId": 1 });
db.projects.createIndex({ "objectives.status": 1 });
db.projects.createIndex({ targetCompletionDate: 1 });
db.projects.createIndex({
  name: "text",
  description: "text",
  riskAssessment: "text"
}, {
  name: "project_text_search",
  weights: { name: 10, description: 3, riskAssessment: 1 }
});

// ============================================================================
// MARKET RESEARCH
// ============================================================================

db.createCollection("market_research", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["researchId", "title", "type", "author", "status", "createdAt", "updatedAt"],
      properties: {
        researchId: { bsonType: "string" },
        title: { bsonType: "string" },
        type: {
          enum: ["market_analysis", "competitor_analysis", "trend_research", "customer_research"]
        },
        findings: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "title", "description"],
            properties: {
              id: { bsonType: "string" },
              title: { bsonType: "string" },
              description: { bsonType: "string" },
              impact: { enum: ["high", "medium", "low"] },
              evidence: { bsonType: "array", items: { bsonType: "string" } },
              implications: { bsonType: "string" }
            }
          }
        },
        targetMarkets: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["segment"],
            properties: {
              segment: { bsonType: "string" },
              size: { bsonType: "double" },
              growth_rate: { bsonType: "double" },
              key_players: { bsonType: "array", items: { bsonType: "string" } },
              opportunities: { bsonType: "array", items: { bsonType: "string" } },
              threats: { bsonType: "array", items: { bsonType: "string" } }
            }
          }
        },
        competitors: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "name"],
            properties: {
              id: { bsonType: "string" },
              name: { bsonType: "string" },
              strengths: { bsonType: "array", items: { bsonType: "string" } },
              weaknesses: { bsonType: "array", items: { bsonType: "string" } },
              market_share: { bsonType: "double" },
              pricing_strategy: { bsonType: "string" },
              unique_selling_proposition: { bsonType: "string" }
            }
          }
        },
        researchDate: { bsonType: "date" },
        nextReviewDate: { bsonType: "date" },
        sources: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["name", "accessedDate"],
            properties: {
              name: { bsonType: "string" },
              url: { bsonType: "string" },
              accessedDate: { bsonType: "date" },
              credibility: { enum: ["high", "medium", "low"] }
            }
          }
        },
        author: { bsonType: "objectId" },
        status: { enum: ["draft", "validated", "published"] },
        createdAt: { bsonType: "date" },
        updatedAt: { bsonType: "date" }
      }
    }
  }
});

db.market_research.createIndex({ researchId: 1 }, { unique: true });
db.market_research.createIndex({ type: 1, status: 1 });
db.market_research.createIndex({ author: 1 });
db.market_research.createIndex({ researchDate: -1 });
db.market_research.createIndex({ nextReviewDate: 1 });
db.market_research.createIndex({ "competitors.name": 1 });
db.market_research.createIndex({ "targetMarkets.segment": 1 });
db.market_research.createIndex({
  title: "text",
  "findings.title": "text",
  "findings.description": "text",
  "findings.implications": "text",
  "targetMarkets.segment": "text",
  "competitors.name": "text"
}, {
  name: "research_text_search",
  weights: {
    title: 10,
    "competitors.name": 5,
    "targetMarkets.segment": 5,
    "findings.title": 3,
    "findings.description": 2,
    "findings.implications": 1
  }
});

// ============================================================================
// STRATEGIC PLANS
// ============================================================================

db.createCollection("strategic_plans", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["planId", "title", "planType", "owner", "status", "createdAt", "updatedAt"],
      properties: {
        planId: { bsonType: "string" },
        title: { bsonType: "string" },
        planType: {
          enum: ["annual", "quarterly", "multi_year", "product", "market"]
        },
        fiscalYear: { bsonType: "int" },
        quarter: { bsonType: "int" },
        startDate: { bsonType: "date" },
        endDate: { bsonType: "date" },
        vision: { bsonType: "string" },
        mission: { bsonType: "string" },
        keyObjectives: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "title", "description"],
            properties: {
              id: { bsonType: "string" },
              title: { bsonType: "string" },
              description: { bsonType: "string" },
              keyResults: {
                bsonType: "array",
                items: {
                  bsonType: "object",
                  required: ["id", "description", "target", "unit", "dueDate", "status"],
                  properties: {
                    id: { bsonType: "string" },
                    description: { bsonType: "string" },
                    target: { bsonType: "double" },
                    actual: { bsonType: "double" },
                    unit: { bsonType: "string" },
                    dueDate: { bsonType: "date" },
                    status: { enum: ["on_track", "at_risk", "off_track", "completed"] }
                  }
                }
              }
            }
          }
        },
        initiatives: { bsonType: "array", items: { bsonType: "objectId" } },
        budgetAllocation: {
          bsonType: "object",
          properties: {
            total: { bsonType: "double" },
            byFunction: {
              bsonType: "object",
              properties: {
                sales: { bsonType: "double" },
                marketing: { bsonType: "double" },
                operations: { bsonType: "double" },
                technology: { bsonType: "double" },
                other: { bsonType: "double" }
              }
            }
          }
        },
        risks: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "description"],
            properties: {
              id: { bsonType: "string" },
              description: { bsonType: "string" },
              probability: { enum: ["high", "medium", "low"] },
              impact: { enum: ["high", "medium", "low"] },
              mitigation_strategy: { bsonType: "string" },
              owner: { bsonType: "objectId" }
            }
          }
        },
        kpis: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["id", "name", "target", "unit", "measurement_frequency", "owner"],
            properties: {
              id: { bsonType: "string" },
              name: { bsonType: "string" },
              target: { bsonType: "double" },
              unit: { bsonType: "string" },
              measurement_frequency: { bsonType: "string" },
              owner: { bsonType: "objectId" },
              current_value: { bsonType: "double" },
              last_updated: { bsonType: "date" }
            }
          }
        },
        owner: { bsonType: "objectId" },
        approvers: { bsonType: "array", items: { bsonType: "objectId" } },
        status: {
          enum: ["draft", "under_review", "approved", "executing", "completed", "archived"]
        },
        approvalDate: { bsonType: "date" },
        createdAt: { bsonType: "date" },
        updatedAt: { bsonType: "date" },
        createdBy: { bsonType: "objectId" }
      }
    }
  }
});

db.strategic_plans.createIndex({ planId: 1 }, { unique: true });
db.strategic_plans.createIndex({ planType: 1, fiscalYear: -1 });
db.strategic_plans.createIndex({ status: 1, startDate: -1 });
db.strategic_plans.createIndex({ owner: 1 });
db.strategic_plans.createIndex({ "keyObjectives.keyResults.status": 1 });
db.strategic_plans.createIndex({ "risks.impact": 1, "risks.probability": 1 });
db.strategic_plans.createIndex({ endDate: 1 });
db.strategic_plans.createIndex({
  title: "text",
  vision: "text",
  mission: "text",
  "keyObjectives.title": "text",
  "keyObjectives.description": "text",
  "risks.description": "text",
  "kpis.name": "text"
}, {
  name: "strategy_text_search",
  weights: {
    title: 10,
    vision: 5,
    mission: 5,
    "keyObjectives.title": 3,
    "keyObjectives.description": 2,
    "kpis.name": 2,
    "risks.description": 1
  }
});

// ============================================================================
// SALES OPPORTUNITIES
// ============================================================================

db.createCollection("sales_opportunities", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["opportunityId", "account", "title", "stage", "owner", "createdAt", "updatedAt"],
      properties: {
        opportunityId: { bsonType: "string" },
        account: { bsonType: "string" },
        title: { bsonType: "string" },
        description: { bsonType: "string" },
        value: { bsonType: "double" },
        currency: { bsonType: "string" },
        stage: {
          enum: [
            "prospect", "qualified_lead", "proposal",
            "negotiation", "won", "lost", "stalled"
          ]
        },
        createdDate: { bsonType: "date" },
        expectedCloseDate: { bsonType: "date" },
        actualCloseDate: { bsonType: "date" },
        owner: { bsonType: "objectId" },
        accountManager: { bsonType: "objectId" },
        contacts: {
          bsonType: "array",
          items: {
            bsonType: "object",
            properties: {
              name: { bsonType: "string" },
              email: { bsonType: "string" },
              phone: { bsonType: "string" },
              title: { bsonType: "string" },
              department: { bsonType: "string" },
              lastContact: { bsonType: "date" }
            }
          }
        },
        nextAction: { bsonType: "string" },
        nextActionDate: { bsonType: "date" },
        activities: {
          bsonType: "array",
          items: {
            bsonType: "object",
            required: ["type", "date"],
            properties: {
              type: { bsonType: "string" },
              date: { bsonType: "date" },
              notes: { bsonType: "string" },
              participant: { bsonType: "objectId" }
            }
          }
        },
        source: { enum: ["inbound", "outbound", "referral", "event", "partnership"] },
        priority: { enum: ["high", "medium", "low"] },
        probability: { bsonType: "double" },
        linkedDocuments: { bsonType: "array", items: { bsonType: "string" } },
        createdAt: { bsonType: "date" },
        updatedAt: { bsonType: "date" },
        createdBy: { bsonType: "objectId" }
      }
    }
  }
});

db.sales_opportunities.createIndex({ opportunityId: 1 }, { unique: true });
db.sales_opportunities.createIndex({ stage: 1, expectedCloseDate: 1 });
db.sales_opportunities.createIndex({ owner: 1, stage: 1 });
db.sales_opportunities.createIndex({ account: 1 });
db.sales_opportunities.createIndex({ value: -1 });
db.sales_opportunities.createIndex({ priority: 1, stage: 1 });
db.sales_opportunities.createIndex({ source: 1 });
db.sales_opportunities.createIndex({ "contacts.email": 1 });
db.sales_opportunities.createIndex({
  title: "text",
  account: "text",
  description: "text",
  nextAction: "text"
}, {
  name: "sales_text_search",
  weights: { account: 10, title: 5, description: 2, nextAction: 1 }
});

// ============================================================================
// KNOWLEDGE ARTICLES
// ============================================================================

db.createCollection("knowledge_articles", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["articleId", "title", "slug", "category", "content", "author", "status", "createdAt"],
      properties: {
        articleId: { bsonType: "string" },
        title: { bsonType: "string" },
        slug: { bsonType: "string" },
        category: { bsonType: "string" },
        content: { bsonType: "string" },
        summary: { bsonType: "string" },
        author: { bsonType: "objectId" },
        publishedDate: { bsonType: "date" },
        updatedDate: { bsonType: "date" },
        status: { enum: ["draft", "published", "archived"] },
        featured: { bsonType: "bool" },
        views: { bsonType: "int" },
        shares: { bsonType: "int" },
        likes: { bsonType: "int" },
        seo: {
          bsonType: "object",
          properties: {
            metaTitle: { bsonType: "string" },
            metaDescription: { bsonType: "string" },
            keywords: { bsonType: "array", items: { bsonType: "string" } }
          }
        },
        tags: { bsonType: "array", items: { bsonType: "string" } },
        relatedArticles: { bsonType: "array", items: { bsonType: "string" } },
        linkedResources: { bsonType: "array", items: { bsonType: "string" } },
        createdAt: { bsonType: "date" },
        createdBy: { bsonType: "objectId" }
      }
    }
  }
});

db.knowledge_articles.createIndex({ articleId: 1 }, { unique: true });
db.knowledge_articles.createIndex({ slug: 1 }, { unique: true });
db.knowledge_articles.createIndex({ category: 1, status: 1 });
db.knowledge_articles.createIndex({ author: 1 });
db.knowledge_articles.createIndex({ status: 1, publishedDate: -1 });
db.knowledge_articles.createIndex({ featured: -1, publishedDate: -1 });
db.knowledge_articles.createIndex({ tags: 1 });
db.knowledge_articles.createIndex({ views: -1 });
db.knowledge_articles.createIndex({
  title: "text",
  content: "text",
  summary: "text",
  "seo.keywords": "text",
  tags: "text"
}, {
  name: "article_text_search",
  weights: {
    title: 10,
    "seo.keywords": 5,
    tags: 3,
    summary: 2,
    content: 1
  }
});

// ============================================================================
// TTL INDEXES FOR EXPIRABLE DATA
// ============================================================================

// Auto-expire collaboration lock entries after 30 minutes
// (Used via application-level lock collection — not a core CMS collection,
//  but referenced in the Redis key structure for fallback)
