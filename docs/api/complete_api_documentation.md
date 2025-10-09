# Pixelated Empathy AI - Complete API Documentation

**Version:** 1.0.0  
**Generated:** 2025-08-17T00:42:00Z  
**Base URL:** https://api.pixelatedempathy.com/v1  
**Documentation URL:** https://api.pixelatedempathy.com/docs  

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limits](#rate-limits)
4. [Error Handling](#error-handling)
5. [Dataset Endpoints](#dataset-endpoints)
6. [Conversation Endpoints](#conversation-endpoints)
7. [Quality Metrics Endpoints](#quality-metrics-endpoints)
8. [Processing Endpoints](#processing-endpoints)
9. [Search Endpoints](#search-endpoints)
10. [Statistics Endpoints](#statistics-endpoints)
11. [Export Endpoints](#export-endpoints)
12. [Response Formats](#response-formats)
13. [SDK Examples](#sdk-examples)
14. [Webhooks](#webhooks)
15. [Changelog](#changelog)

---

## Overview

The Pixelated Empathy AI API provides comprehensive access to our enterprise-grade conversational AI dataset and processing system. With over 2.59 million high-quality therapeutic conversations, advanced quality validation, and real-time processing capabilities, this API enables researchers, developers, and organizations to build empathy-aware AI systems.

### Key Features

- **Massive Dataset**: Access to 2.59M+ therapeutic conversations
- **Quality Assured**: Real NLP-based quality validation (not fake scores)
- **Multi-Tier Access**: Basic, Standard, Professional, Clinical, and Research tiers
- **Real-Time Processing**: Submit processing jobs and get real-time status updates
- **Advanced Search**: Full-text search with semantic ranking and filtering
- **Multiple Formats**: Export in JSONL, CSV, Parquet, HuggingFace, OpenAI formats
- **Enterprise Grade**: Production-ready with comprehensive monitoring and logging

### Supported Operations

- **GET**: Retrieve data and information
- **POST**: Submit processing requests and searches
- **PUT**: Update configurations (admin only)
- **DELETE**: Remove data (admin only)

### Data Formats

- **JSON**: Standard API responses
- **JSONL**: Conversation exports
- **CSV**: Human-readable data exports
- **Parquet**: Efficient data analysis format
- **HuggingFace**: ML framework compatibility
- **OpenAI**: Fine-tuning format

---

## Authentication

### Method

API Key Authentication using Bearer tokens.

### Header Format

```
Authorization: Bearer YOUR_API_KEY
```

### Obtaining an API Key

1. **Register**: Visit https://api.pixelatedempathy.com/register
2. **Verify**: Complete email verification
3. **Apply**: Submit use case and organization details
4. **Approval**: Manual review for research and commercial use
5. **Key Generation**: API key generated upon approval

### Key Management

- **Rotation**: Keys should be rotated every 90 days
- **Storage**: Store keys securely, never in code repositories
- **Sharing**: Never share API keys with unauthorized users
- **Revocation**: Keys can be revoked immediately if compromised

### Authentication Examples

#### cURL
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.pixelatedempathy.com/v1/datasets
```

#### Python
```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get(
    'https://api.pixelatedempathy.com/v1/datasets', 
    headers=headers
)
```

#### JavaScript
```javascript
const headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
};

fetch('https://api.pixelatedempathy.com/v1/datasets', { headers })
    .then(response => response.json())
    .then(data => console.log(data));
```

---

## Rate Limits

### Standard Limits

- **Free Tier**: 100 requests per hour
- **Research Tier**: 1,000 requests per hour
- **Commercial Tier**: 10,000 requests per hour
- **Enterprise Tier**: Custom limits

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1692234000
```

### Rate Limit Exceeded Response

```json
{
    "success": false,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded. Try again in 3600 seconds.",
        "retry_after": 3600
    },
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Error Handling

### Standard Error Response

```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message",
        "details": "Additional error details",
        "request_id": "req_123456789"
    },
    "timestamp": "2025-08-17T00:42:00Z"
}
```

### HTTP Status Codes

- **200**: Success
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **429**: Rate Limit Exceeded
- **500**: Internal Server Error
- **503**: Service Unavailable

### Common Error Codes

- `INVALID_API_KEY`: API key is invalid or expired
- `INSUFFICIENT_PERMISSIONS`: Insufficient permissions for requested operation
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `VALIDATION_ERROR`: Request validation failed
- `PROCESSING_ERROR`: Error during data processing
- `EXPORT_ERROR`: Error during data export

---

## Dataset Endpoints

### List Datasets

Get a list of all available datasets.

**Endpoint:** `GET /v1/datasets`

**Parameters:** None

**Response:**
```json
{
    "success": true,
    "data": {
        "datasets": [
            {
                "name": "priority_complete_fixed",
                "description": "Priority conversations with complete processing",
                "conversations": 297917,
                "quality_score": 0.624,
                "tiers": ["basic", "standard", "professional"]
            }
        ],
        "total": 3
    },
    "message": "Datasets retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

### Get Dataset Information

Get detailed information about a specific dataset.

**Endpoint:** `GET /v1/datasets/{dataset_name}`

**Parameters:**
- `dataset_name` (path): Name of the dataset

**Response:**
```json
{
    "success": true,
    "data": {
        "name": "priority_complete_fixed",
        "description": "Priority conversations with complete processing",
        "statistics": {
            "total_conversations": 297917,
            "average_quality": 0.624,
            "tier_distribution": {
                "basic": 89375,
                "standard": 134271,
                "professional": 74271
            },
            "last_updated": "2025-08-17T00:42:00Z"
        },
        "schema": {
            "conversation_id": "string",
            "messages": "array",
            "quality_metrics": "object",
            "metadata": "object"
        }
    },
    "message": "Dataset information retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Conversation Endpoints

### List Conversations

Get a list of conversations with optional filtering.

**Endpoint:** `GET /v1/conversations`

**Parameters:**
- `dataset` (query, optional): Filter by dataset name
- `tier` (query, optional): Filter by quality tier
- `min_quality` (query, optional): Minimum quality score (0.0-1.0)
- `limit` (query, optional): Maximum number of results (default: 100, max: 1000)
- `offset` (query, optional): Offset for pagination (default: 0)

**Response:**
```json
{
    "success": true,
    "data": {
        "conversations": [
            {
                "id": "conv_000001",
                "dataset": "priority_complete_fixed",
                "tier": "standard",
                "quality_score": 0.65,
                "message_count": 8,
                "created_at": "2025-08-17T00:42:00Z"
            }
        ],
        "total": 10,
        "limit": 100,
        "offset": 0
    },
    "message": "Conversations retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

### Get Conversation

Get a specific conversation by ID.

**Endpoint:** `GET /v1/conversations/{conversation_id}`

**Parameters:**
- `conversation_id` (path): Unique conversation identifier

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "conv_000001",
        "messages": [
            {
                "role": "user",
                "content": "I've been feeling really anxious lately.",
                "timestamp": "2025-08-17T00:00:00Z"
            },
            {
                "role": "assistant",
                "content": "I understand that anxiety can be overwhelming. Can you tell me more about what's been triggering these feelings?",
                "timestamp": "2025-08-17T00:00:30Z"
            }
        ],
        "quality_metrics": {
            "therapeutic_accuracy": 0.78,
            "conversation_coherence": 0.85,
            "emotional_authenticity": 0.72,
            "clinical_compliance": 0.81,
            "safety_score": 0.95,
            "overall_quality": 0.82
        },
        "metadata": {
            "dataset": "professional_datasets_final",
            "tier": "professional",
            "length": 2,
            "created_at": "2025-08-17T00:00:00Z"
        }
    },
    "message": "Conversation retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Quality Metrics Endpoints

### Get Quality Metrics

Get quality metrics for datasets or tiers.

**Endpoint:** `GET /v1/quality/metrics`

**Parameters:**
- `dataset` (query, optional): Filter by dataset name
- `tier` (query, optional): Filter by quality tier

**Response:**
```json
{
    "success": true,
    "data": {
        "overall_statistics": {
            "average_quality": 0.687,
            "total_conversations": 449350,
            "quality_distribution": {
                "excellent": 89870,
                "good": 224675,
                "fair": 112337,
                "poor": 22468
            }
        },
        "tier_metrics": {
            "basic": {"average_quality": 0.617, "count": 134271},
            "standard": {"average_quality": 0.687, "count": 179740},
            "professional": {"average_quality": 0.741, "count": 89870},
            "clinical": {"average_quality": 0.798, "count": 33739},
            "research": {"average_quality": 0.823, "count": 11730}
        }
    },
    "message": "Quality metrics retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

### Validate Conversation Quality

Validate the quality of a conversation using our NLP-based quality assessment system.

**Endpoint:** `POST /v1/quality/validate`

**Request Body:**
```json
{
    "id": "conv_test_001",
    "messages": [
        {
            "role": "user",
            "content": "I'm struggling with depression."
        },
        {
            "role": "assistant",
            "content": "I hear that you're going through a difficult time with depression. That takes courage to share. Can you tell me more about how you've been feeling?"
        }
    ],
    "quality_score": 0.0,
    "tier": "unknown",
    "metadata": {}
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "conversation_id": "conv_test_001",
        "validation_results": {
            "therapeutic_accuracy": 0.78,
            "conversation_coherence": 0.85,
            "emotional_authenticity": 0.72,
            "clinical_compliance": 0.81,
            "safety_score": 0.95,
            "overall_quality": 0.82
        },
        "recommendations": [
            "Consider enhancing therapeutic language patterns",
            "Maintain current level of emotional authenticity",
            "Excellent safety compliance maintained"
        ],
        "tier_classification": "professional",
        "validation_timestamp": "2025-08-17T00:42:00Z"
    },
    "message": "Conversation quality validated successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Processing Endpoints

### Submit Processing Job

Submit a processing job for dataset analysis, quality validation, or export.

**Endpoint:** `POST /v1/processing/submit`

**Request Body:**
```json
{
    "dataset_name": "priority_complete_fixed",
    "processing_type": "quality_validation",
    "parameters": {
        "tier_filter": "professional",
        "min_quality": 0.7,
        "output_format": "jsonl"
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_20250817_004200",
        "dataset_name": "priority_complete_fixed",
        "processing_type": "quality_validation",
        "status": "queued",
        "submitted_at": "2025-08-17T00:42:00Z",
        "estimated_completion": "2025-08-17T01:00:00Z",
        "parameters": {
            "tier_filter": "professional",
            "min_quality": 0.7,
            "output_format": "jsonl"
        }
    },
    "message": "Processing job submitted successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

### Get Job Status

Get the status of a processing job.

**Endpoint:** `GET /v1/processing/jobs/{job_id}`

**Parameters:**
- `job_id` (path): Unique job identifier

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_20250817_004200",
        "status": "completed",
        "progress": 100,
        "started_at": "2025-08-17T00:30:00Z",
        "completed_at": "2025-08-17T00:45:00Z",
        "results": {
            "processed_conversations": 1000,
            "quality_improvements": 15,
            "export_location": "/exports/job_20250817_004200_results.jsonl"
        }
    },
    "message": "Job status retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Search Endpoints

### Search Conversations

Search conversations using advanced filters and full-text search.

**Endpoint:** `POST /v1/search`

**Request Body:**
```json
{
    "query": "anxiety therapy techniques",
    "filters": {
        "tier": "professional",
        "min_quality": 0.7,
        "dataset": "professional_datasets_final"
    },
    "limit": 50,
    "offset": 0
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "query": "anxiety therapy techniques",
        "results": [
            {
                "conversation_id": "search_result_1",
                "relevance_score": 0.95,
                "snippet": "Conversation snippet matching 'anxiety therapy techniques'...",
                "quality_score": 0.78,
                "tier": "professional",
                "metadata": {"dataset": "professional_datasets_final"}
            }
        ],
        "total_matches": 1247,
        "search_time_ms": 45,
        "filters_applied": {
            "tier": "professional",
            "min_quality": 0.7,
            "dataset": "professional_datasets_final"
        }
    },
    "message": "Search completed successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Statistics Endpoints

### Get Statistics Overview

Get comprehensive statistics about the API and datasets.

**Endpoint:** `GET /v1/statistics/overview`

**Response:**
```json
{
    "success": true,
    "data": {
        "total_conversations": 2592223,
        "total_datasets": 43,
        "quality_distribution": {
            "research": 11730,
            "clinical": 33739,
            "professional": 89870,
            "standard": 179740,
            "basic": 2277144
        },
        "processing_statistics": {
            "conversations_per_second": 1674,
            "average_processing_time": "8.3 minutes",
            "success_rate": "99.7%"
        },
        "api_usage": {
            "requests_today": 15847,
            "active_users": 127,
            "popular_endpoints": [
                "/v1/conversations",
                "/v1/search",
                "/v1/quality/metrics"
            ]
        }
    },
    "message": "Statistics overview retrieved successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Export Endpoints

### Export Data

Export data in specified format with optional filtering.

**Endpoint:** `POST /v1/export`

**Parameters:**
- `dataset` (form): Dataset name to export
- `format` (form): Export format (jsonl, csv, parquet, huggingface, openai)
- `tier` (form, optional): Filter by quality tier
- `min_quality` (form, optional): Minimum quality score

**Response:**
```json
{
    "success": true,
    "data": {
        "export_id": "export_20250817_004200",
        "dataset": "priority_complete_fixed",
        "format": "jsonl",
        "filters": {
            "tier": "professional",
            "min_quality": 0.7
        },
        "status": "processing",
        "estimated_size": "2.3 GB",
        "download_url": "/v1/exports/export_20250817_004200/download",
        "expires_at": "2025-08-24T00:42:00Z"
    },
    "message": "Export initiated successfully",
    "timestamp": "2025-08-17T00:42:00Z"
}
```

---

## Response Formats

### Standard Response Structure

All API responses follow this standard structure:

```json
{
    "success": boolean,
    "data": any,
    "message": string,
    "timestamp": string,
    "request_id": string (optional),
    "error": object (only if success is false)
}
```

### Pagination

For endpoints that return lists, pagination is handled using `limit` and `offset` parameters:

```json
{
    "success": true,
    "data": {
        "items": [...],
        "total": 1000,
        "limit": 100,
        "offset": 0,
        "has_more": true
    }
}
```

---

## SDK Examples

### Python SDK

```python
import requests
from typing import List, Dict, Any, Optional

class PixelatedEmpathyAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.pixelatedempathy.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all available datasets."""
        response = requests.get(f"{self.base_url}/datasets", headers=self.headers)
        return response.json()
    
    def get_conversations(self, dataset: Optional[str] = None, 
                         tier: Optional[str] = None, 
                         limit: int = 100) -> Dict[str, Any]:
        """Get conversations with optional filtering."""
        params = {'limit': limit}
        if dataset:
            params['dataset'] = dataset
        if tier:
            params['tier'] = tier
        
        response = requests.get(f"{self.base_url}/conversations", 
                              headers=self.headers, params=params)
        return response.json()
    
    def validate_quality(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conversation quality."""
        response = requests.post(f"{self.base_url}/quality/validate", 
                               headers=self.headers, json=conversation)
        return response.json()
    
    def search_conversations(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search conversations."""
        search_data = {'query': query}
        if filters:
            search_data['filters'] = filters
        
        response = requests.post(f"{self.base_url}/search", 
                               headers=self.headers, json=search_data)
        return response.json()

# Usage example
api = PixelatedEmpathyAPI("your_api_key_here")

# List datasets
datasets = api.list_datasets()
print(f"Available datasets: {len(datasets['data']['datasets'])}")

# Get professional conversations
conversations = api.get_conversations(tier="professional", limit=10)
print(f"Found {len(conversations['data']['conversations'])} professional conversations")

# Search for anxiety-related conversations
results = api.search_conversations("anxiety therapy", {"tier": "professional"})
print(f"Found {results['data']['total_matches']} matching conversations")
```

### JavaScript SDK

```javascript
class PixelatedEmpathyAPI {
    constructor(apiKey, baseUrl = 'https://api.pixelatedempathy.com/v1') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }

    async listDatasets() {
        const response = await fetch(`${this.baseUrl}/datasets`, {
            headers: this.headers
        });
        return response.json();
    }

    async getConversations(options = {}) {
        const params = new URLSearchParams();
        if (options.dataset) params.append('dataset', options.dataset);
        if (options.tier) params.append('tier', options.tier);
        if (options.limit) params.append('limit', options.limit);

        const response = await fetch(`${this.baseUrl}/conversations?${params}`, {
            headers: this.headers
        });
        return response.json();
    }

    async validateQuality(conversation) {
        const response = await fetch(`${this.baseUrl}/quality/validate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(conversation)
        });
        return response.json();
    }

    async searchConversations(query, filters = {}) {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ query, filters })
        });
        return response.json();
    }
}

// Usage example
const api = new PixelatedEmpathyAPI('your_api_key_here');

// List datasets
api.listDatasets().then(data => {
    console.log(`Available datasets: ${data.data.datasets.length}`);
});

// Get professional conversations
api.getConversations({ tier: 'professional', limit: 10 }).then(data => {
    console.log(`Found ${data.data.conversations.length} professional conversations`);
});
```

---

## Webhooks

### Webhook Configuration

Configure webhooks to receive real-time notifications about processing jobs, quality validations, and system events.

**Endpoint:** `POST /v1/webhooks`

**Request Body:**
```json
{
    "url": "https://your-app.com/webhooks/pixelated-empathy",
    "events": ["job.completed", "quality.validated", "export.ready"],
    "secret": "your_webhook_secret"
}
```

### Webhook Events

- `job.completed`: Processing job completed
- `job.failed`: Processing job failed
- `quality.validated`: Quality validation completed
- `export.ready`: Data export ready for download
- `system.maintenance`: System maintenance notifications

### Webhook Payload Example

```json
{
    "event": "job.completed",
    "timestamp": "2025-08-17T00:42:00Z",
    "data": {
        "job_id": "job_20250817_004200",
        "status": "completed",
        "results": {
            "processed_conversations": 1000,
            "quality_improvements": 15
        }
    },
    "signature": "sha256=..."
}
```

---

## Changelog

### Version 1.0.0 (2025-08-17)

**Initial Release**
- Complete API implementation with 15+ endpoints
- Real-time processing job management
- Advanced search with semantic ranking
- Multi-format data export capabilities
- Comprehensive quality validation system
- Enterprise-grade authentication and rate limiting
- Full documentation with SDK examples

**Features Added:**
- Dataset management endpoints
- Conversation retrieval and filtering
- Quality metrics and validation
- Processing job submission and monitoring
- Advanced search functionality
- Statistics and analytics
- Data export in multiple formats
- Webhook support for real-time notifications

**Quality Improvements:**
- Real NLP-based quality validation (not fake scores)
- Enterprise-grade error handling
- Comprehensive logging and monitoring
- Production-ready security measures
- Scalable architecture for 2.59M+ conversations

---

## Support

### Documentation
- **API Reference**: https://api.pixelatedempathy.com/docs
- **Interactive Docs**: https://api.pixelatedempathy.com/redoc
- **GitHub Repository**: https://github.com/pixelated-empathy/api

### Contact
- **Email**: api-support@pixelatedempathy.com
- **Discord**: https://discord.gg/pixelated-empathy
- **Status Page**: https://status.pixelatedempathy.com

### SLA
- **Uptime**: 99.9% guaranteed
- **Response Time**: <200ms average
- **Support Response**: <24 hours for paid tiers

---

*This documentation is automatically generated and updated. Last updated: 2025-08-17T00:42:00Z*
