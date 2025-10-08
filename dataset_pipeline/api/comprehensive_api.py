#!/usr/bin/env python3
"""
Comprehensive Documentation and API
Complete API interface and documentation for the dataset pipeline.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIEndpoint(Enum):
    """Available API endpoints."""
    VALIDATE_CONVERSATION = "/api/v1/validate/conversation"
    EXPORT_DATASET = "/api/v1/export/dataset"
    GET_ANALYTICS = "/api/v1/analytics/dashboard"
    SUBMIT_FEEDBACK = "/api/v1/feedback/effectiveness"
    GET_QUALITY_METRICS = "/api/v1/quality/metrics"
    TRIGGER_MAINTENANCE = "/api/v1/maintenance/trigger"
    GET_SYSTEM_STATUS = "/api/v1/system/status"
    ADAPTIVE_LEARNING = "/api/v1/learning/feedback"


@dataclass
class APIDocumentation:
    """API endpoint documentation."""
    endpoint: str
    method: str
    description: str
    parameters: dict[str, Any]
    request_example: dict[str, Any]
    response_example: dict[str, Any]
    error_codes: list[dict[str, str]]
    rate_limits: str | None = None
    authentication: str | None = None


class ComprehensiveAPI:
    """
    Comprehensive API and documentation system for the dataset pipeline.
    """

    def __init__(self):
        """Initialize the comprehensive API system."""
        self.api_version = "1.0.0"
        self.base_url = "https://api.pixelated-empathy.com"
        self.documentation = self._generate_api_documentation()
        self.usage_examples = self._generate_usage_examples()
        self.integration_guides = self._generate_integration_guides()

    def _generate_api_documentation(self) -> dict[str, APIDocumentation]:
        """Generate comprehensive API documentation."""
        docs = {}

        # Conversation Validation API
        docs["validate_conversation"] = APIDocumentation(
            endpoint="/api/v1/validate/conversation",
            method="POST",
            description="Validate a therapeutic conversation using multi-tier quality assessment",
            parameters={
                "conversation": {
                    "type": "object",
                    "required": True,
                    "description": "Conversation object with id, content, turns, and metadata"
                },
                "validation_level": {
                    "type": "string",
                    "required": False,
                    "default": "comprehensive",
                    "options": ["basic", "standard", "comprehensive", "clinical"]
                },
                "include_recommendations": {
                    "type": "boolean",
                    "required": False,
                    "default": True
                }
            },
            request_example={
                "conversation": {
                    "id": "conv_001",
                    "content": "I understand you're feeling anxious. Let's explore some coping strategies.",
                    "turns": [
                        {"speaker": "user", "text": "I'm feeling anxious lately."},
                        {"speaker": "therapist", "text": "I understand. Let's explore coping strategies."}
                    ],
                    "metadata": {
                        "source": "professional",
                        "condition": "anxiety",
                        "approach": "CBT"
                    }
                },
                "validation_level": "comprehensive",
                "include_recommendations": True
            },
            response_example={
                "validation_id": "val_12345",
                "overall_quality_score": 0.85,
                "tier_assessment": "professional",
                "validation_results": {
                    "multi_tier_validation": {"passed": True, "score": 0.87},
                    "dsm5_accuracy": {"passed": True, "score": 0.83},
                    "safety_ethics": {"passed": True, "score": 0.91},
                    "effectiveness_prediction": {"score": 0.78, "confidence": "high"},
                    "coherence_validation": {"score": 0.82, "level": "moderately_coherent"}
                },
                "issues": [],
                "recommendations": [
                    "Consider adding more specific therapeutic techniques",
                    "Enhance empathetic responses"
                ],
                "processing_time_ms": 245
            },
            error_codes=[
                {"code": "400", "description": "Invalid conversation format"},
                {"code": "422", "description": "Validation failed - conversation quality too low"},
                {"code": "429", "description": "Rate limit exceeded"},
                {"code": "500", "description": "Internal validation error"}
            ],
            rate_limits="100 requests per minute",
            authentication="API key required"
        )

        # Dataset Export API
        docs["export_dataset"] = APIDocumentation(
            endpoint="/api/v1/export/dataset",
            method="POST",
            description="Export dataset in specified format with tiered access control",
            parameters={
                "export_config": {
                    "type": "object",
                    "required": True,
                    "description": "Export configuration including formats, tiers, and options"
                },
                "filters": {
                    "type": "object",
                    "required": False,
                    "description": "Optional filters for conversation selection"
                }
            },
            request_example={
                "export_config": {
                    "formats": ["json", "csv"],
                    "access_tiers": ["priority", "professional"],
                    "quality_threshold": 0.8,
                    "include_metadata": True,
                    "compress_output": True
                },
                "filters": {
                    "conditions": ["anxiety", "depression"],
                    "date_range": {
                        "start": "2025-01-01",
                        "end": "2025-08-10"
                    }
                }
            },
            response_example={
                "export_id": "exp_67890",
                "status": "completed",
                "export_metadata": [
                    {
                        "format": "json",
                        "tier": "priority",
                        "conversations": 1542,
                        "file_path": "/exports/v1/priority/conversations.json.zip",
                        "checksum": "sha256:abc123..."
                    }
                ],
                "total_conversations": 4626,
                "export_time_seconds": 45.2
            },
            error_codes=[
                {"code": "400", "description": "Invalid export configuration"},
                {"code": "403", "description": "Insufficient access permissions for requested tier"},
                {"code": "413", "description": "Export size exceeds limits"},
                {"code": "500", "description": "Export processing error"}
            ],
            rate_limits="10 exports per hour",
            authentication="API key with export permissions required"
        )

        # Analytics Dashboard API
        docs["get_analytics"] = APIDocumentation(
            endpoint="/api/v1/analytics/dashboard",
            method="GET",
            description="Get comprehensive analytics dashboard data",
            parameters={
                "time_range": {
                    "type": "string",
                    "required": False,
                    "default": "24h",
                    "options": ["1h", "24h", "7d", "30d"]
                },
                "include_trends": {
                    "type": "boolean",
                    "required": False,
                    "default": True
                }
            },
            request_example={},
            response_example={
                "dashboard_data": {
                    "total_conversations": 15420,
                    "quality_distribution": {
                        "excellent": 3084,
                        "good": 6168,
                        "acceptable": 4626,
                        "poor": 1542
                    },
                    "safety_metrics": {
                        "overall_safety_score": 0.91,
                        "compliance_rate": 0.94
                    },
                    "performance_trends": {
                        "quality_scores": [0.78, 0.79, 0.81, 0.82]
                    }
                },
                "summary_report": {
                    "performance_status": "🟢 EXCELLENT",
                    "key_insights": ["High quality conversations", "Excellent safety compliance"]
                }
            },
            error_codes=[
                {"code": "400", "description": "Invalid time range parameter"},
                {"code": "500", "description": "Analytics processing error"}
            ],
            rate_limits="60 requests per minute"
        )

        # System Status API
        docs["get_system_status"] = APIDocumentation(
            endpoint="/api/v1/system/status",
            method="GET",
            description="Get real-time system status and health metrics",
            parameters={},
            request_example={},
            response_example={
                "system_status": "healthy",
                "components": {
                    "validation_pipeline": {"status": "operational", "response_time_ms": 150},
                    "export_system": {"status": "operational", "queue_size": 2},
                    "analytics_engine": {"status": "operational", "last_update": "2025-08-10T07:30:00Z"},
                    "maintenance_system": {"status": "operational", "next_maintenance": "2025-08-10T12:00:00Z"}
                },
                "performance_metrics": {
                    "total_conversations_processed": 15420,
                    "average_processing_time_ms": 245,
                    "success_rate": 0.998,
                    "uptime_hours": 168.5
                },
                "alerts": []
            },
            error_codes=[
                {"code": "503", "description": "System temporarily unavailable"}
            ],
            rate_limits="120 requests per minute"
        )

        return docs

    def _generate_usage_examples(self) -> dict[str, dict[str, str]]:
        """Generate usage examples for different programming languages."""
        examples = {}

        # Python examples
        examples["python"] = {
            "validate_conversation": """
import requests
import json

# API configuration
API_KEY = "your_api_key_here"
BASE_URL = "https://api.pixelated-empathy.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Conversation to validate
conversation = {
    "id": "conv_001",
    "content": "I understand you're feeling anxious. Let's explore some coping strategies.",
    "turns": [
        {"speaker": "user", "text": "I'm feeling anxious lately."},
        {"speaker": "therapist", "text": "I understand. Let's explore coping strategies."}
    ],
    "metadata": {
        "source": "professional",
        "condition": "anxiety",
        "approach": "CBT"
    }
}

# Make API request
response = requests.post(
    f"{BASE_URL}/api/v1/validate/conversation",
    headers=headers,
    json={
        "conversation": conversation,
        "validation_level": "comprehensive",
        "include_recommendations": True
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Quality Score: {result['overall_quality_score']}")
    print(f"Recommendations: {result['recommendations']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
            """,

            "export_dataset": """
import requests
import time

# Export configuration
export_config = {
    "formats": ["json", "csv"],
    "access_tiers": ["priority", "professional"],
    "quality_threshold": 0.8,
    "include_metadata": True,
    "compress_output": True
}

# Start export
response = requests.post(
    f"{BASE_URL}/api/v1/export/dataset",
    headers=headers,
    json={"export_config": export_config}
)

if response.status_code == 200:
    export_data = response.json()
    export_id = export_data["export_id"]

    print(f"Export started: {export_id}")
    print(f"Total conversations: {export_data['total_conversations']}")

    # Download files
    for metadata in export_data["export_metadata"]:
        file_url = f"{BASE_URL}{metadata['file_path']}"
        file_response = requests.get(file_url, headers=headers)

        if file_response.status_code == 200:
            filename = f"export_{metadata['tier']}_{metadata['format']}.zip"
            with open(filename, 'wb') as f:
                f.write(file_response.content)
            print(f"Downloaded: {filename}")
            """
        }

        # JavaScript examples
        examples["javascript"] = {
            "validate_conversation": """
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.pixelated-empathy.com';

const headers = {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
};

const conversation = {
    id: 'conv_001',
    content: "I understand you're feeling anxious. Let's explore some coping strategies.",
    turns: [
        { speaker: 'user', text: "I'm feeling anxious lately." },
        { speaker: 'therapist', text: "I understand. Let's explore coping strategies." }
    ],
    metadata: {
        source: 'professional',
        condition: 'anxiety',
        approach: 'CBT'
    }
};

async function validateConversation() {
    try {
        const response = await axios.post(
            `${BASE_URL}/api/v1/validate/conversation`,
            {
                conversation: conversation,
                validation_level: 'comprehensive',
                include_recommendations: true
            },
            { headers }
        );

        console.log(`Quality Score: ${response.data.overall_quality_score}`);
        console.log(`Recommendations: ${response.data.recommendations}`);
    } catch (error) {
        console.error(`Error: ${error.response.status} - ${error.response.data}`);
    }
}

validateConversation();
            """
        }

        # cURL examples
        examples["curl"] = {
            "validate_conversation": """
curl -X POST "https://api.pixelated-empathy.com/api/v1/validate/conversation" \\
  -H "Authorization: Bearer your_api_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "conversation": {
      "id": "conv_001",
      "content": "I understand you'\''re feeling anxious. Let'\''s explore some coping strategies.",
      "turns": [
        {"speaker": "user", "text": "I'\''m feeling anxious lately."},
        {"speaker": "therapist", "text": "I understand. Let'\''s explore coping strategies."}
      ],
      "metadata": {
        "source": "professional",
        "condition": "anxiety",
        "approach": "CBT"
      }
    },
    "validation_level": "comprehensive",
    "include_recommendations": true
  }'
            """
        }

        return examples

    def _generate_integration_guides(self) -> dict[str, str]:
        """Generate integration guides for different use cases."""
        guides = {}

        guides["getting_started"] = """
# Getting Started with Pixelated Empathy Dataset Pipeline API

## 1. Authentication
All API requests require authentication using an API key:
```
Authorization: Bearer your_api_key_here
```

## 2. Base URL
All API endpoints are available at:
```
https://api.pixelated-empathy.com
```

## 3. Rate Limits
- Validation API: 100 requests per minute
- Export API: 10 exports per hour
- Analytics API: 60 requests per minute
- System Status: 120 requests per minute

## 4. Response Format
All responses are in JSON format with consistent structure:
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "request_id": "req_12345",
    "processing_time_ms": 245,
    "api_version": "1.0.0"
  }
}
```

## 5. Error Handling
Errors follow standard HTTP status codes:
- 400: Bad Request - Invalid parameters
- 401: Unauthorized - Invalid API key
- 403: Forbidden - Insufficient permissions
- 422: Unprocessable Entity - Validation failed
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server error
        """

        guides["therapeutic_validation"] = """
# Therapeutic Conversation Validation Integration

## Overview
The validation API provides comprehensive assessment of therapeutic conversations using multiple validation layers.

## Validation Levels
1. **Basic**: Quality score and tier assessment
2. **Standard**: Includes safety and coherence validation
3. **Comprehensive**: Full multi-tier validation with recommendations
4. **Clinical**: Includes DSM-5 accuracy and effectiveness prediction

## Integration Steps

### 1. Prepare Conversation Data
```python
conversation = {
    "id": "unique_conversation_id",
    "content": "Full conversation text",
    "turns": [
        {"speaker": "user", "text": "User message"},
        {"speaker": "therapist", "text": "Therapist response"}
    ],
    "metadata": {
        "source": "professional",  # priority, professional, cot, reddit, research
        "condition": "anxiety",    # mental health condition
        "approach": "CBT",         # therapeutic approach
        "timestamp": "2025-08-10T07:30:00Z"
    }
}
```

### 2. Submit for Validation
```python
response = requests.post(
    "https://api.pixelated-empathy.com/api/v1/validate/conversation",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "conversation": conversation,
        "validation_level": "comprehensive"
    }
)
```

### 3. Process Results
```python
if response.status_code == 200:
    result = response.json()

    # Overall quality assessment
    quality_score = result["overall_quality_score"]
    tier = result["tier_assessment"]

    # Detailed validation results
    validations = result["validation_results"]
    safety_score = validations["safety_ethics"]["score"]
    effectiveness = validations["effectiveness_prediction"]["score"]

    # Improvement recommendations
    recommendations = result["recommendations"]
```

## Best Practices
- Validate conversations before adding to training datasets
- Use comprehensive validation for clinical applications
- Implement retry logic for rate limit handling
- Cache validation results to avoid duplicate requests
        """

        guides["dataset_export"] = """
# Dataset Export Integration Guide

## Overview
The export API provides secure, tiered access to validated therapeutic conversations in multiple formats.

## Access Tiers
- **Priority**: Highest quality clinical conversations (99% accuracy)
- **Professional**: Professional therapist conversations (95% accuracy)
- **CoT**: Chain-of-thought reasoning conversations (90% accuracy)
- **Reddit**: Community-sourced conversations (85% accuracy)
- **Research**: Research paper conversations (80% accuracy)
- **Archive**: Historical conversations (75% accuracy)

## Export Formats
- **JSON**: Structured conversation data
- **CSV**: Tabular format for analysis
- **Parquet**: Optimized for big data processing
- **HuggingFace**: Ready for transformer training
- **JSONL**: Line-delimited JSON for streaming

## Integration Example

### 1. Configure Export
```python
export_config = {
    "formats": ["json", "huggingface"],
    "access_tiers": ["priority", "professional"],
    "quality_threshold": 0.8,
    "include_metadata": True,
    "compress_output": True,
    "max_conversations_per_file": 10000
}

filters = {
    "conditions": ["anxiety", "depression"],
    "approaches": ["CBT", "DBT"],
    "date_range": {
        "start": "2025-01-01",
        "end": "2025-08-10"
    }
}
```

### 2. Request Export
```python
response = requests.post(
    "https://api.pixelated-empathy.com/api/v1/export/dataset",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "export_config": export_config,
        "filters": filters
    }
)
```

### 3. Download Files
```python
if response.status_code == 200:
    export_data = response.json()

    for metadata in export_data["export_metadata"]:
        # Download each export file
        file_url = f"https://api.pixelated-empathy.com{metadata['file_path']}"
        file_response = requests.get(file_url, headers=headers)

        # Verify checksum
        import hashlib
        actual_checksum = hashlib.sha256(file_response.content).hexdigest()
        expected_checksum = metadata["checksum"].split(":")[1]

        if actual_checksum == expected_checksum:
            # Save file
            with open(f"dataset_{metadata['tier']}.{metadata['format']}", 'wb') as f:
                f.write(file_response.content)
```

## Security Considerations
- API keys provide tier-specific access
- All downloads are logged and audited
- Checksums ensure data integrity
- Rate limits prevent abuse
        """

        return guides

    def generate_openapi_spec(self) -> dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Pixelated Empathy Dataset Pipeline API",
                "version": self.api_version,
                "description": "Comprehensive API for therapeutic conversation validation, dataset export, and analytics",
                "contact": {
                    "name": "API Support",
                    "email": "api-support@pixelated-empathy.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": self.base_url,
                    "description": "Production server"
                }
            ],
            "security": [
                {"BearerAuth": []}
            ],
            "components": {
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                },
                "schemas": {
                    "Conversation": {
                        "type": "object",
                        "required": ["id", "content", "turns"],
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "turns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {"type": "string"},
                                        "text": {"type": "string"}
                                    }
                                }
                            },
                            "metadata": {"type": "object"}
                        }
                    },
                    "ValidationResult": {
                        "type": "object",
                        "properties": {
                            "validation_id": {"type": "string"},
                            "overall_quality_score": {"type": "number"},
                            "tier_assessment": {"type": "string"},
                            "validation_results": {"type": "object"},
                            "issues": {"type": "array", "items": {"type": "string"}},
                            "recommendations": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "paths": {}
        }

        # Add paths from documentation
        for _endpoint_name, doc in self.documentation.items():
            path = doc.endpoint
            method = doc.method.lower()

            if path not in spec["paths"]:
                spec["paths"][path] = {}

            spec["paths"][path][method] = {
                "summary": doc.description,
                "parameters": [],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "example": doc.response_example
                            }
                        }
                    }
                }
            }

            # Add error responses
            for error in doc.error_codes:
                spec["paths"][path][method]["responses"][error["code"]] = {
                    "description": error["description"]
                }

        return spec

    def export_documentation(self, output_dir: str):
        """Export complete documentation to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export API documentation
        api_docs = {
            "api_version": self.api_version,
            "base_url": self.base_url,
            "endpoints": {
                name: {
                    "endpoint": doc.endpoint,
                    "method": doc.method,
                    "description": doc.description,
                    "parameters": doc.parameters,
                    "request_example": doc.request_example,
                    "response_example": doc.response_example,
                    "error_codes": doc.error_codes,
                    "rate_limits": doc.rate_limits,
                    "authentication": doc.authentication
                }
                for name, doc in self.documentation.items()
            }
        }

        with open(output_path / "api_documentation.json", "w") as f:
            json.dump(api_docs, f, indent=2)

        # Export usage examples
        with open(output_path / "usage_examples.json", "w") as f:
            json.dump(self.usage_examples, f, indent=2)

        # Export integration guides
        for guide_name, guide_content in self.integration_guides.items():
            with open(output_path / f"{guide_name}_guide.md", "w") as f:
                f.write(guide_content)

        # Export OpenAPI specification
        openapi_spec = self.generate_openapi_spec()
        with open(output_path / "openapi.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)

        # Create README
        readme_content = f"""# Pixelated Empathy Dataset Pipeline API

Version: {self.api_version}
Base URL: {self.base_url}

## Overview
This API provides comprehensive access to therapeutic conversation validation, dataset export, analytics, and system management capabilities.

## Documentation Files
- `api_documentation.json` - Complete API endpoint documentation
- `usage_examples.json` - Code examples in multiple languages
- `getting_started_guide.md` - Quick start guide
- `therapeutic_validation_guide.md` - Validation integration guide
- `dataset_export_guide.md` - Export integration guide
- `openapi.json` - OpenAPI 3.0 specification

## Quick Start
1. Obtain API key from the developer portal
2. Review the getting started guide
3. Try the validation API with sample data
4. Explore export capabilities for your use case

## Support
- Documentation: https://docs.pixelated-empathy.com
- API Support: api-support@pixelated-empathy.com
- Developer Portal: https://developers.pixelated-empathy.com

## Rate Limits
- Validation: 100 requests/minute
- Export: 10 exports/hour
- Analytics: 60 requests/minute
- System Status: 120 requests/minute
        """

        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)

        logger.info(f"Complete documentation exported to {output_path}")

    def get_api_summary(self) -> dict[str, Any]:
        """Get API summary information."""
        return {
            "api_version": self.api_version,
            "base_url": self.base_url,
            "total_endpoints": len(self.documentation),
            "available_endpoints": list(self.documentation.keys()),
            "supported_languages": list(self.usage_examples.keys()),
            "integration_guides": list(self.integration_guides.keys()),
            "documentation_generated": datetime.now(timezone.utc).isoformat()
        }


def main():
    """Example usage of the ComprehensiveAPI system."""
    api_system = ComprehensiveAPI()


    # Get API summary
    summary = api_system.get_api_summary()
    for _endpoint in summary["available_endpoints"]:
        pass


    # Export documentation
    api_system.export_documentation("./api_documentation")

    # Show sample endpoint documentation
    api_system.documentation["validate_conversation"]


if __name__ == "__main__":
    main()
