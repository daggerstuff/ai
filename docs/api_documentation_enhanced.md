# Pixelated Empathy AI - Enhanced API Documentation

**Version:** 2.0.0  
**Updated:** 2025-08-12T20:56:05.349119  
**Base URL:** https://api.pixelatedempathy.com/v1

## 🚀 Quick Start

### Authentication
```bash
# Get API key from dashboard
export API_KEY="your_api_key_here"

# Or use JWT token
export JWT_TOKEN="your_jwt_token_here"
```

### Basic Usage
```bash
# Health check
curl -X GET "https://api.pixelatedempathy.com/v1/" \
  -H "X-API-Key: $API_KEY"

# Transcribe audio
curl -X POST "https://api.pixelatedempathy.com/v1/transcribe" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "model": "whisper-1"
  }'
```

## 📋 API Endpoints

### Health Check
**GET** `/`

Returns API health status and version information.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-08-12T20:00:00Z",
  "uptime": 3600
}
```

### Transcription
**POST** `/transcribe`

Transcribe audio content using AI models.

**Request Body:**
```json
{
  "audio_url": "string",
  "language": "string (optional, default: 'en')",
  "model": "string (optional, default: 'whisper-1')",
  "options": {
    "timestamps": true,
    "word_level": false,
    "speaker_detection": false
  }
}
```

**Response:**
```json
{
  "transcript": "string",
  "confidence": 0.95,
  "language": "en",
  "duration": 120.5,
  "timestamps": [
    {"start": 0.0, "end": 5.2, "text": "Hello world"}
  ]
}
```

**Example:**
```python
import requests

response = requests.post(
    "https://api.pixelatedempathy.com/v1/transcribe",
    headers={"X-API-Key": "your_api_key"},
    json={
        "audio_url": "https://example.com/audio.mp3",
        "language": "en",
        "options": {"timestamps": True}
    }
)
print(response.json())
```

### Pipeline Jobs
**POST** `/pipeline/jobs`

Create a new pipeline processing job.

**Request Body:**
```json
{
  "job_type": "transcription|analysis|processing",
  "input_data": {
    "url": "string",
    "format": "string"
  },
  "config": {
    "language": "string",
    "model": "string",
    "options": {}
  }
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "created_at": "2025-08-12T20:00:00Z",
  "estimated_completion": "2025-08-12T20:05:00Z"
}
```

**GET** `/pipeline/jobs`

List all jobs for the authenticated user.

**Response:**
```json
[
  {
    "job_id": "uuid",
    "job_type": "transcription",
    "status": "completed",
    "created_at": "2025-08-12T20:00:00Z",
    "completed_at": "2025-08-12T20:03:00Z",
    "result_url": "https://api.example.com/results/uuid"
  }
]
```

**GET** `/pipeline/jobs/{job_id}`

Get specific job details.

**DELETE** `/pipeline/jobs/{job_id}`

Cancel a running job.

### Data Access
**GET** `/data/{data_type}/latest`

Get latest data of specified type.

**GET** `/data/{data_type}/files`

List available data files.

## 🔐 Authentication

### API Key Authentication
```bash
curl -H "X-API-Key: your_api_key" https://api.pixelatedempathy.com/v1/
```

### JWT Token Authentication
```bash
curl -H "Authorization: Bearer your_jwt_token" https://api.pixelatedempathy.com/v1/
```

### Role-Based Access Control

| Role | Permissions |
|------|-------------|
| **readonly** | Read access to data and job status |
| **standard** | Create jobs, read data, manage own jobs |
| **premium** | Higher rate limits, priority processing |
| **admin** | Full system access, user management |

## ⚡ Rate Limits

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|----------------|---------------|--------------|
| **Free** | 60 | 1,000 | 10,000 |
| **Standard** | 120 | 5,000 | 50,000 |
| **Premium** | 300 | 15,000 | 150,000 |
| **Enterprise** | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1692742800
```

## 🚨 Error Handling

### HTTP Status Codes
- **200** - Success
- **201** - Created
- **202** - Accepted (async processing)
- **400** - Bad Request
- **401** - Unauthorized
- **403** - Forbidden
- **404** - Not Found
- **422** - Validation Error
- **429** - Rate Limited
- **500** - Internal Server Error

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "audio_url",
      "issue": "URL is not accessible"
    },
    "request_id": "uuid"
  }
}
```

## 📚 SDK Examples

### Python
```python
import requests
from typing import Optional, Dict, Any

class PixelatedEmpathyAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.pixelatedempathy.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
    
    def transcribe(self, audio_url: str, language: str = "en", **options) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/transcribe",
            json={
                "audio_url": audio_url,
                "language": language,
                "options": options
            }
        )
        response.raise_for_status()
        return response.json()
    
    def create_job(self, job_type: str, input_data: Dict, config: Dict) -> str:
        response = self.session.post(
            f"{self.base_url}/pipeline/jobs",
            json={
                "job_type": job_type,
                "input_data": input_data,
                "config": config
            }
        )
        response.raise_for_status()
        return response.json()["job_id"]
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/pipeline/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

# Usage
api = PixelatedEmpathyAPI("your_api_key")
result = api.transcribe("https://example.com/audio.mp3", timestamps=True)
print(result["transcript"])
```

### JavaScript/Node.js
```javascript
class PixelatedEmpathyAPI {
    constructor(apiKey, baseUrl = 'https://api.pixelatedempathy.com/v1') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }
    
    async transcribe(audioUrl, options = {}) {
        const response = await fetch(`${this.baseUrl}/transcribe`, {
            method: 'POST',
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                audio_url: audioUrl,
                ...options
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async createJob(jobType, inputData, config) {
        const response = await fetch(`${this.baseUrl}/pipeline/jobs`, {
            method: 'POST',
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_type: jobType,
                input_data: inputData,
                config: config
            })
        });
        
        const result = await response.json();
        return result.job_id;
    }
}

// Usage
const api = new PixelatedEmpathyAPI('your_api_key');
const result = await api.transcribe('https://example.com/audio.mp3', {
    language: 'en',
    timestamps: true
});
console.log(result.transcript);
```

## 🔧 Advanced Features

### Webhooks
Configure webhooks to receive job completion notifications:

```json
{
  "webhook_url": "https://your-app.com/webhooks/pixelated",
  "events": ["job.completed", "job.failed"],
  "secret": "your_webhook_secret"
}
```

### Batch Processing
Process multiple files in a single request:

```json
{
  "job_type": "batch_transcription",
  "input_data": {
    "files": [
      "https://example.com/audio1.mp3",
      "https://example.com/audio2.mp3"
    ]
  },
  "config": {
    "language": "en",
    "parallel": true
  }
}
```

## 📞 Support

- **Documentation**: https://docs.pixelatedempathy.com
- **API Status**: https://status.pixelatedempathy.com
- **Support**: support@pixelatedempathy.com
- **Community**: https://community.pixelatedempathy.com

---

*Last updated: 2025-08-12T20:56:05.349119*
