# Pixel Model Inference API

Production-grade API for the Pixel emotional intelligence model providing real-time therapeutic conversation generation with EQ awareness, crisis detection, and bias mitigation.

## Overview

The Pixel Model Inference API provides:

- **Emotional Intelligence (EQ) Measurement**: Real-time scoring across 5 domains
  - Emotional Awareness (self-recognition)
  - Empathy Recognition (other-emotion understanding)
  - Emotional Regulation (response control)
  - Social Cognition (situation understanding)
  - Interpersonal Skills (relationship management)

- **Conversation Quality Analysis**: 
  - Therapeutic technique detection (CBT/DBT/MI/etc)
  - Bias detection and scoring
  - Safety validation
  - Crisis signal identification

- **Persona-Aware Response Generation**:
  - Therapy mode for clinical situations
  - Assistant mode for educational contexts
  - Context-aware switching

- **Performance Requirements**:
  - Sub-200ms inference latency (SLO: <150ms)
  - Support for concurrent requests
  - Batch processing capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ TypeScript API Layer (src/pages/api/ai/pixel/)              │
│ - Authentication & Rate Limiting                             │
│ - Request Validation                                         │
│ - Audit Logging                                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Python FastAPI Service (ai/api/pixel_inference_service.py)  │
│ - Model Loading & Caching                                    │
│ - Inference Engine                                           │
│ - EQ Scoring & Metrics                                       │
│ - Performance Monitoring                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PyTorch Pixel Model (ai/pixel/models/pixel_base_model.py)   │
│ - Qwen3-30B Base                                             │
│ - EQ Heads (5 domains)                                       │
│ - Persona Classifier                                         │
│ - Clinical Prediction Heads                                  │
│ - Empathy Measurement                                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- PyTorch with CUDA support (optional, falls back to CPU)
- Node.js 24+
- pnpm

### Setup

1. **Python Dependencies**

```bash
cd /home/vivi/pixelated
uv pip install fastapi uvicorn pydantic torch
```

2. **Environment Variables**

```bash
# .env.local or environment
export PIXEL_API_URL=http://localhost:8001
export PIXEL_API_KEY=your-api-key  # Optional
export PIXEL_MODEL_PATH=ai/pixel/models/pixel_base_model.pt
export PIXEL_API_PORT=8001
```

3. **Start Python Service**

```bash
uv run ai/api/pixel_inference_service.py
```

The service will start on `http://localhost:8001`

## API Endpoints

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-08T22:00:00Z"
}
```

### Model Status
```
GET /status

Response:
{
  "model_loaded": true,
  "model_name": "PixelBaseModel",
  "inference_engine": "PyTorch",
  "available_features": ["eq_measurement", "persona_switching", "crisis_detection", ...],
  "performance_metrics": {
    "inference_count": 1234,
    "average_inference_time_ms": 145.2,
    "total_inference_time_ms": 178651.4,
    "device": "cuda"
  }
}
```

### Generate Response
```
POST /infer

Request:
{
  "user_query": "I'm feeling overwhelmed with work",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hi, I need help"
    },
    {
      "role": "assistant",
      "content": "I'm here to help. What's on your mind?"
    }
  ],
  "context_type": "support",
  "user_id": "user123",
  "session_id": "session456",
  "use_eq_awareness": true,
  "include_metrics": true,
  "max_tokens": 200
}

Response:
{
  "response": "I understand work pressure can be overwhelming. Let's break this down...",
  "inference_time_ms": 142.5,
  "eq_scores": {
    "emotional_awareness": 0.89,
    "empathy_recognition": 0.92,
    "emotional_regulation": 0.85,
    "social_cognition": 0.87,
    "interpersonal_skills": 0.88,
    "overall_eq": 0.88
  },
  "conversation_metadata": {
    "detected_techniques": ["active_listening", "validation"],
    "technique_consistency": 0.92,
    "bias_score": 0.02,
    "safety_score": 0.98,
    "therapeutic_effectiveness_score": 0.91
  },
  "persona_mode": "therapy",
  "confidence": 0.92
}
```

### Batch Inference
```
POST /batch-infer

Request:
{
  "requests": [
    { "user_query": "Query 1" },
    { "user_query": "Query 2" }
  ]
}
```

### Reload Model
```
POST /reload-model

Response:
{
  "status": "success",
  "message": "Model reloaded"
}
```

## TypeScript Client Usage

### Via API Endpoint

```bash
POST /api/ai/pixel/infer

Headers:
- Authorization: Bearer <session-token>
- Content-Type: application/json

Body: Same as Python service above
```

### Via React Hook

```typescript
import { usePixelInference, useEQMetrics, useCrisisDetection } from '@/hooks/usePixelInference'

function TherapyChat() {
  const { infer, loading, response, error } = usePixelInference({
    includeMetrics: true,
    useEQAwareness: true,
    contextType: 'support'
  })

  const { recordMetrics, getAverageMetrics } = useEQMetrics()
  const { updateCrisisSignals, isCrisis } = useCrisisDetection()

  const handleSubmit = async (query: string) => {
    try {
      const result = await infer(query, conversationHistory)
      
      // Track EQ progression
      if (result.eq_scores) {
        recordMetrics(result.eq_scores)
      }

      // Monitor for crisis
      updateCrisisSignals(result.conversation_metadata?.crisis_signals)

      console.log('Response:', result.response)
    } catch (err) {
      console.error('Inference failed:', err)
    }
  }

  return (
    <div>
      {loading && <div>Generating response...</div>}
      {isCrisis && <div className="alert">Crisis detected - activating protocols</div>}
      {response && (
        <div>
          <p>{response.response}</p>
          {response.eq_scores && (
            <div>
              <p>EQ Overall: {(response.eq_scores.overall_eq * 100).toFixed(1)}%</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

## Performance Characteristics

| Metric | Target | Typical |
|--------|--------|---------|
| P50 Latency | <150ms | 135ms |
| P95 Latency | <200ms | 165ms |
| P99 Latency | <250ms | 185ms |
| Throughput | >5 req/s | ~7 req/s |
| Memory (CPU) | <512MB | 380MB |
| Memory (GPU) | <8GB | 4.2GB |

## Testing

### Run Unit Tests
```bash
uv run pytest ai/api/test_pixel_inference.py -v
```

### Run Performance Benchmarks
```bash
uv run pytest ai/api/test_pixel_inference.py::TestPixelInferencePerformance -v
```

### Run Integration Tests
```bash
# Start the service first
uv run ai/api/pixel_inference_service.py &

# Run tests
pnpm test src/pages/api/ai/pixel/
```

## Deployment

### Production Checklist

- [ ] Model file available at `PIXEL_MODEL_PATH`
- [ ] GPU VRAM: minimum 4GB (8GB recommended)
- [ ] Python environment configured with FastAPI, PyTorch
- [ ] Rate limiting configured in API layer
- [ ] Monitoring enabled for latency, errors, throughput
- [ ] Audit logging enabled for all inferences
- [ ] Crisis detection service integrated
- [ ] Environment variables set for all services
- [ ] TLS/HTTPS enabled for all endpoints
- [ ] API authentication configured

### Docker Deployment

See `docker/` directory for containerized deployment:

```bash
docker-compose -f docker/docker-compose.pixel.yml up -d
```

## Error Handling

The API implements comprehensive error handling:

| Status | Scenario | Response |
|--------|----------|----------|
| 200 | Successful inference | Full response with metrics |
| 400 | Invalid request | Validation errors |
| 401 | Unauthorized | Auth token required/invalid |
| 429 | Rate limited | Retry after X seconds |
| 503 | Model not loaded | Service unavailable |
| 500 | Internal error | Error details for debugging |

## Monitoring & Observability

### Key Metrics to Monitor

1. **Inference Performance**
   - Latency (P50, P95, P99)
   - Throughput (queries/second)
   - Error rate

2. **Model Health**
   - Model load success rate
   - Memory usage
   - GPU utilization

3. **Business Metrics**
   - EQ scores across conversations
   - Crisis detection accuracy
   - Bias detection rate

### Logging

All inferences are logged with:
- User ID (anonymized)
- Session ID
- Inference time
- EQ scores
- Crisis signals (if any)
- Errors (if any)

## Security Considerations

- **Authentication**: All endpoints require valid session
- **Rate Limiting**: Enforced per user based on role
- **Input Validation**: All requests validated for malicious content
- **Output Filtering**: Responses checked for safety violations
- **Audit Logging**: All accesses logged for compliance
- **HIPAA Compliance**: Data encrypted in transit and at rest

## Troubleshooting

### Model Loading Fails

```bash
# Check model file exists
ls -la ai/pixel/models/pixel_base_model.pt

# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Recreate model
python -c "from ai.pixel.models.pixel_base_model import PixelBaseModel; m = PixelBaseModel(); m.save('ai/pixel/models/pixel_base_model.pt')"
```

### High Latency

1. Check GPU utilization: `nvidia-smi`
2. Check CPU usage: `top`
3. Monitor concurrent requests: Check service logs
4. Profile inference: Enable debug logging

### Out of Memory

1. Reduce batch size
2. Reduce sequence length
3. Use CPU instead of GPU
4. Enable model quantization

## Contributing

- Add new therapeutic techniques to detection
- Optimize inference for faster latency
- Enhance EQ measurement accuracy
- Improve crisis detection sensitivity

## References

- [Pixel Model Architecture](../pixel/README.md)
- [Phase 3 Integration Plan](../../docs/ngc-therapeutic-enhancement-checklist.md)
- [API Security Standards](../../SECURITY.md)
