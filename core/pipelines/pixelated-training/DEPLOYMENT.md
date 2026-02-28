# Wayfarer-2-12B Deployment Guide

## 🚀 Quick Deployment Options

### 1. Local Inference
```bash
# Interactive chat
python inference.py --chat

# Single inference
python inference.py
```

### 2. API Server
```bash
# Start FastAPI server
python api_server.py

# Test endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I need help with anxiety"}'
```

### 3. Docker Deployment
```bash
# Build and run
docker-compose up --build

# Scale for load balancing
docker-compose up --scale wayfarer-api=3
```

## 📊 Performance Expectations
- **Inference Speed**: ~2-3 seconds per response
- **Memory Usage**: ~22GB GPU memory
- **Throughput**: ~10-15 requests/minute per GPU
- **Latency**: <500ms for short responses

## 🛠️ Hardware Requirements
- **GPU**: A100 80GB or V100 32GB
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ for model and cache

## 🔧 Configuration Options
- `max_length`: Response length (default: 512)
- `temperature`: Creativity (0.1-1.0, default: 0.7)
- `batch_size`: Concurrent requests (default: 1)

## 📈 Scaling Options
- **Horizontal**: Multiple API instances
- **Vertical**: Larger GPU memory
- **Load Balancing**: Nginx + multiple containers

Created: 2025-09-26T19:06:32.177648
