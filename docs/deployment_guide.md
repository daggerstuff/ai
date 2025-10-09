# Pixelated Empathy AI - Deployment Guide

**Version:** 2.0.0  
**Last Updated:** 2025-08-12T21:33:00Z  
**Target Audience:** DevOps Engineers, System Administrators, Platform Engineers

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Deployment](#production-deployment)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying Pixelated Empathy AI across different environments:

- **Development**: Local development with hot reload
- **Staging**: Docker-based staging environment
- **Production**: Kubernetes-based production deployment

### Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Application   │
│   (Nginx/ALB)   │◄──►│   (Ingress)     │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SSL/TLS       │    │   Rate Limiting │    │   Database      │
│   (Cert-Manager)│    │   (Redis)       │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 20GB SSD
- **Network**: 100 Mbps

**Recommended for Production:**
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1 Gbps

### Software Dependencies

- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **Kubernetes**: 1.28+ (for production)
- **Helm**: 3.12+ (for Kubernetes deployment)
- **kubectl**: Latest stable version

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/pixelated-empathy/ai-platform.git
cd ai-platform

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f api
```

### Development Environment Variables

```bash
# .env for development
NODE_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://postgres:dev_password@localhost:5432/pixelated_dev
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
RELOAD=true

# External Services (use test/sandbox keys)
OPENAI_API_KEY=sk-test-...
HUGGINGFACE_TOKEN=hf_test_...
```

---

## Docker Deployment

### Production Docker Setup

1. **Build Production Image**:

```bash
# Build optimized production image
docker build -f Dockerfile.prod -t pixelated-empathy-ai:latest .

# Or use multi-stage build
docker build --target production -t pixelated-empathy-ai:prod .
```

2. **Production Docker Compose**:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: pixelated-empathy-ai:latest
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

3. **Deploy with Docker Compose**:

```bash
# Set production environment variables
export DATABASE_URL="postgresql://user:password@db:5432/pixelated_prod"
export REDIS_URL="redis://:password@redis:6379/0"
export JWT_SECRET_KEY="your-super-secret-production-key"

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

---

## Kubernetes Deployment

### Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pixelated-empathy
  labels:
    name: pixelated-empathy
```

### ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pixelated-config
  namespace: pixelated-empathy
data:
  NODE_ENV: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"

---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pixelated-secrets
  namespace: pixelated-empathy
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  REDIS_PASSWORD: <base64-encoded-redis-password>
```

### Application Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pixelated-api
  namespace: pixelated-empathy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pixelated-api
  template:
    metadata:
      labels:
        app: pixelated-api
    spec:
      containers:
      - name: api
        image: pixelated-empathy-ai:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: pixelated-config
        - secretRef:
            name: pixelated-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pixelated-api-service
  namespace: pixelated-empathy
spec:
  selector:
    app: pixelated-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pixelated-ingress
  namespace: pixelated-empathy
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.pixelatedempathy.com
    secretName: pixelated-tls
  rules:
  - host: api.pixelatedempathy.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pixelated-api-service
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n pixelated-empathy
kubectl get services -n pixelated-empathy
kubectl get ingress -n pixelated-empathy

# View logs
kubectl logs -f deployment/pixelated-api -n pixelated-empathy
```

---

## Production Deployment

### Helm Chart Deployment

1. **Install Helm Chart**:

```bash
# Add Helm repository
helm repo add pixelated-empathy https://charts.pixelatedempathy.com
helm repo update

# Install with custom values
helm install pixelated-empathy pixelated-empathy/pixelated-empathy-ai \
  --namespace pixelated-empathy \
  --create-namespace \
  --values production-values.yaml
```

2. **Production Values** (`production-values.yaml`):

```yaml
# production-values.yaml
replicaCount: 3

image:
  repository: pixelated-empathy-ai
  tag: "v2.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
  hosts:
    - host: api.pixelatedempathy.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: pixelated-tls
      hosts:
        - api.pixelatedempathy.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-postgres-password"
    database: "pixelated_prod"
  primary:
    persistence:
      enabled: true
      size: 100Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "secure-redis-password"
  master:
    persistence:
      enabled: true
      size: 10Gi
```

### Blue-Green Deployment

```bash
# Deploy new version (green)
helm upgrade pixelated-empathy pixelated-empathy/pixelated-empathy-ai \
  --namespace pixelated-empathy \
  --values production-values.yaml \
  --set image.tag=v2.1.0 \
  --set nameOverride=pixelated-green

# Test green deployment
kubectl port-forward service/pixelated-green 8080:80 -n pixelated-empathy
curl http://localhost:8080/health

# Switch traffic to green (update ingress)
kubectl patch ingress pixelated-ingress -n pixelated-empathy \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "pixelated-green"}]'

# Remove blue deployment after verification
helm uninstall pixelated-empathy-blue -n pixelated-empathy
```

---

## Environment Configuration

### Production Environment Variables

```bash
# Production .env template
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database (use managed database service)
DATABASE_URL=postgresql://user:password@prod-db.amazonaws.com:5432/pixelated_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis (use managed Redis service)
REDIS_URL=redis://prod-redis.amazonaws.com:6379/0
REDIS_PASSWORD=secure-redis-password
REDIS_POOL_SIZE=10

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# Security
JWT_SECRET_KEY=your-super-secure-256-bit-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30
CORS_ORIGINS=https://app.pixelatedempathy.com,https://admin.pixelatedempathy.com

# External Services (production keys)
OPENAI_API_KEY=sk-prod-...
HUGGINGFACE_TOKEN=hf_prod_...

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_BURST_SIZE=100
```

---

## Database Setup

### PostgreSQL Production Setup

1. **Database Initialization**:

```sql
-- init.sql
CREATE DATABASE pixelated_prod;
CREATE USER pixelated_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE pixelated_prod TO pixelated_user;

-- Connect to pixelated_prod database
\c pixelated_prod;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create tables (run migrations)
-- This will be handled by the application migration system
```

2. **Database Migrations**:

```bash
# Run database migrations
kubectl exec -it deployment/pixelated-api -n pixelated-empathy -- \
  python -m alembic upgrade head

# Or using init container in Kubernetes
```

### Redis Production Setup

```bash
# Redis configuration for production
# redis.conf
bind 0.0.0.0
port 6379
requirepass secure-redis-password
appendonly yes
appendfsync everysec
maxmemory 2gb
maxmemory-policy allkeys-lru
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: pixelated-empathy
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'pixelated-api'
      static_configs:
      - targets: ['pixelated-api-service:80']
      metrics_path: '/metrics'
      scrape_interval: 10s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Pixelated Empathy AI Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

---

## Troubleshooting

### Common Issues

**1. Application Won't Start**

```bash
# Check logs
kubectl logs deployment/pixelated-api -n pixelated-empathy

# Check configuration
kubectl describe configmap pixelated-config -n pixelated-empathy
kubectl describe secret pixelated-secrets -n pixelated-empathy

# Check resource limits
kubectl describe pod <pod-name> -n pixelated-empathy
```

**2. Database Connection Issues**

```bash
# Test database connectivity
kubectl exec -it deployment/pixelated-api -n pixelated-empathy -- \
  python -c "import psycopg2; conn = psycopg2.connect('$DATABASE_URL'); print('Connected!')"

# Check database service
kubectl get service postgres-service -n pixelated-empathy
kubectl describe service postgres-service -n pixelated-empathy
```

**3. High Memory Usage**

```bash
# Check memory usage
kubectl top pods -n pixelated-empathy

# Scale up resources
kubectl patch deployment pixelated-api -n pixelated-empathy \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

**4. SSL/TLS Certificate Issues**

```bash
# Check certificate status
kubectl describe certificate pixelated-tls -n pixelated-empathy

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manually trigger certificate renewal
kubectl delete certificate pixelated-tls -n pixelated-empathy
kubectl apply -f ingress.yaml
```

### Health Checks

```bash
# Application health
curl https://api.pixelatedempathy.com/health

# Detailed health check
curl https://api.pixelatedempathy.com/health/detailed

# Metrics endpoint
curl https://api.pixelatedempathy.com/metrics
```

### Performance Tuning

**Database Optimization**:
```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_conversations_user_id ON conversations(user_id);
CREATE INDEX CONCURRENTLY idx_conversations_created_at ON conversations(created_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM conversations WHERE user_id = 'user123';
```

**Redis Optimization**:
```bash
# Monitor Redis performance
redis-cli --latency-history -h redis-service

# Check memory usage
redis-cli info memory
```

---

## Rollback Procedures

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/pixelated-api -n pixelated-empathy

# Rollback to previous version
kubectl rollout undo deployment/pixelated-api -n pixelated-empathy

# Rollback to specific revision
kubectl rollout undo deployment/pixelated-api --to-revision=2 -n pixelated-empathy
```

### Helm Rollback

```bash
# View release history
helm history pixelated-empathy -n pixelated-empathy

# Rollback to previous release
helm rollback pixelated-empathy -n pixelated-empathy

# Rollback to specific revision
helm rollback pixelated-empathy 2 -n pixelated-empathy
```

---

*Last updated: 2025-08-12T21:33:00Z*  
*For deployment support, contact the DevOps team or create an issue on GitHub.*
