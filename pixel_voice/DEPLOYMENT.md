# Pixel Voice Production Deployment Guide

This guide covers deploying the Pixel Voice API and MCP server to production environments with full monitoring, scaling, and security features.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Kubernetes    │    │   Monitoring    │
│   (Nginx/ALB)   │────│    Cluster      │────│  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼───┐ ┌───▼────┐
            │ API Pods  │ │ Redis │ │ PostgreSQL │
            │ (3 replicas)│ │       │ │          │
            └───────────┘ └───────┘ └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Docker registry access
- Domain name and SSL certificates

### 1. Build and Push Images

```bash
# Build the Docker image
cd pixel_voice
docker build -t your-registry/pixel-voice:latest .

# Push to registry
docker push your-registry/pixel-voice:latest
```

### 2. Configure Secrets

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Update secrets in k8s/configmap.yaml with your values:
# - DATABASE_URL
# - REDIS_URL
# - SECRET_KEY
# - JWT_SECRET

# Apply configuration
kubectl apply -f k8s/configmap.yaml
```

### 3. Deploy Infrastructure

```bash
# Deploy storage
kubectl apply -f k8s/storage.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n pixel-voice --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n pixel-voice --timeout=300s
```

### 4. Deploy Application

```bash
# Update image in k8s/api-deployment.yaml
sed -i 's|pixel-voice:latest|your-registry/pixel-voice:latest|g' k8s/api-deployment.yaml

# Deploy API
kubectl apply -f k8s/api-deployment.yaml

# Wait for deployment
kubectl rollout status deployment/pixel-voice-api -n pixel-voice
```

### 5. Configure Ingress

```bash
# Update domain in k8s/api-deployment.yaml
sed -i 's|api.pixelvoice.example.com|your-domain.com|g' k8s/api-deployment.yaml

# Apply ingress
kubectl apply -f k8s/api-deployment.yaml
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PIXEL_VOICE_ENV` | Environment (development/production) | `production` |
| `PIXEL_VOICE_DEBUG` | Enable debug mode | `false` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `SECRET_KEY` | Application secret key | Required |
| `JWT_SECRET` | JWT signing secret | Required |

### Resource Requirements

#### Minimum Requirements
- **API Pods**: 512Mi RAM, 250m CPU each
- **PostgreSQL**: 1Gi RAM, 500m CPU
- **Redis**: 512Mi RAM, 250m CPU
- **Storage**: 100Gi for data, 50Gi for database

#### Recommended Production
- **API Pods**: 2Gi RAM, 1000m CPU each (3 replicas)
- **PostgreSQL**: 4Gi RAM, 2000m CPU
- **Redis**: 1Gi RAM, 500m CPU
- **Storage**: 500Gi for data, 200Gi for database

## 📊 Monitoring Setup

### Prometheus Configuration

```bash
# Deploy Prometheus
kubectl create namespace monitoring
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/prometheus-deployment.yaml
```

### Grafana Dashboards

1. Import the provided Grafana dashboard
2. Configure data source: `http://prometheus:9090`
3. Set up alerting channels (Slack, email, etc.)

### Key Metrics to Monitor

- **API Performance**: Request rate, latency, error rate
- **Pipeline Jobs**: Success rate, duration, queue length
- **System Resources**: CPU, memory, disk usage
- **Database**: Connection count, query performance
- **Rate Limiting**: Violation rate, quota usage

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Create TLS secret
kubectl create secret tls pixel-voice-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n pixel-voice
```

### Network Policies

```bash
# Apply network policies for security
kubectl apply -f k8s/network-policies.yaml
```

### Authentication

1. Configure OAuth providers in environment variables
2. Set up API key management
3. Configure rate limiting rules
4. Enable audit logging

## 📈 Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pixel-voice-api-hpa
  namespace: pixel-voice
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pixel-voice-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```bash
# Install VPA if not available
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vpa-release.yaml

# Apply VPA configuration
kubectl apply -f k8s/vpa.yaml
```

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. Configure repository secrets:
   - `KUBE_CONFIG_STAGING`
   - `KUBE_CONFIG_PRODUCTION`
   - `DOCKER_REGISTRY_TOKEN`
   - `SLACK_WEBHOOK`

2. The pipeline automatically:
   - Runs tests on PR/push
   - Builds and pushes Docker images
   - Deploys to staging on develop branch
   - Deploys to production on main branch

### Manual Deployment

```bash
# Deploy to staging
make deploy-staging

# Deploy to production
make deploy-production

# Rollback if needed
kubectl rollout undo deployment/pixel-voice-api -n pixel-voice
```

## 🛠️ Maintenance

### Database Migrations

```bash
# Run migrations
kubectl exec -it deployment/pixel-voice-api -n pixel-voice -- \
  uv run alembic upgrade head
```

### Backup Procedures

```bash
# Database backup
kubectl exec -it postgres-0 -n pixel-voice -- \
  pg_dump -U pixel_voice pixel_voice > backup.sql

# Data backup
kubectl exec -it pixel-voice-api-0 -n pixel-voice -- \
  tar -czf /tmp/data-backup.tar.gz /app/data
```

### Log Management

```bash
# View API logs
kubectl logs -f deployment/pixel-voice-api -n pixel-voice

# View all logs with labels
kubectl logs -f -l app=pixel-voice-api -n pixel-voice

# Export logs for analysis
kubectl logs deployment/pixel-voice-api -n pixel-voice --since=24h > api-logs.txt
```

## 🚨 Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl get pods -n pixel-voice

# Describe pod for events
kubectl describe pod <pod-name> -n pixel-voice

# Check logs
kubectl logs <pod-name> -n pixel-voice
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/pixel-voice-api -n pixel-voice -- \
  python -c "from pixel_voice.api.database import init_database; init_database('$DATABASE_URL')"
```

#### High Memory Usage
```bash
# Check resource usage
kubectl top pods -n pixel-voice

# Scale up if needed
kubectl scale deployment pixel-voice-api --replicas=5 -n pixel-voice
```

### Performance Tuning

1. **Database Optimization**
   - Add indexes for frequently queried columns
   - Configure connection pooling
   - Monitor slow queries

2. **Redis Optimization**
   - Configure memory policies
   - Monitor cache hit rates
   - Set appropriate TTL values

3. **API Optimization**
   - Enable response compression
   - Implement request caching
   - Optimize database queries

## 📞 Support

### Health Checks

- **API Health**: `https://your-domain.com/health`
- **Detailed Health**: `https://your-domain.com/health/detailed`
- **Metrics**: `https://your-domain.com/metrics`

### Monitoring Dashboards

- **Grafana**: `https://grafana.your-domain.com`
- **Prometheus**: `https://prometheus.your-domain.com`
- **API Documentation**: `https://your-domain.com/docs`

### Emergency Contacts

- **On-call Engineer**: [Contact Information]
- **DevOps Team**: [Contact Information]
- **Slack Channel**: #pixel-voice-alerts
