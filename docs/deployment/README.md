# Deployment Guides: Pixelated Empathy AI

**Complete production deployment instructions for Pixelated Empathy AI system.**

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Provider Deployments](#cloud-provider-deployments)
7. [Production Checklist](#production-checklist)

---

## Deployment Overview

### Architecture Components

The Pixelated Empathy AI system consists of several components that need to be deployed:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web Dashboard │
│   (NGINX/ALB)   │───▶│   (FastAPI)     │───▶│   (React)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │  Worker Nodes   │
                    │  (Celery/Redis) │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │  PostgreSQL +   │
                    │  Redis + S3     │
                    └─────────────────┘
```

### Deployment Options

1. **Local Development**: Docker Compose for development and testing
2. **Single Server**: Docker deployment on a single server
3. **Kubernetes**: Scalable container orchestration
4. **Cloud Native**: AWS/GCP/Azure managed services
5. **Hybrid**: Mix of managed and self-hosted components

---

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements (Production)
- **CPU**: 8+ cores
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD
- **Network**: 1 Gbps

### Software Dependencies

```bash
# Required software
- Docker 24.0+
- Docker Compose 2.0+
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

# Optional (for Kubernetes)
- kubectl 1.28+
- Helm 3.12+
- Kubernetes 1.28+
```

### Environment Variables

Create a `.env` file with required configuration:

```bash
# API Configuration
PIXELATED_EMPATHY_API_KEY=your_api_key_here
PIXELATED_EMPATHY_BASE_URL=https://api.pixelatedempathy.com/v1
PIXELATED_EMPATHY_SECRET_KEY=your_secret_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/pixelated_empathy
REDIS_URL=redis://localhost:6379/0

# Storage Configuration
S3_BUCKET_NAME=pixelated-empathy-storage
S3_ACCESS_KEY_ID=your_access_key
S3_SECRET_ACCESS_KEY=your_secret_key
S3_REGION=us-west-2

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO
```

---

## Local Development Setup

### Quick Start with Docker Compose

1. **Clone Repository**:
```bash
git clone https://github.com/pixelatedempathy/api.git
cd api
```

2. **Environment Setup**:
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

3. **Start Services**:
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/pixelated_empathy
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # Database
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: pixelated_empathy
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Worker Processes
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/pixelated_empathy
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # Load Balancer
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

### Development Commands

```bash
# Database operations
docker-compose exec db psql -U postgres -d pixelated_empathy

# API shell access
docker-compose exec api python -c "from app import *"

# View real-time logs
docker-compose logs -f api worker

# Restart specific service
docker-compose restart api

# Scale workers
docker-compose up -d --scale worker=3

# Clean up
docker-compose down -v
```

---

## Docker Deployment

### Production Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: pixelatedempathy/api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
```

### Single Server Deployment

```bash
#!/bin/bash
# deploy-single-server.sh

set -e

echo "🚀 Starting Pixelated Empathy AI deployment..."

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
sudo mkdir -p /opt/pixelated-empathy
cd /opt/pixelated-empathy

# Download configuration files
curl -O https://raw.githubusercontent.com/pixelatedempathy/api/main/docker-compose.prod.yml
curl -O https://raw.githubusercontent.com/pixelatedempathy/api/main/.env.example

# Setup environment
cp .env.example .env
echo "Please edit .env file with your configuration"
nano .env

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Setup SSL with Let's Encrypt
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.pixelatedempathy.com

# Setup log rotation
sudo tee /etc/logrotate.d/pixelated-empathy << EOF
/opt/pixelated-empathy/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

echo "✅ Deployment completed!"
echo "API available at: https://api.pixelatedempathy.com"
echo "Health check: curl https://api.pixelatedempathy.com/health"
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pixelated-empathy
  labels:
    name: pixelated-empathy

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pixelated-empathy-config
  namespace: pixelated-empathy
data:
  LOG_LEVEL: "INFO"
  PROMETHEUS_ENABLED: "true"
  REDIS_URL: "redis://redis-service:6379/0"
  DATABASE_URL: "postgresql://postgres:password@postgres-service:5432/pixelated_empathy"
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pixelated-empathy-secrets
  namespace: pixelated-empathy
type: Opaque
data:
  jwt-secret-key: <base64-encoded-jwt-secret>
  database-password: <base64-encoded-db-password>
  api-key: <base64-encoded-api-key>
```

### Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: pixelated-empathy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: "pixelated_empathy"
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pixelated-empathy-secrets
              key: database-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: pixelated-empathy
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: pixelated-empathy
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### API Deployment

```yaml
# k8s/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pixelated-empathy-api
  namespace: pixelated-empathy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pixelated-empathy-api
  template:
    metadata:
      labels:
        app: pixelated-empathy-api
    spec:
      containers:
      - name: api
        image: pixelatedempathy/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: pixelated-empathy-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: pixelated-empathy-config
              key: REDIS_URL
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: pixelated-empathy-secrets
              key: jwt-secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: pixelated-empathy-api-service
  namespace: pixelated-empathy
spec:
  selector:
    app: pixelated-empathy-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pixelated-empathy-ingress
  namespace: pixelated-empathy
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.pixelatedempathy.com
    secretName: pixelated-empathy-tls
  rules:
  - host: api.pixelatedempathy.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pixelated-empathy-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pixelated-empathy-api-hpa
  namespace: pixelated-empathy
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pixelated-empathy-api
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Deployment Script

```bash
#!/bin/bash
# k8s/deploy.sh

set -e

echo "🚀 Deploying Pixelated Empathy AI to Kubernetes..."

# Apply namespace
kubectl apply -f namespace.yaml

# Apply secrets (ensure secrets.yaml is configured)
kubectl apply -f secrets.yaml

# Apply ConfigMap
kubectl apply -f configmap.yaml

# Deploy PostgreSQL
kubectl apply -f postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n pixelated-empathy

# Deploy Redis
kubectl apply -f redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=available --timeout=300s deployment/redis -n pixelated-empathy

# Deploy API
kubectl apply -f api.yaml

# Wait for API to be ready
kubectl wait --for=condition=available --timeout=300s deployment/pixelated-empathy-api -n pixelated-empathy

# Apply Ingress
kubectl apply -f ingress.yaml

# Apply HPA
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n pixelated-empathy
kubectl get services -n pixelated-empathy
kubectl get ingress -n pixelated-empathy

echo "✅ Deployment completed!"
echo "API should be available at: https://api.pixelatedempathy.com"
echo "Check status with: kubectl get pods -n pixelated-empathy"
```

---

## Cloud Provider Deployments

### AWS Deployment with EKS

```bash
#!/bin/bash
# aws/deploy-eks.sh

# Create EKS cluster
eksctl create cluster \
  --name pixelated-empathy \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Deploy application
kubectl apply -f ../k8s/
```

### GCP Deployment with GKE

```bash
#!/bin/bash
# gcp/deploy-gke.sh

# Create GKE cluster
gcloud container clusters create pixelated-empathy \
  --zone us-west1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --machine-type n1-standard-2 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials pixelated-empathy --zone us-west1-a

# Deploy application
kubectl apply -f ../k8s/
```

### Azure Deployment with AKS

```bash
#!/bin/bash
# azure/deploy-aks.sh

# Create resource group
az group create --name pixelated-empathy-rg --location westus2

# Create AKS cluster
az aks create \
  --resource-group pixelated-empathy-rg \
  --name pixelated-empathy \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group pixelated-empathy-rg --name pixelated-empathy

# Deploy application
kubectl apply -f ../k8s/
```

---

## Production Checklist

### Pre-Deployment Checklist

- [ ] **Environment Configuration**
  - [ ] All environment variables configured
  - [ ] Secrets properly encrypted and stored
  - [ ] Database connection strings validated
  - [ ] API keys and tokens configured

- [ ] **Security Setup**
  - [ ] SSL/TLS certificates installed
  - [ ] Firewall rules configured
  - [ ] Network security groups configured
  - [ ] API rate limiting enabled

- [ ] **Database Setup**
  - [ ] Database initialized with schema
  - [ ] Database backups configured
  - [ ] Connection pooling configured
  - [ ] Performance monitoring enabled

- [ ] **Monitoring Setup**
  - [ ] Application monitoring configured
  - [ ] Log aggregation setup
  - [ ] Health checks implemented
  - [ ] Alerting rules configured

### Post-Deployment Validation

```bash
#!/bin/bash
# scripts/validate-deployment.sh

echo "🔍 Validating Pixelated Empathy AI deployment..."

# Check API health
echo "Checking API health..."
curl -f https://api.pixelatedempathy.com/health || exit 1

# Check database connectivity
echo "Checking database connectivity..."
curl -f https://api.pixelatedempathy.com/v1/datasets || exit 1

# Check authentication
echo "Checking authentication..."
curl -H "Authorization: Bearer test_key" https://api.pixelatedempathy.com/v1/datasets || echo "Authentication check failed (expected)"

# Check rate limiting
echo "Checking rate limiting..."
for i in {1..10}; do
  curl -s -o /dev/null -w "%{http_code}\n" https://api.pixelatedempathy.com/health
done

# Check SSL certificate
echo "Checking SSL certificate..."
curl -I https://api.pixelatedempathy.com 2>&1 | grep -q "SSL certificate verify ok" || echo "SSL check failed"

echo "✅ Deployment validation completed!"
```

### Monitoring and Maintenance

```bash
# Daily maintenance script
#!/bin/bash
# scripts/daily-maintenance.sh

# Check disk space
df -h | grep -E "9[0-9]%" && echo "WARNING: Disk space critical"

# Check memory usage
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.pixelatedempathy.com/health

# Rotate logs
docker-compose exec api logrotate /etc/logrotate.conf

# Update SSL certificates
certbot renew --quiet

# Backup database
pg_dump $DATABASE_URL | gzip > /backups/db-$(date +%Y%m%d).sql.gz

echo "Daily maintenance completed at $(date)"
```

---

**For specific cloud provider configurations and advanced deployment scenarios, see our [Cloud Deployment Guides](cloud_deployments.md) and [Infrastructure as Code](infrastructure_as_code.md) documentation.**

*Deployment guides are updated with each release. Last updated: 2025-08-17*
