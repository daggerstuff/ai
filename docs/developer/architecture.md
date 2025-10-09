# Architecture Documentation: Pixelated Empathy AI

**Comprehensive technical architecture documentation for developers and system architects.**

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Architecture](#data-architecture)
4. [Security Architecture](#security-architecture)
5. [Deployment Architecture](#deployment-architecture)
6. [Performance Architecture](#performance-architecture)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Web Apps      │   Mobile Apps   │   CLI Tools & SDKs         │
│   (React/Vue)   │   (iOS/Android) │   (Python/JS/CLI)          │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │  (Rate Limiting │
                    │  Authentication │
                    │   Load Balancer)│
                    └─────────┬───────┘
                              │
                ┌─────────────┼─────────────┐
                │                           │
    ┌─────────────────┐              ┌─────────────────┐
    │  FastAPI Server │              │  Processing     │
    │  - REST API     │              │  Workers        │
    │  - WebSockets   │              │  - Quality Val  │
    │  - Real-time    │              │  - Data Export  │
    └─────────┬───────┘              │  - Analytics    │
              │                      └─────────┬───────┘
              │                                │
              └────────────┬───────────────────┘
                           │
              ┌─────────────────┐
              │   Data Layer    │
              │  - PostgreSQL   │
              │  - Redis Cache  │
              │  - File Storage │
              │  - Search Index │
              └─────────────────┘
```

### Technology Stack

#### Backend Services
- **API Server**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 15+ with FTS5 full-text search
- **Cache**: Redis 7+ for session and data caching
- **Message Queue**: Redis with Celery for async processing
- **File Storage**: S3-compatible storage for exports and media

#### Frontend & Client
- **Web Dashboard**: React 18+ with TypeScript
- **Mobile Apps**: React Native with TypeScript
- **SDKs**: Python 3.8+ and Node.js 16+ client libraries
- **CLI Tools**: Python-based command-line interface

#### Infrastructure
- **Container Platform**: Docker with Kubernetes orchestration
- **Load Balancer**: NGINX with SSL termination
- **Monitoring**: Prometheus + Grafana + ELK stack
- **CI/CD**: GitHub Actions with automated testing and deployment

---

## Component Architecture

### API Gateway Layer

```python
# API Gateway Configuration
class APIGateway:
    """
    Handles authentication, rate limiting, and request routing.
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.auth_service = AuthenticationService()
        self.load_balancer = LoadBalancer()
    
    async def process_request(self, request):
        # 1. Authentication
        user = await self.auth_service.authenticate(request)
        
        # 2. Rate limiting
        await self.rate_limiter.check_limits(user)
        
        # 3. Request routing
        backend = self.load_balancer.select_backend()
        
        # 4. Forward request
        return await backend.handle_request(request)
```

#### Features
- **JWT Authentication**: Stateless token-based authentication
- **Rate Limiting**: Per-user and per-endpoint rate limits
- **Load Balancing**: Round-robin and health-based routing
- **Request Logging**: Comprehensive request/response logging
- **CORS Handling**: Cross-origin request support

### FastAPI Application Layer

```python
# FastAPI Application Structure
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Pixelated Empathy AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_database():
    db = DatabaseConnection()
    try:
        yield db
    finally:
        await db.close()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    return await authenticate_user(token)

# Route handlers
@app.get("/v1/conversations")
async def list_conversations(
    db: Database = Depends(get_database),
    user: User = Depends(get_current_user),
    tier: Optional[str] = None,
    limit: int = 100
):
    return await conversation_service.list_conversations(
        db, user, tier, limit
    )
```

#### Key Components
- **Route Handlers**: RESTful endpoint implementations
- **Dependency Injection**: Database, authentication, and service dependencies
- **Middleware**: CORS, authentication, logging, and error handling
- **Background Tasks**: Async job processing and notifications
- **WebSocket Support**: Real-time updates and streaming

### Processing Worker Layer

```python
# Celery Worker Configuration
from celery import Celery

celery_app = Celery(
    "pixelated_empathy_workers",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task(bind=True)
def process_quality_validation(self, conversation_data):
    """
    Background task for conversation quality validation.
    """
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # Load NLP models
        quality_validator = QualityValidator()
        self.update_state(state='PROGRESS', meta={'progress': 25})
        
        # Validate conversation
        results = quality_validator.validate(conversation_data)
        self.update_state(state='PROGRESS', meta={'progress': 75})
        
        # Store results
        store_validation_results(conversation_data['id'], results)
        self.update_state(state='PROGRESS', meta={'progress': 100})
        
        return {
            'status': 'completed',
            'results': results
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc)}
        )
        raise
```

#### Worker Types
- **Quality Validation Workers**: NLP-based conversation assessment
- **Export Workers**: Multi-format data export processing
- **Analytics Workers**: Statistical analysis and reporting
- **Search Indexing Workers**: Full-text search index updates

---

## Data Architecture

### Database Schema

```sql
-- Core conversation storage
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) NOT NULL,
    quality_score DECIMAL(4,3) NOT NULL,
    message_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    
    -- Indexes for performance
    INDEX idx_conversations_dataset (dataset_name),
    INDEX idx_conversations_tier (tier),
    INDEX idx_conversations_quality (quality_score),
    INDEX idx_conversations_created (created_at)
);

-- Message storage with full-text search
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    
    -- Full-text search index
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,
    
    INDEX idx_messages_conversation (conversation_id),
    INDEX idx_messages_search USING GIN (search_vector)
);

-- Quality metrics storage
CREATE TABLE quality_metrics (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    therapeutic_accuracy DECIMAL(4,3) NOT NULL,
    conversation_coherence DECIMAL(4,3) NOT NULL,
    emotional_authenticity DECIMAL(4,3) NOT NULL,
    clinical_compliance DECIMAL(4,3) NOT NULL,
    safety_score DECIMAL(4,3) NOT NULL,
    overall_quality DECIMAL(4,3) NOT NULL,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_version VARCHAR(20) NOT NULL,
    
    INDEX idx_quality_therapeutic (therapeutic_accuracy),
    INDEX idx_quality_overall (overall_quality)
);

-- Dataset metadata
CREATE TABLE datasets (
    name VARCHAR(255) PRIMARY KEY,
    description TEXT,
    total_conversations INTEGER DEFAULT 0,
    average_quality DECIMAL(4,3),
    tier_distribution JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User management and API keys
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    organization VARCHAR(255),
    tier VARCHAR(50) NOT NULL DEFAULT 'basic',
    rate_limit_per_hour INTEGER NOT NULL DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    last_used TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Processing jobs tracking
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    parameters JSONB,
    results JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    INDEX idx_jobs_user (user_id),
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_created (created_at)
);
```

### Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │   Processing    │    │   Validated     │
│   Ingestion     │───▶│   Pipeline      │───▶│   Storage       │
│   (JSONL/CSV)   │    │   (Quality Val) │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Storage  │    │   Redis Queue   │    │   Search Index  │
│   (S3/MinIO)    │    │   (Celery)      │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Caching Strategy

```python
# Multi-level caching implementation
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = redis.Redis()  # Redis cache
        self.l3_cache = DatabaseConnection()  # Database
    
    async def get(self, key):
        # L1: Check in-memory cache
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2: Check Redis cache
        cached_value = await self.l2_cache.get(key)
        if cached_value:
            value = json.loads(cached_value)
            self.l1_cache[key] = value  # Populate L1
            return value
        
        # L3: Fetch from database
        value = await self.l3_cache.fetch(key)
        if value:
            # Populate both cache levels
            self.l1_cache[key] = value
            await self.l2_cache.setex(key, 3600, json.dumps(value))
        
        return value
```

---

## Security Architecture

### Authentication & Authorization

```python
# JWT-based authentication system
from jose import JWTError, jwt
from passlib.context import CryptContext

class SecurityManager:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return user_id
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def hash_password(self, password: str):
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str):
        return self.pwd_context.verify(plain_password, hashed_password)
```

### API Security Layers

1. **Transport Security**
   - TLS 1.3 encryption for all communications
   - Certificate pinning for mobile applications
   - HSTS headers for web security

2. **Authentication Security**
   - JWT tokens with short expiration times
   - API key rotation every 90 days
   - Multi-factor authentication for admin accounts

3. **Authorization Security**
   - Role-based access control (RBAC)
   - Resource-level permissions
   - Tier-based data access restrictions

4. **Input Validation**
   - Pydantic models for request validation
   - SQL injection prevention with parameterized queries
   - XSS protection with content sanitization

### Data Privacy & Compliance

```python
# Data anonymization and privacy protection
class PrivacyManager:
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        }
    
    def anonymize_conversation(self, conversation):
        """Remove or mask PII from conversation data."""
        
        anonymized = conversation.copy()
        
        for message in anonymized['messages']:
            content = message['content']
            
            # Replace PII with placeholders
            for pii_type, pattern in self.pii_patterns.items():
                content = re.sub(pattern, f'<{pii_type.upper()}>', content)
            
            message['content'] = content
        
        return anonymized
    
    def audit_data_access(self, user_id, resource_id, action):
        """Log data access for compliance auditing."""
        
        audit_record = {
            'user_id': user_id,
            'resource_id': resource_id,
            'action': action,
            'timestamp': datetime.utcnow(),
            'ip_address': request.client.host,
            'user_agent': request.headers.get('user-agent')
        }
        
        # Store in secure audit log
        self.audit_logger.log(audit_record)
```

---

## Deployment Architecture

### Container Architecture

```dockerfile
# Multi-stage Docker build for API server
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# API server deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pixelated-empathy-api
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
        image: pixelated-empathy/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
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
# Load balancer service
apiVersion: v1
kind: Service
metadata:
  name: pixelated-empathy-api-service
spec:
  selector:
    app: pixelated-empathy-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Infrastructure as Code

```terraform
# Terraform configuration for AWS deployment
provider "aws" {
  region = var.aws_region
}

# EKS cluster for container orchestration
resource "aws_eks_cluster" "pixelated_empathy" {
  name     = "pixelated-empathy-cluster"
  role_arn = aws_iam_role.cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_iam_role_policy_attachment.service_policy,
  ]
}

# RDS PostgreSQL database
resource "aws_db_instance" "main" {
  identifier = "pixelated-empathy-db"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "pixelated_empathy"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "pixelated-empathy-final-snapshot"
  
  tags = {
    Name = "Pixelated Empathy Database"
  }
}

# ElastiCache Redis cluster
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "pixelated-empathy-redis"
  description                = "Redis cluster for Pixelated Empathy AI"
  
  node_type                  = "cache.r6g.large"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "Pixelated Empathy Redis"
  }
}
```

---

## Performance Architecture

### Horizontal Scaling Strategy

```python
# Auto-scaling configuration
class AutoScaler:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.kubernetes_client = KubernetesClient()
    
    async def check_scaling_conditions(self):
        """Check if scaling is needed based on metrics."""
        
        metrics = await self.metrics_collector.get_current_metrics()
        
        # CPU-based scaling
        if metrics['cpu_usage'] > 80:
            await self.scale_up('cpu_high')
        elif metrics['cpu_usage'] < 20:
            await self.scale_down('cpu_low')
        
        # Memory-based scaling
        if metrics['memory_usage'] > 85:
            await self.scale_up('memory_high')
        
        # Request queue-based scaling
        if metrics['queue_length'] > 100:
            await self.scale_up('queue_high')
        elif metrics['queue_length'] < 10:
            await self.scale_down('queue_low')
    
    async def scale_up(self, reason):
        """Scale up the application."""
        
        current_replicas = await self.kubernetes_client.get_replica_count()
        new_replicas = min(current_replicas + 2, 20)  # Max 20 replicas
        
        await self.kubernetes_client.set_replica_count(new_replicas)
        
        self.logger.info(f"Scaled up to {new_replicas} replicas. Reason: {reason}")
    
    async def scale_down(self, reason):
        """Scale down the application."""
        
        current_replicas = await self.kubernetes_client.get_replica_count()
        new_replicas = max(current_replicas - 1, 3)  # Min 3 replicas
        
        await self.kubernetes_client.set_replica_count(new_replicas)
        
        self.logger.info(f"Scaled down to {new_replicas} replicas. Reason: {reason}")
```

### Database Performance Optimization

```sql
-- Performance optimization queries and indexes

-- Conversation search optimization
CREATE INDEX CONCURRENTLY idx_conversations_search 
ON conversations USING GIN (
    (
        setweight(to_tsvector('english', coalesce(metadata->>'title', '')), 'A') ||
        setweight(to_tsvector('english', coalesce(metadata->>'description', '')), 'B')
    )
);

-- Quality-based filtering optimization
CREATE INDEX CONCURRENTLY idx_conversations_quality_tier 
ON conversations (tier, quality_score DESC) 
WHERE quality_score >= 0.7;

-- Time-based partitioning for large datasets
CREATE TABLE conversations_2025_q1 PARTITION OF conversations
FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

CREATE TABLE conversations_2025_q2 PARTITION OF conversations
FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

-- Materialized view for analytics
CREATE MATERIALIZED VIEW conversation_analytics AS
SELECT 
    dataset_name,
    tier,
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as conversation_count,
    AVG(quality_score) as avg_quality,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quality_score) as median_quality
FROM conversations
GROUP BY dataset_name, tier, DATE_TRUNC('day', created_at);

-- Refresh materialized view daily
CREATE OR REPLACE FUNCTION refresh_analytics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY conversation_analytics;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh
SELECT cron.schedule('refresh-analytics', '0 2 * * *', 'SELECT refresh_analytics();');
```

### Monitoring & Observability

```python
# Comprehensive monitoring system
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog

class MonitoringSystem:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.logger = structlog.get_logger()
        
        # Metrics
        self.request_count = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'api_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
    
    def track_request(self, method, endpoint, status_code, duration):
        """Track API request metrics."""
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Structured logging
        self.logger.info(
            "api_request",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration
        )
    
    def track_database_performance(self, query_type, duration, rows_affected):
        """Track database performance metrics."""
        
        self.logger.info(
            "database_query",
            query_type=query_type,
            duration=duration,
            rows_affected=rows_affected
        )
    
    def track_cache_performance(self, hit_rate):
        """Track cache performance."""
        
        self.cache_hit_rate.set(hit_rate)
        
        self.logger.info(
            "cache_performance",
            hit_rate=hit_rate
        )
```

---

**This architecture documentation provides a comprehensive technical overview of the Pixelated Empathy AI system. For implementation details, see our [SDK Reference](sdk_reference.md) and [Integration Patterns](integration_patterns.md).**

*Architecture documentation is updated with each major release. Last updated: 2025-08-17*
