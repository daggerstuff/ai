# Pixelated Empathy AI - Configuration Documentation

**Version:** 2.0.0  
**Last Updated:** 2025-08-12T21:38:00Z  
**Target Audience:** System Administrators, DevOps Engineers, Developers

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Database Configuration](#database-configuration)
- [Redis Configuration](#redis-configuration)
- [API Configuration](#api-configuration)
- [Security Configuration](#security-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Deployment Configurations](#deployment-configurations)

---

## Overview

Pixelated Empathy AI uses a hierarchical configuration system:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env, config.yaml)
3. **Default Values** (lowest priority)

### Configuration Loading Order

```
Environment Variables → .env file → config.yaml → defaults.py
```

---

## Environment Variables

### Core Application Settings

```bash
# Application Environment
NODE_ENV=production                    # Environment: development, staging, production
DEBUG=false                           # Enable debug mode (true/false)
LOG_LEVEL=INFO                        # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server Configuration
API_HOST=0.0.0.0                      # Host to bind the server
API_PORT=8000                         # Port to run the server
API_WORKERS=4                         # Number of worker processes
API_TIMEOUT=30                        # Request timeout in seconds
API_MAX_REQUEST_SIZE=10485760         # Max request size in bytes (10MB)

# Application Metadata
APP_NAME="Pixelated Empathy AI"       # Application name
APP_VERSION=2.0.0                     # Application version
APP_DESCRIPTION="AI-powered empathy and conversation analysis"
```

### Database Configuration

```bash
# Primary Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/pixelated_ai
DATABASE_POOL_SIZE=20                 # Connection pool size
DATABASE_MAX_OVERFLOW=30              # Max overflow connections
DATABASE_POOL_TIMEOUT=30              # Pool timeout in seconds
DATABASE_POOL_RECYCLE=3600            # Connection recycle time in seconds
DATABASE_ECHO=false                   # Echo SQL queries (true/false)

# Development Database (SQLite)
DATABASE_URL_DEV=sqlite:///./dev.db   # Development database URL
DATABASE_MIGRATE_ON_START=true        # Auto-migrate on startup

# Database SSL Configuration
DATABASE_SSL_MODE=require             # SSL mode: disable, allow, prefer, require
DATABASE_SSL_CERT=/path/to/cert.pem   # SSL certificate path
DATABASE_SSL_KEY=/path/to/key.pem     # SSL key path
DATABASE_SSL_ROOT_CERT=/path/to/ca.pem # SSL root certificate path
```

### Redis Configuration

```bash
# Redis Connection
REDIS_URL=redis://localhost:6379/0    # Redis connection URL
REDIS_PASSWORD=your_redis_password    # Redis password
REDIS_USERNAME=default                # Redis username (Redis 6+)
REDIS_DB=0                           # Redis database number

# Redis Pool Configuration
REDIS_POOL_SIZE=10                   # Connection pool size
REDIS_POOL_TIMEOUT=5                 # Connection timeout in seconds
REDIS_POOL_RETRY_ON_TIMEOUT=true     # Retry on timeout
REDIS_SOCKET_TIMEOUT=5               # Socket timeout in seconds
REDIS_SOCKET_CONNECT_TIMEOUT=5       # Socket connect timeout

# Redis Cluster Configuration (if using cluster)
REDIS_CLUSTER_NODES=redis1:6379,redis2:6379,redis3:6379
REDIS_CLUSTER_SKIP_FULL_COVERAGE_CHECK=false
REDIS_CLUSTER_MAX_CONNECTIONS=32
```

### Authentication & Security

```bash
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-256-bit-key  # JWT signing key (min 32 chars)
JWT_ALGORITHM=HS256                           # JWT algorithm
JWT_EXPIRE_MINUTES=30                         # Access token expiry
JWT_REFRESH_EXPIRE_DAYS=7                     # Refresh token expiry

# API Key Configuration
API_KEY_HEADER_NAME=X-API-Key                 # API key header name
API_KEY_LENGTH=32                             # Generated API key length
API_KEY_PREFIX=pk_                            # API key prefix

# Password Security
PASSWORD_MIN_LENGTH=8                         # Minimum password length
PASSWORD_REQUIRE_UPPERCASE=true               # Require uppercase letters
PASSWORD_REQUIRE_LOWERCASE=true               # Require lowercase letters
PASSWORD_REQUIRE_NUMBERS=true                 # Require numbers
PASSWORD_REQUIRE_SPECIAL=true                 # Require special characters
PASSWORD_HASH_ROUNDS=12                       # Bcrypt hash rounds

# CORS Configuration
CORS_ORIGINS=https://app.pixelatedempathy.com,https://admin.pixelatedempathy.com
CORS_ALLOW_CREDENTIALS=true                   # Allow credentials
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS # Allowed methods
CORS_ALLOW_HEADERS=*                          # Allowed headers
CORS_MAX_AGE=86400                           # Preflight cache time
```

### Rate Limiting

```bash
# Global Rate Limits
RATE_LIMIT_ENABLED=true                       # Enable rate limiting
RATE_LIMIT_STORAGE=redis                      # Storage backend: redis, memory
RATE_LIMIT_KEY_FUNC=get_remote_address        # Key function for rate limiting

# Default Rate Limits
RATE_LIMIT_DEFAULT_PER_MINUTE=100            # Default requests per minute
RATE_LIMIT_DEFAULT_PER_HOUR=1000             # Default requests per hour
RATE_LIMIT_DEFAULT_PER_DAY=10000             # Default requests per day

# Burst Configuration
RATE_LIMIT_BURST_SIZE=50                     # Burst allowance
RATE_LIMIT_BURST_WINDOW=60                   # Burst window in seconds

# User-specific Rate Limits
RATE_LIMIT_FREE_TIER_PER_MINUTE=60           # Free tier limit
RATE_LIMIT_PREMIUM_TIER_PER_MINUTE=300       # Premium tier limit
RATE_LIMIT_ENTERPRISE_TIER_PER_MINUTE=1000   # Enterprise tier limit

# Rate Limit Headers
RATE_LIMIT_HEADERS_ENABLED=true              # Include rate limit headers
RATE_LIMIT_HEADER_LIMIT=X-RateLimit-Limit    # Limit header name
RATE_LIMIT_HEADER_REMAINING=X-RateLimit-Remaining # Remaining header name
RATE_LIMIT_HEADER_RESET=X-RateLimit-Reset    # Reset header name
```

### External Services

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key        # OpenAI API key
OPENAI_ORG_ID=org-your-organization-id       # OpenAI organization ID
OPENAI_MODEL=gpt-4                           # Default model
OPENAI_MAX_TOKENS=2048                       # Max tokens per request
OPENAI_TEMPERATURE=0.7                       # Model temperature
OPENAI_TIMEOUT=30                            # Request timeout

# Hugging Face Configuration
HUGGINGFACE_TOKEN=hf_your-huggingface-token  # Hugging Face token
HUGGINGFACE_MODEL_CACHE_DIR=/app/models      # Model cache directory
HUGGINGFACE_OFFLINE=false                    # Offline mode

# Email Configuration (SMTP)
SMTP_HOST=smtp.gmail.com                     # SMTP server host
SMTP_PORT=587                                # SMTP server port
SMTP_USERNAME=your-email@gmail.com           # SMTP username
SMTP_PASSWORD=your-app-password              # SMTP password
SMTP_USE_TLS=true                           # Use TLS encryption
SMTP_FROM_EMAIL=noreply@pixelatedempathy.com # From email address
SMTP_FROM_NAME="Pixelated Empathy AI"        # From name
```

### Monitoring & Observability

```bash
# Prometheus Metrics
PROMETHEUS_ENABLED=true                       # Enable Prometheus metrics
PROMETHEUS_PORT=9090                         # Metrics port
PROMETHEUS_PATH=/metrics                     # Metrics endpoint path
PROMETHEUS_INCLUDE_DEFAULT_METRICS=true      # Include default metrics

# Logging Configuration
LOG_FORMAT=json                              # Log format: json, text
LOG_FILE=/var/log/pixelated-ai.log          # Log file path
LOG_MAX_SIZE=100MB                          # Max log file size
LOG_BACKUP_COUNT=5                          # Number of backup files
LOG_ROTATION=daily                          # Rotation: daily, weekly, monthly

# Structured Logging Fields
LOG_INCLUDE_TIMESTAMP=true                   # Include timestamp
LOG_INCLUDE_LEVEL=true                      # Include log level
LOG_INCLUDE_LOGGER=true                     # Include logger name
LOG_INCLUDE_THREAD=false                    # Include thread info
LOG_INCLUDE_PROCESS=false                   # Include process info

# Sentry Configuration (Error Tracking)
SENTRY_DSN=https://your-dsn@sentry.io/project-id # Sentry DSN
SENTRY_ENVIRONMENT=production                # Environment name
SENTRY_RELEASE=v2.0.0                       # Release version
SENTRY_SAMPLE_RATE=1.0                      # Error sample rate
SENTRY_TRACES_SAMPLE_RATE=0.1               # Performance sample rate
```

### Feature Flags

```bash
# Feature Toggles
FEATURE_NEW_CONVERSATION_ENGINE=true         # Enable new conversation engine
FEATURE_ADVANCED_ANALYTICS=false            # Enable advanced analytics
FEATURE_BETA_FEATURES=false                 # Enable beta features
FEATURE_EXPERIMENTAL_MODELS=false           # Enable experimental models

# A/B Testing
AB_TEST_ENABLED=true                        # Enable A/B testing
AB_TEST_CONVERSATION_ALGORITHM=50           # Percentage for new algorithm
AB_TEST_UI_REDESIGN=25                      # Percentage for UI redesign
```

---

## Configuration Files

### .env File Structure

```bash
# .env file template
# Copy this to .env and customize for your environment

# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_URL=postgresql://user:password@localhost:5432/pixelated_ai
DATABASE_POOL_SIZE=20

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
JWT_SECRET_KEY=your-super-secret-256-bit-key
CORS_ORIGINS=https://app.pixelatedempathy.com

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================
OPENAI_API_KEY=sk-your-openai-api-key
HUGGINGFACE_TOKEN=hf_your-huggingface-token

# =============================================================================
# MONITORING
# =============================================================================
PROMETHEUS_ENABLED=true
SENTRY_DSN=https://your-dsn@sentry.io/project-id
```

### config.yaml Structure

```yaml
# config.yaml - Advanced configuration
app:
  name: "Pixelated Empathy AI"
  version: "2.0.0"
  description: "AI-powered empathy and conversation analysis"
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  max_request_size: 10485760
  
database:
  url: "${DATABASE_URL}"
  pool:
    size: 20
    max_overflow: 30
    timeout: 30
    recycle: 3600
  ssl:
    mode: "require"
    cert_path: "/path/to/cert.pem"
    key_path: "/path/to/key.pem"
    
redis:
  url: "${REDIS_URL}"
  password: "${REDIS_PASSWORD}"
  pool:
    size: 10
    timeout: 5
    retry_on_timeout: true
    
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expire_minutes: 30
  cors:
    origins:
      - "https://app.pixelatedempathy.com"
      - "https://admin.pixelatedempathy.com"
    allow_credentials: true
    
rate_limiting:
  enabled: true
  storage: "redis"
  default_limits:
    per_minute: 100
    per_hour: 1000
    per_day: 10000
  tiers:
    free:
      per_minute: 60
    premium:
      per_minute: 300
    enterprise:
      per_minute: 1000
      
external_services:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    max_tokens: 2048
    temperature: 0.7
  huggingface:
    token: "${HUGGINGFACE_TOKEN}"
    cache_dir: "/app/models"
    
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
  logging:
    level: "INFO"
    format: "json"
    file: "/var/log/pixelated-ai.log"
  sentry:
    dsn: "${SENTRY_DSN}"
    environment: "production"
    sample_rate: 1.0
```

---

## Database Configuration

### PostgreSQL Production Settings

```bash
# Connection Settings
DATABASE_URL=postgresql://pixelated_user:secure_password@prod-db.amazonaws.com:5432/pixelated_prod

# Pool Configuration
DATABASE_POOL_SIZE=20                 # Connections per worker
DATABASE_MAX_OVERFLOW=30              # Additional connections
DATABASE_POOL_TIMEOUT=30              # Wait time for connection
DATABASE_POOL_RECYCLE=3600            # Recycle connections every hour
DATABASE_POOL_PRE_PING=true           # Validate connections

# SSL Configuration
DATABASE_SSL_MODE=require             # Require SSL
DATABASE_SSL_CERT=/etc/ssl/client-cert.pem
DATABASE_SSL_KEY=/etc/ssl/client-key.pem
DATABASE_SSL_ROOT_CERT=/etc/ssl/ca-cert.pem

# Query Configuration
DATABASE_ECHO=false                   # Don't log SQL queries in production
DATABASE_ECHO_POOL=false             # Don't log pool events
DATABASE_STATEMENT_TIMEOUT=30000      # Statement timeout (30 seconds)
DATABASE_LOCK_TIMEOUT=10000           # Lock timeout (10 seconds)
```

### Database Migration Settings

```bash
# Alembic Configuration
ALEMBIC_CONFIG_FILE=alembic.ini       # Alembic config file
ALEMBIC_SCRIPT_LOCATION=migrations    # Migration scripts directory
ALEMBIC_VERSION_TABLE=alembic_version # Version tracking table
ALEMBIC_AUTO_MIGRATE=false           # Auto-migrate on startup (not recommended for prod)
```

---

## Redis Configuration

### Redis Connection Settings

```bash
# Basic Connection
REDIS_URL=redis://username:password@redis-cluster.amazonaws.com:6379/0
REDIS_PASSWORD=your_secure_redis_password
REDIS_USERNAME=default               # Redis 6+ ACL username
REDIS_DB=0                          # Database number

# Connection Pool
REDIS_POOL_SIZE=10                  # Connection pool size
REDIS_POOL_TIMEOUT=5                # Connection timeout
REDIS_SOCKET_TIMEOUT=5              # Socket timeout
REDIS_SOCKET_CONNECT_TIMEOUT=5      # Connect timeout
REDIS_RETRY_ON_TIMEOUT=true         # Retry on timeout
REDIS_HEALTH_CHECK_INTERVAL=30      # Health check interval
```

### Redis Cluster Configuration

```bash
# Cluster Settings
REDIS_CLUSTER_ENABLED=true
REDIS_CLUSTER_NODES=redis1.cluster:6379,redis2.cluster:6379,redis3.cluster:6379
REDIS_CLUSTER_SKIP_FULL_COVERAGE_CHECK=false
REDIS_CLUSTER_MAX_CONNECTIONS=32
REDIS_CLUSTER_READONLY_MODE=false
```

### Redis Usage Configuration

```bash
# Caching Configuration
CACHE_DEFAULT_TTL=3600              # Default cache TTL (1 hour)
CACHE_MAX_KEY_LENGTH=250            # Maximum key length
CACHE_KEY_PREFIX=pixelated:         # Key prefix for namespacing

# Session Configuration
SESSION_REDIS_DB=1                  # Separate DB for sessions
SESSION_TTL=86400                   # Session TTL (24 hours)
SESSION_KEY_PREFIX=session:         # Session key prefix

# Rate Limiting Configuration
RATE_LIMIT_REDIS_DB=2              # Separate DB for rate limiting
RATE_LIMIT_KEY_PREFIX=ratelimit:   # Rate limit key prefix
```

---

## API Configuration

### Server Settings

```bash
# FastAPI Configuration
API_TITLE="Pixelated Empathy AI"
API_DESCRIPTION="AI-powered empathy and conversation analysis"
API_VERSION=2.0.0
API_DOCS_URL=/docs                  # Swagger UI URL
API_REDOC_URL=/redoc               # ReDoc URL
API_OPENAPI_URL=/openapi.json      # OpenAPI schema URL

# Server Performance
API_WORKERS=4                      # Number of worker processes
API_WORKER_CLASS=uvicorn.workers.UvicornWorker
API_WORKER_CONNECTIONS=1000        # Max connections per worker
API_MAX_REQUESTS=1000              # Max requests per worker
API_MAX_REQUESTS_JITTER=50         # Jitter for max requests
API_TIMEOUT=30                     # Request timeout
API_KEEPALIVE=2                    # Keep-alive timeout
```

### Request/Response Configuration

```bash
# Request Limits
API_MAX_REQUEST_SIZE=10485760      # Max request size (10MB)
API_MAX_FIELD_SIZE=1048576         # Max field size (1MB)
API_MAX_FIELDS=100                 # Max number of fields

# Response Configuration
API_RESPONSE_COMPRESSION=true      # Enable response compression
API_RESPONSE_COMPRESSION_LEVEL=6   # Compression level (1-9)
API_RESPONSE_COMPRESSION_MIN_SIZE=1024 # Min size to compress

# JSON Configuration
API_JSON_SORT_KEYS=false          # Sort JSON keys
API_JSON_ENSURE_ASCII=false       # Ensure ASCII encoding
API_JSON_SEPARATORS=(",", ":")    # JSON separators
```

---

## Security Configuration

### Authentication Settings

```bash
# JWT Configuration
JWT_SECRET_KEY=your-256-bit-secret-key-here-make-it-long-and-random
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30
JWT_REFRESH_EXPIRE_DAYS=7
JWT_ISSUER=pixelatedempathy.com
JWT_AUDIENCE=api.pixelatedempathy.com

# API Key Configuration
API_KEY_ENABLED=true
API_KEY_HEADER=X-API-Key
API_KEY_QUERY_PARAM=api_key
API_KEY_LENGTH=32
API_KEY_PREFIX=pk_live_
API_KEY_TEST_PREFIX=pk_test_
```

### Security Headers

```bash
# Security Headers
SECURITY_HEADERS_ENABLED=true
SECURITY_HSTS_ENABLED=true
SECURITY_HSTS_MAX_AGE=31536000
SECURITY_CONTENT_TYPE_NOSNIFF=true
SECURITY_X_FRAME_OPTIONS=DENY
SECURITY_X_XSS_PROTECTION=1; mode=block
SECURITY_REFERRER_POLICY=strict-origin-when-cross-origin

# Content Security Policy
CSP_ENABLED=true
CSP_DEFAULT_SRC="'self'"
CSP_SCRIPT_SRC="'self' 'unsafe-inline'"
CSP_STYLE_SRC="'self' 'unsafe-inline'"
CSP_IMG_SRC="'self' data: https:"
CSP_FONT_SRC="'self'"
CSP_CONNECT_SRC="'self'"
```

### Encryption Settings

```bash
# Data Encryption
ENCRYPTION_KEY=your-32-byte-encryption-key-here
ENCRYPTION_ALGORITHM=AES-256-GCM
ENCRYPTION_KEY_DERIVATION=PBKDF2
ENCRYPTION_SALT_LENGTH=16
ENCRYPTION_IV_LENGTH=12

# Password Hashing
PASSWORD_HASH_ALGORITHM=bcrypt
PASSWORD_HASH_ROUNDS=12
PASSWORD_SALT_LENGTH=16
```

---

## Monitoring Configuration

### Prometheus Metrics

```bash
# Prometheus Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
PROMETHEUS_PATH=/metrics
PROMETHEUS_NAMESPACE=pixelated_empathy
PROMETHEUS_SUBSYSTEM=api

# Metric Collection
PROMETHEUS_COLLECT_DEFAULT_METRICS=true
PROMETHEUS_COLLECT_GC_METRICS=true
PROMETHEUS_COLLECT_PROCESS_METRICS=true
PROMETHEUS_HISTOGRAM_BUCKETS=0.005,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1.0,2.5,5.0,7.5,10.0
```

### Health Checks

```bash
# Health Check Configuration
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_PATH=/health
HEALTH_CHECK_DETAILED_PATH=/health/detailed
HEALTH_CHECK_LIVENESS_PATH=/health/live
HEALTH_CHECK_READINESS_PATH=/health/ready

# Health Check Timeouts
HEALTH_CHECK_DATABASE_TIMEOUT=5
HEALTH_CHECK_REDIS_TIMEOUT=3
HEALTH_CHECK_EXTERNAL_SERVICE_TIMEOUT=10
```

---

## Deployment Configurations

### Docker Configuration

```bash
# Docker Environment Variables
DOCKER_IMAGE_TAG=v2.0.0
DOCKER_REGISTRY=your-registry.com
DOCKER_NAMESPACE=pixelated-empathy

# Container Configuration
CONTAINER_PORT=8000
CONTAINER_HEALTH_CHECK_INTERVAL=30s
CONTAINER_HEALTH_CHECK_TIMEOUT=10s
CONTAINER_HEALTH_CHECK_RETRIES=3
CONTAINER_MEMORY_LIMIT=2g
CONTAINER_CPU_LIMIT=1000m
```

### Kubernetes Configuration

```bash
# Kubernetes Deployment
K8S_NAMESPACE=pixelated-empathy
K8S_DEPLOYMENT_NAME=pixelated-api
K8S_SERVICE_NAME=pixelated-api-service
K8S_INGRESS_NAME=pixelated-ingress

# Resource Limits
K8S_MEMORY_REQUEST=512Mi
K8S_MEMORY_LIMIT=1Gi
K8S_CPU_REQUEST=250m
K8S_CPU_LIMIT=500m

# Scaling Configuration
K8S_MIN_REPLICAS=3
K8S_MAX_REPLICAS=10
K8S_TARGET_CPU_UTILIZATION=70
K8S_TARGET_MEMORY_UTILIZATION=80
```

---

## Configuration Validation

### Required Variables Checklist

**Critical (Application won't start without these):**
- [ ] `DATABASE_URL`
- [ ] `JWT_SECRET_KEY`
- [ ] `REDIS_URL`

**Important (Application will start but with limited functionality):**
- [ ] `OPENAI_API_KEY`
- [ ] `HUGGINGFACE_TOKEN`
- [ ] `CORS_ORIGINS`

**Optional (Application will use defaults):**
- [ ] `LOG_LEVEL`
- [ ] `API_WORKERS`
- [ ] `PROMETHEUS_ENABLED`

### Configuration Validation Script

```bash
#!/bin/bash
# validate-config.sh

echo "Validating Pixelated Empathy AI Configuration..."

# Check required variables
required_vars=("DATABASE_URL" "JWT_SECRET_KEY" "REDIS_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: Required variable $var is not set"
        exit 1
    else
        echo "✅ $var is set"
    fi
done

# Check JWT secret key length
if [ ${#JWT_SECRET_KEY} -lt 32 ]; then
    echo "❌ ERROR: JWT_SECRET_KEY must be at least 32 characters long"
    exit 1
fi

# Test database connection
python -c "
import psycopg2
try:
    conn = psycopg2.connect('$DATABASE_URL')
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

# Test Redis connection
python -c "
import redis
try:
    r = redis.from_url('$REDIS_URL')
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

echo "✅ All configuration checks passed!"
```

---

*Last updated: 2025-08-12T21:38:00Z*  
*For configuration support, contact the DevOps team or create an issue on GitHub.*
