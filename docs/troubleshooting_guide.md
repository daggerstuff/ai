# Pixelated Empathy AI - Troubleshooting Guide

**Version:** 2.0.0  
**Last Updated:** 2025-08-12T21:34:00Z  
**Target Audience:** Developers, DevOps Engineers, Support Teams

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Common Issues](#common-issues)
- [API Issues](#api-issues)
- [Database Issues](#database-issues)
- [Authentication Issues](#authentication-issues)
- [Performance Issues](#performance-issues)
- [Deployment Issues](#deployment-issues)
- [Monitoring & Logging](#monitoring--logging)
- [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostics

### Health Check Commands

```bash
# Basic health check
curl -f http://localhost:8000/health || echo "API is down"

# Detailed health check
curl -s http://localhost:8000/health/detailed | jq '.'

# Check all services
docker-compose ps
kubectl get pods -n pixelated-empathy
```

### System Status Overview

```bash
# Check system resources
df -h                    # Disk usage
free -h                  # Memory usage
top -bn1 | head -20      # CPU usage

# Check network connectivity
ping google.com
nslookup api.pixelatedempathy.com

# Check service ports
netstat -tlnp | grep :8000
ss -tlnp | grep :8000
```

---

## Common Issues

### Issue 1: Application Won't Start

**Symptoms:**
- Container exits immediately
- "Connection refused" errors
- Import errors in logs

**Diagnosis:**
```bash
# Check application logs
docker logs pixelated-api
kubectl logs deployment/pixelated-api -n pixelated-empathy

# Check environment variables
docker exec pixelated-api env | grep -E "(DATABASE|REDIS|JWT)"

# Test Python imports
docker exec pixelated-api python -c "import pixel_voice.api.server"
```

**Solutions:**

1. **Missing Environment Variables:**
```bash
# Check required variables are set
echo $DATABASE_URL
echo $JWT_SECRET_KEY
echo $REDIS_URL

# Set missing variables
export JWT_SECRET_KEY="your-secret-key"
```

2. **Database Connection Issues:**
```bash
# Test database connectivity
docker exec pixelated-api python -c "
import psycopg2
try:
    conn = psycopg2.connect('$DATABASE_URL')
    print('Database connected successfully')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

3. **Port Already in Use:**
```bash
# Find process using port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port
export API_PORT=8001
```

### Issue 2: High Memory Usage

**Symptoms:**
- OOMKilled pods in Kubernetes
- Slow response times
- Memory warnings in logs

**Diagnosis:**
```bash
# Check memory usage
docker stats pixelated-api
kubectl top pods -n pixelated-empathy

# Check memory leaks
docker exec pixelated-api python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**

1. **Increase Memory Limits:**
```yaml
# In Kubernetes deployment
resources:
  limits:
    memory: "2Gi"
  requests:
    memory: "1Gi"
```

2. **Optimize Database Connections:**
```python
# In database configuration
DATABASE_POOL_SIZE = 5  # Reduce from default 20
DATABASE_MAX_OVERFLOW = 10  # Reduce from default 30
```

3. **Enable Garbage Collection:**
```python
# Add to application startup
import gc
gc.set_threshold(700, 10, 10)  # More aggressive GC
```

### Issue 3: Slow API Response Times

**Symptoms:**
- Response times > 5 seconds
- Timeout errors
- High CPU usage

**Diagnosis:**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# curl-format.txt content:
#     time_namelookup:  %{time_namelookup}\n
#        time_connect:  %{time_connect}\n
#     time_appconnect:  %{time_appconnect}\n
#    time_pretransfer:  %{time_pretransfer}\n
#       time_redirect:  %{time_redirect}\n
#  time_starttransfer:  %{time_starttransfer}\n
#                     ----------\n
#          time_total:  %{time_total}\n

# Check database query performance
docker exec postgres psql -U postgres -d pixelated_prod -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"
```

**Solutions:**

1. **Database Optimization:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_conversations_user_created 
ON conversations(user_id, created_at);

-- Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM conversations 
WHERE user_id = 'user123' 
ORDER BY created_at DESC 
LIMIT 50;
```

2. **Enable Caching:**
```python
# Add Redis caching to frequently accessed data
@cache.memoize(timeout=300)  # 5 minutes
def get_user_conversations(user_id: str):
    # Database query here
    pass
```

3. **Increase Worker Processes:**
```bash
# For Docker deployment
export API_WORKERS=4

# For Kubernetes
kubectl scale deployment pixelated-api --replicas=3 -n pixelated-empathy
```

---

## API Issues

### Issue: 401 Unauthorized Errors

**Symptoms:**
- All API requests return 401
- "Invalid token" messages
- Authentication failures

**Diagnosis:**
```bash
# Test token generation
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'

# Verify JWT secret
echo $JWT_SECRET_KEY | wc -c  # Should be > 32 characters

# Check token expiration
python -c "
import jwt
token = 'your-jwt-token-here'
try:
    payload = jwt.decode(token, verify=False)
    print(f'Token expires at: {payload.get(\"exp\")}')
except Exception as e:
    print(f'Token decode error: {e}')
"
```

**Solutions:**

1. **Fix JWT Configuration:**
```bash
# Generate new secret key
export JWT_SECRET_KEY=$(openssl rand -base64 32)

# Restart application
docker-compose restart api
```

2. **Check System Time:**
```bash
# Ensure system time is correct
timedatectl status
ntpdate -s time.nist.gov
```

### Issue: 429 Rate Limited Errors

**Symptoms:**
- "Too Many Requests" errors
- Rate limit headers in response
- Legitimate requests being blocked

**Diagnosis:**
```bash
# Check Redis rate limit data
docker exec redis redis-cli keys "*rate_limit*"
docker exec redis redis-cli get "rate_limit:user123"

# Check rate limit configuration
grep -r "RATE_LIMIT" .env
```

**Solutions:**

1. **Adjust Rate Limits:**
```python
# In rate_limiting.py
RATE_LIMIT_REQUESTS_PER_MINUTE = 1000  # Increase from 100
RATE_LIMIT_BURST_SIZE = 200  # Increase burst allowance
```

2. **Whitelist IP Addresses:**
```python
# Add IP whitelist
RATE_LIMIT_WHITELIST = [
    "192.168.1.0/24",  # Internal network
    "10.0.0.0/8"       # VPC network
]
```

---

## Database Issues

### Issue: Connection Pool Exhausted

**Symptoms:**
- "Connection pool exhausted" errors
- Long database query times
- Application hangs

**Diagnosis:**
```bash
# Check active connections
docker exec postgres psql -U postgres -c "
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';
"

# Check connection pool settings
grep -E "(POOL_SIZE|MAX_OVERFLOW)" .env
```

**Solutions:**

1. **Increase Pool Size:**
```bash
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=30
export DATABASE_POOL_TIMEOUT=30
```

2. **Close Idle Connections:**
```sql
-- Kill idle connections older than 1 hour
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND state_change < now() - interval '1 hour';
```

### Issue: Database Migration Failures

**Symptoms:**
- Migration scripts fail
- Schema version mismatches
- Data corruption warnings

**Diagnosis:**
```bash
# Check migration status
docker exec pixelated-api python -m alembic current
docker exec pixelated-api python -m alembic history

# Check database schema
docker exec postgres psql -U postgres -d pixelated_prod -c "\dt"
```

**Solutions:**

1. **Manual Migration:**
```bash
# Run specific migration
docker exec pixelated-api python -m alembic upgrade +1

# Rollback migration
docker exec pixelated-api python -m alembic downgrade -1
```

2. **Reset Database (Development Only):**
```bash
# WARNING: This will delete all data
docker-compose down -v
docker-compose up -d db
docker exec pixelated-api python -m alembic upgrade head
```

---

## Authentication Issues

### Issue: JWT Token Validation Failures

**Symptoms:**
- Valid tokens rejected
- "Token has expired" for new tokens
- Inconsistent authentication behavior

**Diagnosis:**
```bash
# Check JWT configuration
echo "Secret: $JWT_SECRET_KEY"
echo "Algorithm: $JWT_ALGORITHM"
echo "Expiry: $JWT_EXPIRE_MINUTES minutes"

# Decode token manually
python -c "
import jwt
import json
token = 'your-token-here'
try:
    # Decode without verification to see payload
    payload = jwt.decode(token, options={'verify_signature': False})
    print(json.dumps(payload, indent=2))
except Exception as e:
    print(f'Error: {e}')
"
```

**Solutions:**

1. **Synchronize JWT Settings:**
```bash
# Ensure all instances use same secret
kubectl create secret generic jwt-secret \
  --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
  -n pixelated-empathy
```

2. **Fix Time Synchronization:**
```bash
# Install NTP
sudo apt-get install ntp
sudo systemctl enable ntp
sudo systemctl start ntp
```

---

## Performance Issues

### Issue: High CPU Usage

**Symptoms:**
- CPU usage consistently > 80%
- Slow response times
- Application timeouts

**Diagnosis:**
```bash
# Check CPU usage by process
top -p $(pgrep -f "uvicorn")
htop

# Profile Python application
docker exec pixelated-api python -m cProfile -o profile.stats -m uvicorn pixel_voice.api.server:app
```

**Solutions:**

1. **Optimize Code:**
```python
# Use async/await for I/O operations
async def get_user_data(user_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/users/{user_id}") as response:
            return await response.json()
```

2. **Scale Horizontally:**
```bash
# Add more replicas
kubectl scale deployment pixelated-api --replicas=5 -n pixelated-empathy
```

### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increases
- OOMKilled containers
- Garbage collection warnings

**Diagnosis:**
```bash
# Monitor memory over time
while true; do
  docker stats pixelated-api --no-stream | grep pixelated-api
  sleep 60
done

# Use memory profiler
pip install memory-profiler
python -m memory_profiler your_script.py
```

**Solutions:**

1. **Fix Memory Leaks:**
```python
# Properly close database connections
async def get_data():
    async with get_db_session() as session:
        # Use session here
        pass  # Session automatically closed
```

2. **Implement Circuit Breaker:**
```python
# Prevent memory buildup from failed requests
from circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, recovery_timeout=30)
async def external_api_call():
    # API call here
    pass
```

---

## Deployment Issues

### Issue: Kubernetes Pod CrashLoopBackOff

**Symptoms:**
- Pods continuously restart
- CrashLoopBackOff status
- Application never becomes ready

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n pixelated-empathy
kubectl describe pod <pod-name> -n pixelated-empathy

# Check logs
kubectl logs <pod-name> -n pixelated-empathy --previous
```

**Solutions:**

1. **Fix Readiness Probe:**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 30  # Increase delay
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

2. **Check Resource Limits:**
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"  # Increase if needed
    cpu: "500m"
```

### Issue: Ingress Not Working

**Symptoms:**
- 404 errors from load balancer
- SSL certificate issues
- Traffic not reaching pods

**Diagnosis:**
```bash
# Check ingress status
kubectl get ingress -n pixelated-empathy
kubectl describe ingress pixelated-ingress -n pixelated-empathy

# Check ingress controller
kubectl get pods -n ingress-nginx
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

**Solutions:**

1. **Fix Ingress Configuration:**
```yaml
# Ensure correct service name and port
spec:
  rules:
  - host: api.pixelatedempathy.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pixelated-api-service  # Correct service name
            port:
              number: 80  # Correct port
```

2. **Check DNS Configuration:**
```bash
# Verify DNS resolution
nslookup api.pixelatedempathy.com
dig api.pixelatedempathy.com
```

---

## Monitoring & Logging

### Centralized Logging Setup

```bash
# View application logs
kubectl logs -f deployment/pixelated-api -n pixelated-empathy

# Search logs with grep
kubectl logs deployment/pixelated-api -n pixelated-empathy | grep ERROR

# Export logs to file
kubectl logs deployment/pixelated-api -n pixelated-empathy > app.log
```

### Metrics Collection

```bash
# Check Prometheus metrics
curl http://localhost:8000/metrics

# Query specific metrics
curl -s http://localhost:8000/metrics | grep http_requests_total
```

### Alert Configuration

```yaml
# prometheus-alerts.yaml
groups:
- name: pixelated-empathy
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
```

---

## Emergency Procedures

### Emergency Rollback

```bash
# Kubernetes rollback
kubectl rollout undo deployment/pixelated-api -n pixelated-empathy

# Helm rollback
helm rollback pixelated-empathy -n pixelated-empathy

# Docker Compose rollback
docker-compose down
docker-compose up -d --scale api=0  # Stop current version
docker tag pixelated-empathy-ai:backup pixelated-empathy-ai:latest
docker-compose up -d
```

### Emergency Scale Down

```bash
# Scale down to minimum replicas
kubectl scale deployment pixelated-api --replicas=1 -n pixelated-empathy

# Stop all traffic
kubectl patch ingress pixelated-ingress -n pixelated-empathy \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "maintenance-service"}]'
```

### Database Emergency Procedures

```bash
# Create emergency backup
kubectl exec deployment/postgres -n pixelated-empathy -- \
  pg_dump -U postgres pixelated_prod > emergency_backup.sql

# Enable read-only mode
kubectl exec deployment/postgres -n pixelated-empathy -- \
  psql -U postgres -c "ALTER DATABASE pixelated_prod SET default_transaction_read_only = on;"

# Restart database
kubectl rollout restart deployment/postgres -n pixelated-empathy
```

---

## Support Contacts

### Escalation Matrix

| Issue Severity | Contact | Response Time |
|----------------|---------|---------------|
| **Critical** (System Down) | On-call Engineer | 15 minutes |
| **High** (Performance Issues) | DevOps Team | 1 hour |
| **Medium** (Feature Issues) | Development Team | 4 hours |
| **Low** (Questions) | Support Team | 24 hours |

### Emergency Contacts

- **On-call Engineer**: +1-555-ONCALL
- **DevOps Team**: devops@pixelatedempathy.com
- **Development Team**: dev@pixelatedempathy.com
- **Support Team**: support@pixelatedempathy.com

### Useful Links

- **Status Page**: https://status.pixelatedempathy.com
- **Monitoring Dashboard**: https://grafana.pixelatedempathy.com
- **Log Aggregation**: https://logs.pixelatedempathy.com
- **Documentation**: https://docs.pixelatedempathy.com

---

*Last updated: 2025-08-12T21:34:00Z*  
*For additional support, contact the appropriate team or create an issue on GitHub.*
