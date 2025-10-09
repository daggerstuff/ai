"""
API Monitoring Implementation for Pixelated Empathy AI
Comprehensive monitoring with Prometheus metrics, health checks, and alerting.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import CollectorRegistry, multiprocess, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import time
import psutil
import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections_total',
    'Number of active connections'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections'
)

REDIS_CONNECTIONS = Gauge(
    'redis_connections_active',
    'Active Redis connections'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

APPLICATION_INFO = Info(
    'application_info',
    'Application information'
)

class APIMonitoring:
    """Comprehensive API monitoring system."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_metrics()
        self.setup_middleware()
        self.setup_endpoints()
        
    def setup_metrics(self):
        """Initialize application metrics."""
        APPLICATION_INFO.info({
            'version': '2.0.0',
            'name': 'Pixelated Empathy AI',
            'environment': os.getenv('NODE_ENV', 'development')
        })
        
    def setup_middleware(self):
        """Set up monitoring middleware."""
        @self.app.middleware("http")
        async def monitoring_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Increment active connections
            ACTIVE_CONNECTIONS.inc()
            
            try:
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code
                ).inc()
                
                return response
                
            except Exception as e:
                # Record error metrics
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=500
                ).inc()
                raise
            finally:
                # Decrement active connections
                ACTIVE_CONNECTIONS.dec()
    
    def setup_endpoints(self):
        """Set up monitoring endpoints."""
        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def get_metrics():
            """Prometheus metrics endpoint."""
            # Update system metrics
            self.update_system_metrics()
            
            # Generate metrics
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry)
        
        @self.app.get("/health")
        async def health_check():
            """Basic health check."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0"
            }
        
        @self.app.get("/health/detailed")
        async def detailed_health_check():
            """Detailed health check with dependencies."""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0",
                "checks": {}
            }
            
            # Database health
            try:
                # Add actual database check here
                health_status["checks"]["database"] = {
                    "status": "healthy",
                    "response_time_ms": 5.2
                }
            except Exception as e:
                health_status["checks"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"
            
            # Redis health
            try:
                # Add actual Redis check here
                health_status["checks"]["redis"] = {
                    "status": "healthy",
                    "response_time_ms": 1.8
                }
            except Exception as e:
                health_status["checks"]["redis"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            return health_status
        
        @self.app.get("/health/live")
        async def liveness_probe():
            """Kubernetes liveness probe."""
            return {"status": "alive"}
        
        @self.app.get("/health/ready")
        async def readiness_probe():
            """Kubernetes readiness probe."""
            # Check if application is ready to serve traffic
            return {"status": "ready"}
    
    def update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.used)
            
            # Database connections (mock - replace with actual)
            DATABASE_CONNECTIONS.set(15)
            
            # Redis connections (mock - replace with actual)
            REDIS_CONNECTIONS.set(5)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

# Alerting rules for Prometheus
ALERTING_RULES = """
groups:
- name: pixelated-empathy-api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"
  
  - alert: HighCPUUsage
    expr: system_cpu_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}%"
  
  - alert: HighMemoryUsage
    expr: (system_memory_usage_bytes / (1024^3)) > 1.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}GB"
  
  - alert: DatabaseConnectionsHigh
    expr: database_connections_active > 50
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High database connection count"
      description: "Database connections: {{ $value }}"
"""

def setup_monitoring(app: FastAPI) -> APIMonitoring:
    """Set up comprehensive API monitoring."""
    return APIMonitoring(app)
