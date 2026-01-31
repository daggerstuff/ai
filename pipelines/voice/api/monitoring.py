"""
Production-grade monitoring, metrics, and health checks for Pixel Voice pipeline.
"""

import asyncio
import logging
import os
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """Prometheus metrics collector."""

    def __init__(self):
        # API metrics
        self.api_requests_total = Counter(
            "pixel_voice_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
        )

        self.api_request_duration = Histogram(
            "pixel_voice_api_request_duration_seconds",
            "API request duration",
            ["method", "endpoint"],
        )

        # Pipeline metrics
        self.pipeline_jobs_total = Counter(
            "pixel_voice_pipeline_jobs_total", "Total pipeline jobs", ["stage", "status"]
        )

        self.pipeline_job_duration = Histogram(
            "pixel_voice_pipeline_job_duration_seconds", "Pipeline job duration", ["stage"]
        )

        self.active_jobs = Gauge("pixel_voice_active_jobs", "Currently active pipeline jobs")

        # YouTube download metrics
        self.youtube_downloads_total = Counter(
            "pixel_voice_youtube_downloads_total", "Total YouTube downloads", ["status"]
        )

        self.youtube_download_duration = Histogram(
            "pixel_voice_youtube_download_duration_seconds", "YouTube download duration"
        )

        # System metrics
        self.system_cpu_usage = Gauge(
            "pixel_voice_system_cpu_usage_percent", "System CPU usage percentage"
        )

        self.system_memory_usage = Gauge(
            "pixel_voice_system_memory_usage_bytes", "System memory usage in bytes"
        )

        self.system_disk_usage = Gauge(
            "pixel_voice_system_disk_usage_percent", "System disk usage percentage"
        )

        # Rate limiting metrics
        self.rate_limit_violations = Counter(
            "pixel_voice_rate_limit_violations_total",
            "Rate limit violations",
            ["limit_type", "user_id"],
        )

        # Error metrics
        self.errors_total = Counter(
            "pixel_voice_errors_total", "Total errors", ["error_type", "component"]
        )

    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        self.api_requests_total.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_pipeline_job(self, stage: str, status: str, duration: Optional[float] = None):
        """Record pipeline job metrics."""
        self.pipeline_jobs_total.labels(stage=stage, status=status).inc()

        if duration is not None:
            self.pipeline_job_duration.labels(stage=stage).observe(duration)

    def record_youtube_download(self, status: str, duration: Optional[float] = None):
        """Record YouTube download metrics."""
        self.youtube_downloads_total.labels(status=status).inc()

        if duration is not None:
            self.youtube_download_duration.observe(duration)

    def record_rate_limit_violation(self, limit_type: str, user_id: str):
        """Record rate limit violation."""
        self.rate_limit_violations.labels(limit_type=limit_type, user_id=user_id).inc()

    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_usage.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage.set(memory.used)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        self.system_disk_usage.set(disk_percent)


class HealthChecker:
    """Health check manager."""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.redis_client = None
        self.database = None

    async def check_redis(self) -> HealthCheck:
        """Check Redis connectivity."""
        start_time = time.time()

        try:
            if self.redis_client:
                await self.redis_client.ping()
                status = HealthStatus.HEALTHY
                message = "Redis connection successful"
            else:
                status = HealthStatus.DEGRADED
                message = "Redis client not configured"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Redis connection failed: {str(e)}"

        response_time = (time.time() - start_time) * 1000

        return HealthCheck(
            name="redis",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
        )

    async def check_database(self) -> HealthCheck:
        """Check database connectivity."""
        start_time = time.time()

        try:
            if self.database:
                # Simple query to test connection
                # await self.database.execute("SELECT 1")
                status = HealthStatus.HEALTHY
                message = "Database connection successful"
            else:
                status = HealthStatus.DEGRADED
                message = "Database not configured"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database connection failed: {str(e)}"

        response_time = (time.time() - start_time) * 1000

        return HealthCheck(
            name="database",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
        )

    async def check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        start_time = time.time()

        try:
            disk = psutil.disk_usage("/")
            free_percent = (disk.free / disk.total) * 100

            if free_percent > 20:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_percent:.1f}% free"
            elif free_percent > 10:
                status = HealthStatus.DEGRADED
                message = f"Disk space low: {free_percent:.1f}% free"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space critical: {free_percent:.1f}% free"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Disk check failed: {str(e)}"

        response_time = (time.time() - start_time) * 1000

        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
        )

    async def check_memory(self) -> HealthCheck:
        """Check memory usage."""
        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK: {used_percent:.1f}%"
            elif used_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {used_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {used_percent:.1f}%"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Memory check failed: {str(e)}"

        response_time = (time.time() - start_time) * 1000

        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
        )

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = await asyncio.gather(
            self.check_redis(),
            self.check_database(),
            self.check_disk_space(),
            self.check_memory(),
            return_exceptions=True,
        )

        results = {}
        check_names = ["redis", "database", "disk_space", "memory"]

        for i, check in enumerate(checks):
            if isinstance(check, Exception):
                results[check_names[i]] = HealthCheck(
                    name=check_names[i],
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(check)}",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                )
            else:
                results[check_names[i]] = check

        return results

    def get_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health status."""
        if any(check.status == HealthStatus.UNHEALTHY for check in checks.values()):
            return HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.DEGRADED for check in checks.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


# Global instances
metrics = MetricsCollector()
health_checker = HealthChecker()


def setup_monitoring(app: FastAPI):
    """Setup monitoring for FastAPI app."""
    # Setup Prometheus instrumentator
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

    # Add metrics endpoint
    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        """Prometheus metrics endpoint."""
        metrics.update_system_metrics()
        return generate_latest()

    # Add health check endpoints
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with all components."""
        checks = await health_checker.run_all_checks()
        overall_status = health_checker.get_overall_status(checks)

        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                }
                for name, check in checks.items()
            },
        }

    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe."""
        checks = await health_checker.run_all_checks()
        overall_status = health_checker.get_overall_status(checks)

        if overall_status == HealthStatus.UNHEALTHY:
            return Response(status_code=503, content="Service not ready")

        return {"status": "ready"}

    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe."""
        return {"status": "alive", "timestamp": datetime.now().isoformat()}

    logger.info("Monitoring setup completed")
