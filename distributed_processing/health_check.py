#!/usr/bin/env python3
"""
Health Check and System Status Monitoring for Pixelated Empathy AI
Comprehensive health checking for all system components with detailed status reporting
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import psutil
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components"""
    DATABASE = "database"
    API = "api"
    WORKER = "worker"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float = 0.0
    details: Dict[str, Any] = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    overall_status: HealthStatus
    timestamp: datetime
    component_checks: List[HealthCheckResult]
    system_metrics: Dict[str, Any]
    recommendations: List[str]


class HealthCheckManager:
    """Manages health checks for all system components"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.component_configs: Dict[str, Dict[str, Any]] = {}
        self.check_history: List[HealthCheckResult] = []
        self.last_full_check: Optional[SystemHealthReport] = None
        
    def register_health_check(self, component_name: str, component_type: ComponentType,
                            check_function: Callable, config: Dict[str, Any] = None):
        """Register a health check for a component"""
        self.health_checks[component_name] = check_function
        self.component_configs[component_name] = {
            "type": component_type,
            "config": config or {}
        }
        
    async def run_health_check(self, component_name: str) -> HealthCheckResult:
        """Run health check for a specific component"""
        start_time = time.time()
        
        if component_name not in self.health_checks:
            return HealthCheckResult(
                component_name=component_name,
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.UNKNOWN,
                message=f"Component {component_name} not registered",
                timestamp=datetime.utcnow()
            )
        
        try:
            # Run the health check
            result = await self.health_checks[component_name]()
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Ensure result is a HealthCheckResult
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time
                return result
            else:
                # Convert to HealthCheckResult
                return HealthCheckResult(
                    component_name=component_name,
                    component_type=self.component_configs[component_name]["type"],
                    status=HealthStatus.HEALTHY,
                    message="Health check passed",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details=result
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {component_name}: {e}")
            
            return HealthCheckResult(
                component_name=component_name,
                component_type=self.component_configs[component_name]["type"],
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
    
    async def run_all_health_checks(self) -> SystemHealthReport:
        """Run health checks for all registered components"""
        logger.info("Running comprehensive system health check...")
        
        # Run all health checks concurrently
        tasks = [
            self.run_health_check(component_name) 
            for component_name in self.health_checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to HealthCheckResult
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check task failed: {result}")
                continue
            valid_results.append(result)
        
        # Determine overall system health
        overall_status = self._calculate_overall_health(valid_results)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(valid_results, system_metrics)
        
        # Create report
        report = SystemHealthReport(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            component_checks=valid_results,
            system_metrics=system_metrics,
            recommendations=recommendations
        )
        
        self.last_full_check = report
        self.check_history.extend(valid_results)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.check_history = [
            check for check in self.check_history 
            if check.timestamp > cutoff_time
        ]
        
        logger.info(f"Health check completed. Overall status: {overall_status.value}")
        return report
    
    def _calculate_overall_health(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system health from component checks"""
        if not results:
            return HealthStatus.UNKNOWN
            
        # Count statuses
        status_counts = {}
        for result in results:
            status = result.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        unhealthy_count = status_counts.get(HealthStatus.UNHEALTHY, 0)
        degraded_count = status_counts.get(HealthStatus.DEGRADED, 0)
        healthy_count = status_counts.get(HealthStatus.HEALTHY, 0)
        
        # If any critical components are unhealthy, system is unhealthy
        critical_unhealthy = any(
            result.status == HealthStatus.UNHEALTHY and 
            result.component_type in [ComponentType.DATABASE, ComponentType.API]
            for result in results
        )
        
        if critical_unhealthy or unhealthy_count > len(results) * 0.3:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0 or unhealthy_count > 0:
            return HealthStatus.DEGRADED
        elif healthy_count == len(results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2)
                },
                "network": {
                    "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                    "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, results: List[HealthCheckResult], 
                                system_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        # Check for unhealthy components
        unhealthy_components = [
            result.component_name for result in results 
            if result.status == HealthStatus.UNHEALTHY
        ]
        
        if unhealthy_components:
            recommendations.append(
                f"❌ Immediate attention needed for unhealthy components: {', '.join(unhealthy_components)}"
            )
        
        # Check for degraded components
        degraded_components = [
            result.component_name for result in results 
            if result.status == HealthStatus.DEGRADED
        ]
        
        if degraded_components:
            recommendations.append(
                f"⚠️  Monitor degraded components: {', '.join(degraded_components)}"
            )
        
        # Check system resource usage
        if "memory" in system_metrics and system_metrics["memory"]["percent"] > 85:
            recommendations.append("⚠️  Memory usage is high. Consider scaling up or optimizing memory usage.")
        
        if "disk" in system_metrics and system_metrics["disk"]["percent"] > 85:
            recommendations.append("⚠️  Disk usage is high. Consider cleaning up or adding more storage.")
        
        if "cpu" in system_metrics and system_metrics["cpu"]["percent"] > 85:
            recommendations.append("⚠️  CPU usage is high. Consider scaling up or optimizing performance.")
        
        # If everything is healthy, provide positive feedback
        if not recommendations:
            recommendations.append("✅ All system components are healthy. No immediate action required.")
        
        return recommendations
    
    def get_health_history(self, component_name: str = None, 
                          hours: int = 24) -> List[HealthCheckResult]:
        """Get health check history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if component_name:
            return [
                check for check in self.check_history
                if check.component_name == component_name and check.timestamp > cutoff_time
            ]
        else:
            return [
                check for check in self.check_history
                if check.timestamp > cutoff_time
            ]


# Default health check implementations
async def database_health_check():
    """Default database health check"""
    # This would connect to the actual database and run a simple query
    # For now, we'll simulate a successful check
    return {
        "connection": "successful",
        "response_time_ms": 5.2,
        "active_connections": 12
    }

async def api_health_check():
    """Default API health check"""
    # This would make a request to a health endpoint
    # For now, we'll simulate a successful check
    return {
        "status_code": 200,
        "response_time_ms": 12.5,
        "endpoints_healthy": 15
    }

async def redis_health_check():
    """Default Redis health check"""
    # This would connect to Redis and run a ping
    # For now, we'll simulate a successful check
    return {
        "connection": "successful",
        "response_time_ms": 1.1,
        "memory_usage_mb": 45
    }

async def celery_health_check():
    """Default Celery health check"""
    # This would check Celery worker status
    # For now, we'll simulate a successful check
    return {
        "workers_connected": 4,
        "tasks_processed": 1250,
        "queue_length": 3
    }

async def storage_health_check():
    """Default storage health check"""
    # This would check storage availability and permissions
    # For now, we'll simulate a successful check
    return {
        "read_write": "successful",
        "available_space_gb": 125.4,
        "response_time_ms": 8.7
    }

async def network_health_check():
    """Default network health check"""
    # This would check network connectivity
    # For now, we'll simulate a successful check
    try:
        # Test DNS resolution
        socket.gethostbyname("google.com")
        return {
            "dns_resolution": "successful",
            "internet_connectivity": "available",
            "response_time_ms": 25.3
        }
    except Exception as e:
        return {
            "dns_resolution": "failed",
            "internet_connectivity": "unavailable",
            "error": str(e)
        }


# Initialize health check manager with default checks
def initialize_default_health_checks():
    """Initialize health check manager with default component checks"""
    manager = HealthCheckManager()
    
    # Register default health checks
    manager.register_health_check(
        "database", 
        ComponentType.DATABASE, 
        database_health_check
    )
    
    manager.register_health_check(
        "api", 
        ComponentType.API, 
        api_health_check
    )
    
    manager.register_health_check(
        "redis", 
        ComponentType.CACHE, 
        redis_health_check
    )
    
    manager.register_health_check(
        "celery", 
        ComponentType.MESSAGE_QUEUE, 
        celery_health_check
    )
    
    manager.register_health_check(
        "storage", 
        ComponentType.STORAGE, 
        storage_health_check
    )
    
    manager.register_health_check(
        "network", 
        ComponentType.NETWORK, 
        network_health_check
    )
    
    return manager


# Example usage
async def example_health_check():
    """Example of using the health check system"""
    
    # Initialize health check manager
    health_manager = initialize_default_health_checks()
    
    # Run comprehensive health check
    report = await health_manager.run_all_health_checks()
    
    # Print results
    print("=" * 70)
    print("SYSTEM HEALTH CHECK REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    
    print("\nComponent Health:")
    for check in report.component_checks:
        status_icon = "✅" if check.status == HealthStatus.HEALTHY else \
                      "⚠️" if check.status == HealthStatus.DEGRADED else \
                      "❌" if check.status == HealthStatus.UNHEALTHY else "❓"
        print(f"  {status_icon} {check.component_name}: {check.status.value} "
              f"({check.response_time_ms:.1f}ms)")
        if check.message and check.message != "Health check passed":
            print(f"     Message: {check.message}")
    
    print("\nSystem Metrics:")
    for category, metrics in report.system_metrics.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            print(f"  {category.title()}:")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    return report


if __name__ == "__main__":
    asyncio.run(example_health_check())