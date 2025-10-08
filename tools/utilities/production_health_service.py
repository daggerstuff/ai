#!/usr/bin/env python3
"""
Production Health Check Service
Comprehensive health monitoring for production deployment
"""

import json
import time
import psutil
import requests
from datetime import datetime
from pathlib import Path

class ProductionHealthService:
    """Production health check service."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def check_system_health(self):
        """Check system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy" if cpu_percent < 80 and memory.percent < 85 else "degraded",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime": time.time() - self.start_time
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def check_dependencies(self):
        """Check critical dependencies."""
        dependencies = {
            "bcrypt": False,
            "prometheus_client": False,
            "redis": False,
            "cryptography": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
        
        all_available = all(dependencies.values())
        return {
            "status": "healthy" if all_available else "degraded",
            "dependencies": dependencies
        }
    
    def get_comprehensive_health(self):
        """Get comprehensive health status."""
        system_health = self.check_system_health()
        dependency_health = self.check_dependencies()
        
        overall_status = "healthy"
        if system_health["status"] != "healthy" or dependency_health["status"] != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "system": system_health,
            "dependencies": dependency_health,
            "version": "1.0.0",
            "environment": "production"
        }

if __name__ == "__main__":
    health_service = ProductionHealthService()
    health_data = health_service.get_comprehensive_health()
    print(json.dumps(health_data, indent=2))
