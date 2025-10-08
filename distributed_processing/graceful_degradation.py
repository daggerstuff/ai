#!/usr/bin/env python3
"""
Graceful Degradation System for Pixelated Empathy AI
Handles partial system failures with intelligent service prioritization and fallbacks
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServicePriority(Enum):
    """Service priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ServiceStatus(Enum):
    """Service status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"

@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    priority: ServicePriority
    fallback_handler: Optional[Callable] = None
    health_check: Optional[Callable] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class GracefulDegradationManager:
    """Manages graceful degradation during partial system failures"""
    
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.degradation_active = False
        self.disabled_services: set = set()
    
    def register_service(self, config: ServiceConfig):
        """Register a service for degradation management"""
        self.services[config.name] = config
        self.service_status[config.name] = ServiceStatus.HEALTHY
    
    async def handle_service_failure(self, service_name: str):
        """Handle failure of a specific service"""
        if service_name not in self.services:
            return
        
        self.service_status[service_name] = ServiceStatus.FAILED
        
        # Determine degradation strategy
        service_config = self.services[service_name]
        
        if service_config.priority == ServicePriority.CRITICAL:
            # Critical service failed - enable system-wide degradation
            await self._enable_system_degradation()
        else:
            # Non-critical service - disable dependent services
            await self._disable_dependent_services(service_name)
    
    async def _enable_system_degradation(self):
        """Enable system-wide graceful degradation"""
        self.degradation_active = True
        
        # Disable low priority services first
        for name, config in self.services.items():
            if config.priority == ServicePriority.LOW:
                await self._disable_service(name)
        
        logger.warning("System-wide graceful degradation activated")
    
    async def _disable_service(self, service_name: str):
        """Disable a service gracefully"""
        self.disabled_services.add(service_name)
        self.service_status[service_name] = ServiceStatus.DISABLED
        logger.info(f"Service {service_name} disabled for graceful degradation")
    
    async def call_service(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """Call service with degradation handling"""
        
        if service_name in self.disabled_services:
            # Service is disabled - use fallback
            return await self._use_fallback(service_name, *args, **kwargs)
        
        try:
            result = await func(*args, **kwargs)
            # Mark service as healthy on success
            self.service_status[service_name] = ServiceStatus.HEALTHY
            return result
            
        except Exception as e:
            # Service call failed
            await self.handle_service_failure(service_name)
            return await self._use_fallback(service_name, *args, **kwargs)
    
    async def _use_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Use fallback for failed/disabled service"""
        
        if service_name not in self.services:
            raise Exception(f"Service {service_name} not registered")
        
        config = self.services[service_name]
        
        if config.fallback_handler:
            try:
                return await config.fallback_handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback failed for {service_name}: {e}")
                raise
        else:
            raise Exception(f"Service {service_name} unavailable and no fallback configured")

# Example usage
async def example_graceful_degradation():
    """Example of graceful degradation system"""
    
    manager = GracefulDegradationManager()
    
    # Fallback handlers
    async def ai_processing_fallback(*args, **kwargs):
        return {"result": "cached_response", "source": "fallback"}
    
    async def analytics_fallback(*args, **kwargs):
        return {"analytics": "disabled", "reason": "service_unavailable"}
    
    # Register services
    manager.register_service(ServiceConfig(
        name="ai_processing",
        priority=ServicePriority.CRITICAL,
        fallback_handler=ai_processing_fallback
    ))
    
    manager.register_service(ServiceConfig(
        name="analytics",
        priority=ServicePriority.LOW,
        fallback_handler=analytics_fallback
    ))
    
    # Simulate service calls
    async def ai_service_call():
        raise Exception("AI service temporarily unavailable")
    
    async def analytics_service_call():
        return {"metrics": "processed"}
    
    # Test degradation
    try:
        result = await manager.call_service("ai_processing", ai_service_call)
        print(f"AI result: {result}")
    except Exception as e:
        print(f"AI failed: {e}")
    
    try:
        result = await manager.call_service("analytics", analytics_service_call)
        print(f"Analytics result: {result}")
    except Exception as e:
        print(f"Analytics failed: {e}")

if __name__ == "__main__":
    asyncio.run(example_graceful_degradation())
