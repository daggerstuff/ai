"""
Graceful Degradation for Pipeline Communication - Fallback Mechanisms.

This module provides comprehensive graceful degradation with fallback mechanisms,
circuit breakers, and service degradation strategies for the six-stage pipeline.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .event_bus import EventBus, EventMessage, EventType
from ..error_handling.custom_errors import (
    ServiceUnavailableError, CircuitBreakerError, DegradationError
)
from ..utils.logger import get_request_logger


class DegradationLevel(Enum):
    """Service degradation levels."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    half_open_success_threshold: float = 0.8
    fallback_timeout_seconds: float = 30.0
    degradation_levels: Dict[str, Any] = None
    service_health_check_interval: float = 30.0


@dataclass
class ServiceHealth:
    """Service health status."""
    service_name: str
    status: str
    last_check: datetime
    failure_count: int
    success_count: int
    circuit_state: CircuitState
    next_retry_time: Optional[datetime]


@dataclass
class DegradationStrategy:
    """Degradation strategy configuration."""
    service_name: str
    normal_strategy: str
    degraded_strategy: str
    critical_strategy: str
    fallback_enabled: bool = True
    cache_fallback: bool = True
    simplified_processing: bool = True


class GracefulDegradationManager:
    """Comprehensive graceful degradation manager for pipeline services."""
    
    def __init__(self, event_bus: EventBus, config: Optional[Dict[str, Any]] = None):
        """
        Initialize graceful degradation manager.
        
        Args:
            event_bus: Event bus for coordination
            config: Optional configuration dictionary
        """
        self.event_bus = event_bus
        self.config = DegradationConfig(**(config or {}))
        self.logger = get_request_logger()
        
        # Service health tracking
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Degradation strategies
        self.degradation_strategies: Dict[str, DegradationStrategy] = {}
        
        # Fallback caches
        self.fallback_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_fallbacks': 0,
            'successful_fallbacks': 0,
            'circuit_breaker_activations': 0,
            'service_degradations': 0
        }
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        self.logger.info("GracefulDegradationManager initialized")
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default degradation strategies for pipeline services."""
        default_strategies = {
            'bias_detection': DegradationStrategy(
                service_name='bias_detection',
                normal_strategy='full_bias_analysis',
                degraded_strategy='simplified_bias_check',
                critical_strategy='skip_bias_detection',
                fallback_enabled=True,
                cache_fallback=True,
                simplified_processing=True
            ),
            'redis_connection': DegradationStrategy(
                service_name='redis_connection',
                normal_strategy='full_redis_operations',
                degraded_strategy='memory_caching_only',
                critical_strategy='no_caching',
                fallback_enabled=True,
                cache_fallback=False,
                simplified_processing=True
            ),
            'event_processing': DegradationStrategy(
                service_name='event_processing',
                normal_strategy='full_event_handling',
                degraded_strategy='batched_event_processing',
                critical_strategy='event_queuing',
                fallback_enabled=True,
                cache_fallback=True,
                simplified_processing=True
            ),
            'state_persistence': DegradationStrategy(
                service_name='state_persistence',
                normal_strategy='full_state_management',
                degraded_strategy='minimal_state_tracking',
                critical_strategy='in_memory_state_only',
                fallback_enabled=True,
                cache_fallback=True,
                simplified_processing=True
            ),
            'progress_tracking': DegradationStrategy(
                service_name='progress_tracking',
                normal_strategy='real_time_updates',
                degraded_strategy='periodic_updates',
                critical_strategy='completion_notifications_only',
                fallback_enabled=True,
                cache_fallback=True,
                simplified_processing=True
            )
        }
        
        self.degradation_strategies.update(default_strategies)
    
    async def execute_with_fallback(self, service_name: str, primary_func: Callable,
                                  fallback_func: Optional[Callable] = None,
                                  context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute function with fallback mechanism.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to execute
            fallback_func: Optional fallback function
            context: Execution context
            
        Returns:
            Function result
            
        Raises:
            ServiceUnavailableError: If service is unavailable and no fallback
        """
        try:
            # Check service health and circuit breaker
            if not await self._is_service_available(service_name):
                self.logger.warning(f"Service {service_name} is not available, attempting fallback")
                return await self._execute_fallback(service_name, fallback_func, context)
            
            # Execute primary function
            result = await primary_func()
            
            # Record success
            await self._record_service_success(service_name)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Primary function failed for service {service_name}: {e}")
            
            # Record failure
            await self._record_service_failure(service_name, str(e))
            
            # Execute fallback
            return await self._execute_fallback(service_name, fallback_func, context)
    
    async def _execute_fallback(self, service_name: str, fallback_func: Optional[Callable],
                              context: Optional[Dict[str, Any]]) -> Any:
        """Execute fallback function or strategy."""
        try:
            # Check if fallback function is provided
            if fallback_func:
                self.logger.info(f"Executing custom fallback for service {service_name}")
                result = await fallback_func()
                self.performance_metrics['successful_fallbacks'] += 1
                return result
            
            # Use predefined degradation strategy
            strategy = self.degradation_strategies.get(service_name)
            if strategy and strategy.fallback_enabled:
                return await self._execute_degradation_strategy(service_name, context)
            
            # No fallback available
            self.logger.error(f"No fallback available for service {service_name}")
            raise ServiceUnavailableError(f"Service {service_name} is unavailable and no fallback exists")
            
        except Exception as e:
            self.logger.error(f"Fallback execution failed for service {service_name}: {e}")
            raise ServiceUnavailableError(f"Fallback failed for service {service_name}: {str(e)}")
    
    async def _execute_degradation_strategy(self, service_name: str,
                                          context: Optional[Dict[str, Any]]) -> Any:
        """Execute predefined degradation strategy."""
        try:
            strategy = self.degradation_strategies[service_name]
            degradation_level = await self._get_service_degradation_level(service_name)
            
            self.logger.info(
                f"Executing degradation strategy for {service_name} at level {degradation_level.value}"
            )
            
            # Execute based on degradation level
            if degradation_level == DegradationLevel.NORMAL:
                result = await self._execute_normal_strategy(service_name, context)
            elif degradation_level == DegradationLevel.DEGRADED:
                result = await self._execute_degraded_strategy(service_name, context)
            elif degradation_level == DegradationLevel.CRITICAL:
                result = await self._execute_critical_strategy(service_name, context)
            else:  # OFFLINE
                raise ServiceUnavailableError(f"Service {service_name} is offline")
            
            self.performance_metrics['successful_fallbacks'] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Degradation strategy execution failed: {e}")
            raise
    
    async def _execute_normal_strategy(self, service_name: str,
                                     context: Optional[Dict[str, Any]]) -> Any:
        """Execute normal service strategy."""
        # Simulate normal operation
        if service_name == 'bias_detection':
            return {
                'bias_score': 0.1,
                'confidence': 0.95,
                'strategy': 'normal',
                'status': 'success'
            }
        elif service_name == 'redis_connection':
            return {
                'connection_status': 'healthy',
                'operations_performed': 100,
                'strategy': 'normal',
                'status': 'success'
            }
        
        return {'status': 'success', 'strategy': 'normal', 'service': service_name}
    
    async def _execute_degraded_strategy(self, service_name: str,
                                       context: Optional[Dict[str, Any]]) -> Any:
        """Execute degraded service strategy."""
        # Simulate degraded operation
        if service_name == 'bias_detection':
            return {
                'bias_score': 0.3,
                'confidence': 0.8,
                'strategy': 'degraded',
                'status': 'degraded_success',
                'note': 'Simplified bias check performed'
            }
        elif service_name == 'redis_connection':
            return {
                'connection_status': 'degraded',
                'operations_performed': 50,
                'strategy': 'degraded',
                'status': 'degraded_success',
                'note': 'Memory caching only'
            }
        
        return {
            'status': 'degraded_success',
            'strategy': 'degraded',
            'service': service_name,
            'note': 'Service operating in degraded mode'
        }
    
    async def _execute_critical_strategy(self, service_name: str,
                                       context: Optional[Dict[str, Any]]) -> Any:
        """Execute critical service strategy."""
        # Simulate critical/minimal operation
        if service_name == 'bias_detection':
            return {
                'bias_score': 0.0,
                'confidence': 0.5,
                'strategy': 'critical',
                'status': 'critical_success',
                'note': 'Bias detection skipped due to service issues'
            }
        elif service_name == 'redis_connection':
            return {
                'connection_status': 'critical',
                'operations_performed': 0,
                'strategy': 'critical',
                'status': 'critical_success',
                'note': 'No caching available'
            }
        
        return {
            'status': 'critical_success',
            'strategy': 'critical',
            'service': service_name,
            'note': 'Service operating in critical mode with minimal functionality'
        }
    
    async def _is_service_available(self, service_name: str) -> bool:
        """Check if service is available based on health and circuit breaker."""
        try:
            # Get service health
            health = self.service_health.get(service_name)
            if not health:
                # Initialize health record
                health = ServiceHealth(
                    service_name=service_name,
                    status='unknown',
                    last_check=datetime.utcnow(),
                    failure_count=0,
                    success_count=0,
                    circuit_state=CircuitState.CLOSED,
                    next_retry_time=None
                )
                self.service_health[service_name] = health
            
            # Check circuit breaker
            if health.circuit_state == CircuitState.OPEN:
                if health.next_retry_time and datetime.utcnow() < health.next_retry_time:
                    return False
                else:
                    # Move to half-open state
                    health.circuit_state = CircuitState.HALF_OPEN
                    health.next_retry_time = None
            
            return health.circuit_state != CircuitState.OPEN
            
        except Exception as e:
            self.logger.error(f"Error checking service availability for {service_name}: {e}")
            return False
    
    async def _get_service_degradation_level(self, service_name: str) -> DegradationLevel:
        """Determine service degradation level based on health metrics."""
        try:
            health = self.service_health.get(service_name)
            if not health:
                return DegradationLevel.NORMAL
            
            # Calculate failure rate
            total_calls = health.failure_count + health.success_count
            if total_calls == 0:
                return DegradationLevel.NORMAL
            
            failure_rate = health.failure_count / total_calls
            
            # Determine degradation level
            if failure_rate > 0.5:  # >50% failure rate
                return DegradationLevel.CRITICAL
            elif failure_rate > 0.2:  # >20% failure rate
                return DegradationLevel.DEGRADED
            else:
                return DegradationLevel.NORMAL
                
        except Exception as e:
            self.logger.error(f"Error determining degradation level for {service_name}: {e}")
            return DegradationLevel.NORMAL
    
    async def _record_service_success(self, service_name: str) -> None:
        """Record successful service execution."""
        try:
            health = self.service_health.get(service_name)
            if health:
                health.success_count += 1
                health.last_check = datetime.utcnow()
                
                # Reset circuit breaker on success if in half-open state
                if health.circuit_state == CircuitState.HALF_OPEN:
                    health.circuit_state = CircuitState.CLOSED
                    health.failure_count = 0
                
                self.logger.debug(f"Recorded success for service {service_name}")
                
        except Exception as e:
            self.logger.error(f"Error recording service success for {service_name}: {e}")
    
    async def _record_service_failure(self, service_name: str, error_message: str) -> None:
        """Record failed service execution."""
        try:
            health = self.service_health.get(service_name)
            if health:
                health.failure_count += 1
                health.last_check = datetime.utcnow()
                
                # Check circuit breaker threshold
                if (health.failure_count >= self.config.circuit_breaker_threshold and
                    health.circuit_state == CircuitState.CLOSED):
                    
                    # Open circuit breaker
                    health.circuit_state = CircuitState.OPEN
                    health.next_retry_time = (
                        datetime.utcnow() + 
                        timedelta(seconds=self.config.circuit_breaker_timeout_seconds)
                    )
                    
                    self.performance_metrics['circuit_breaker_activations'] += 1
                    
                    self.logger.warning(
                        f"Circuit breaker opened for service {service_name} after "
                        f"{health.failure_count} failures"
                    )
                
                self.logger.debug(f"Recorded failure for service {service_name}: {error_message}")
                
        except Exception as e:
            self.logger.error(f"Error recording service failure for {service_name}: {e}")
    
    async def monitor_service_health(self) -> None:
        """Monitor service health and update status."""
        try:
            while True:
                current_time = datetime.utcnow()
                
                for service_name, health in self.service_health.items():
                    # Check if it's time for health check
                    if (current_time - health.last_check).total_seconds() >= self.config.service_health_check_interval:
                        
                        # Perform health check
                        is_healthy = await self._perform_health_check(service_name)
                        
                        # Update health status
                        if is_healthy:
                            health.status = 'healthy'
                            if health.circuit_state == CircuitState.HALF_OPEN:
                                # Check if we should close the circuit
                                success_rate = health.success_count / max(1, health.success_count + health.failure_count)
                                if success_rate >= self.config.half_open_success_threshold:
                                    health.circuit_state = CircuitState.CLOSED
                                    health.failure_count = 0
                                    self.logger.info(f"Circuit breaker closed for service {service_name}")
                        else:
                            health.status = 'unhealthy'
                        
                        health.last_check = current_time
                
                # Wait for next health check cycle
                await asyncio.sleep(self.config.service_health_check_interval)
                
        except Exception as e:
            self.logger.error(f"Service health monitoring error: {e}")
    
    async def _perform_health_check(self, service_name: str) -> bool:
        """Perform health check for a specific service."""
        try:
            # Simulate health check - in real implementation, this would check actual service health
            import random
            
            # Simulate occasional failures for demonstration
            failure_probability = 0.1  # 10% chance of failure
            
            is_healthy = random.random() > failure_probability
            
            self.logger.debug(f"Health check for {service_name}: {'healthy' if is_healthy else 'unhealthy'}")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed for service {service_name}: {e}")
            return False
    
    def get_service_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive service health summary.
        
        Returns:
            Service health summary
        """
        try:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'services': {},
                'overall_status': 'healthy',
                'degraded_services': [],
                'offline_services': [],
                'metrics': self.performance_metrics.copy()
            }
            
            for service_name, health in self.service_health.items():
                service_summary = {
                    'status': health.status,
                    'circuit_state': health.circuit_state.value,
                    'failure_count': health.failure_count,
                    'success_count': health.success_count,
                    'last_check': health.last_check.isoformat(),
                    'degradation_level': self._get_service_degradation_level(service_name).value
                }
                
                summary['services'][service_name] = service_summary
                
                # Categorize by status
                if health.status == 'unhealthy':
                    if health.circuit_state == CircuitState.OPEN:
                        summary['offline_services'].append(service_name)
                    else:
                        summary['degraded_services'].append(service_name)
            
            # Determine overall status
            if summary['offline_services']:
                summary['overall_status'] = 'critical'
            elif summary['degraded_services']:
                summary['overall_status'] = 'degraded'
            else:
                summary['overall_status'] = 'healthy'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate service health summary: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def get_degradation_recommendations(self) -> List[str]:
        """
        Get recommendations for service degradation issues.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Check for circuit breaker activations
            if self.performance_metrics['circuit_breaker_activations'] > 0:
                recommendations.append(
                    f"{self.performance_metrics['circuit_breaker_activations']} circuit breaker activations detected. "
                    "Review service configuration and error handling."
                )
            
            # Check for high failure rate
            total_fallbacks = self.performance_metrics['total_fallbacks']
            successful_fallbacks = self.performance_metrics['successful_fallbacks']
            
            if total_fallbacks > 0:
                success_rate = successful_fallbacks / total_fallbacks
                if success_rate < 0.8:
                    recommendations.append(
                        f"Fallback success rate is {success_rate:.1%}. "
                        "Consider improving fallback mechanisms."
                    )
            
            # Check for services in critical state
            critical_services = []
            for service_name, health in self.service_health.items():
                degradation_level = self._get_service_degradation_level(service_name)
                if degradation_level == DegradationLevel.CRITICAL:
                    critical_services.append(service_name)
            
            if critical_services:
                recommendations.append(
                    f"Services in critical state: {', '.join(critical_services)}. "
                    "Immediate investigation required."
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append("All services are operating within normal parameters.")
            
            return recommendations[:3]  # Limit to top 3
            
        except Exception as e:
            self.logger.error(f"Failed to generate degradation recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of graceful degradation manager.
        
        Returns:
            Health check results
        """
        try:
            # Check configuration
            config_healthy = (
                self.config.circuit_breaker_threshold > 0 and
                self.config.circuit_breaker_timeout_seconds > 0 and
                self.config.half_open_max_calls > 0
            )
            
            # Check services
            services_healthy = len(self.service_health) > 0
            
            # Check strategies
            strategies_loaded = len(self.degradation_strategies) > 0
            
            status = 'healthy' if all([config_healthy, services_healthy, strategies_loaded]) else 'degraded'
            
            return {
                'status': status,
                'configuration_healthy': config_healthy,
                'services_configured': services_healthy,
                'strategies_loaded': strategies_loaded,
                'total_fallbacks': self.performance_metrics['total_fallbacks'],
                'successful_fallbacks': self.performance_metrics['successful_fallbacks'],
                'circuit_breaker_activations': self.performance_metrics['circuit_breaker_activations'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GracefulDegradationManager health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }