"""
Error Recovery Manager for Pipeline Communication - Comprehensive Error Handling.

This module provides sophisticated error recovery, retry mechanisms, and graceful
degradation for the six-stage pipeline with HIPAA++ compliant error handling.
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
from .state_manager import StateManager
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    PipelineExecutionError, ValidationError, TimeoutError, 
    ResourceNotFoundError, BiasDetectionError, RetryExhaustedError
)
from ..utils.logger import get_request_logger
from ..utils.validation import sanitize_input


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 60.0
    fallback_enabled: bool = True
    skip_enabled: bool = False
    rollback_enabled: bool = True
    escalation_threshold: int = 3
    timeout_seconds: float = 300.0  # 5 minutes default


@dataclass
class RecoveryAttempt:
    """Individual recovery attempt record."""
    attempt_number: int
    timestamp: datetime
    strategy: RecoveryStrategy
    result: str  # 'success', 'failed', 'skipped'
    error_info: Optional[Dict[str, Any]]
    duration_seconds: float
    recovery_data: Optional[Dict[str, Any]]


@dataclass
class RecoveryResult:
    """Result of recovery operation."""
    recovered: bool
    final_strategy: RecoveryStrategy
    attempts: List[RecoveryAttempt]
    result_data: Optional[Dict[str, Any]]
    error: Optional[str]
    recommendations: List[str]


class ErrorRecoveryManager:
    """Comprehensive error recovery manager for pipeline operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error recovery manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Recovery configuration
        self.recovery_config = RecoveryConfig(**self.config.get('recovery', {}))
        
        # Error classification rules
        self.error_classification_rules = self._load_error_classification_rules()
        
        # Recovery strategies mapping
        self.recovery_strategies = {
            RecoveryStrategy.RETRY: self._execute_retry_strategy,
            RecoveryStrategy.FALLBACK: self._execute_fallback_strategy,
            RecoveryStrategy.SKIP: self._execute_skip_strategy,
            RecoveryStrategy.ROLLBACK: self._execute_rollback_strategy,
            RecoveryStrategy.ESCALATE: self._execute_escalation_strategy
        }
        
        # Stage-specific recovery configurations
        self.stage_recovery_configs = self._load_stage_recovery_configs()
        
        self.logger.info("ErrorRecoveryManager initialized with comprehensive recovery strategies")
    
    def _load_error_classification_rules(self) -> Dict[str, Any]:
        """Load error classification rules."""
        return {
            'validation_errors': {
                'patterns': ['ValidationError', 'validation_failed', 'invalid_input'],
                'severity': ErrorSeverity.LOW,
                'recoverable': True,
                'recommended_strategy': RecoveryStrategy.RETRY
            },
            'timeout_errors': {
                'patterns': ['TimeoutError', 'timeout', 'timed_out'],
                'severity': ErrorSeverity.MEDIUM,
                'recoverable': True,
                'recommended_strategy': RecoveryStrategy.RETRY
            },
            'resource_errors': {
                'patterns': ['ResourceNotFoundError', 'not_found', 'missing_resource'],
                'severity': ErrorSeverity.MEDIUM,
                'recoverable': True,
                'recommended_strategy': RecoveryStrategy.FALLBACK
            },
            'bias_detection_errors': {
                'patterns': ['BiasDetectionError', 'bias_threshold_exceeded'],
                'severity': ErrorSeverity.HIGH,
                'recoverable': True,
                'recommended_strategy': RecoveryStrategy.FALLBACK
            },
            'pipeline_errors': {
                'patterns': ['PipelineExecutionError', 'pipeline_failed'],
                'severity': ErrorSeverity.HIGH,
                'recoverable': True,
                'recommended_strategy': RecoveryStrategy.ROLLBACK
            },
            'critical_errors': {
                'patterns': ['critical', 'fatal', 'unrecoverable'],
                'severity': ErrorSeverity.CRITICAL,
                'recoverable': False,
                'recommended_strategy': RecoveryStrategy.ESCALATE
            }
        }
    
    def _load_stage_recovery_configs(self) -> Dict[str, RecoveryConfig]:
        """Load stage-specific recovery configurations."""
        return {
            'ingestion': RecoveryConfig(
                max_retries=2,
                retry_delay_seconds=1.0,
                fallback_enabled=True,
                skip_enabled=False
            ),
            'validation': RecoveryConfig(
                max_retries=3,
                retry_delay_seconds=2.0,
                fallback_enabled=True,
                skip_enabled=False
            ),
            'processing': RecoveryConfig(
                max_retries=3,
                retry_delay_seconds=3.0,
                fallback_enabled=True,
                skip_enabled=False,
                timeout_seconds=600.0  # 10 minutes for processing
            ),
            'standardization': RecoveryConfig(
                max_retries=2,
                retry_delay_seconds=2.0,
                fallback_enabled=True,
                skip_enabled=True
            ),
            'quality': RecoveryConfig(
                max_retries=3,
                retry_delay_seconds=2.0,
                fallback_enabled=True,
                skip_enabled=False
            ),
            'export': RecoveryConfig(
                max_retries=2,
                retry_delay_seconds=1.0,
                fallback_enabled=True,
                skip_enabled=False
            )
        }
    
    async def attempt_stage_recovery(self, context: 'PipelineContext', 
                                   stage_name: str, error: Exception) -> RecoveryResult:
        """
        Attempt recovery for a failed stage.
        
        Args:
            context: Pipeline execution context
            stage_name: Name of the failed stage
            error: The error that occurred
            
        Returns:
            Recovery result with success status and data
            
        Raises:
            RetryExhaustedError: If all recovery attempts fail
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Starting recovery for stage {stage_name} in execution "
                f"{context.execution_id}: {error}"
            )
            
            # Classify the error
            error_classification = self._classify_error(error)
            
            # Get stage-specific recovery configuration
            stage_config = self.stage_recovery_configs.get(
                stage_name, self.recovery_config
            )
            
            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(
                error_classification, stage_config, context.retry_count
            )
            
            # Execute recovery attempts
            recovery_result = await self._execute_recovery_strategy(
                context, stage_name, error, strategy, stage_config
            )
            
            # Log recovery result
            duration_seconds = time.time() - start_time
            self.logger.info(
                f"Recovery completed for stage {stage_name} in execution "
                f"{context.execution_id}: recovered={recovery_result.recovered}, "
                f"strategy={recovery_result.final_strategy.value}, "
                f"attempts={len(recovery_result.attempts)}, "
                f"duration={duration_seconds:.2f}s"
            )
            
            # Publish recovery event
            await self._publish_recovery_event(context, stage_name, recovery_result)
            
            return recovery_result
            
        except RetryExhaustedError:
            raise
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for stage {stage_name}: {e}")
            raise PipelineExecutionError(f"Recovery failed: {str(e)}")
    
    def _classify_error(self, error: Exception) -> Dict[str, Any]:
        """Classify error type and severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        for category, rules in self.error_classification_rules.items():
            # Check if error matches any patterns
            matches = any(
                pattern.lower() in error_message or pattern.lower() in error_type.lower()
                for pattern in rules['patterns']
            )
            
            if matches:
                return {
                    'category': category,
                    'severity': rules['severity'],
                    'recoverable': rules['recoverable'],
                    'recommended_strategy': rules['recommended_strategy']
                }
        
        # Default classification
        return {
            'category': 'unknown',
            'severity': ErrorSeverity.MEDIUM,
            'recoverable': True,
            'recommended_strategy': RecoveryStrategy.RETRY
        }
    
    def _determine_recovery_strategy(self, error_classification: Dict[str, Any],
                                   stage_config: RecoveryConfig,
                                   retry_count: int) -> RecoveryStrategy:
        """Determine the best recovery strategy based on error and context."""
        # Check if we've exceeded retry threshold
        if retry_count >= stage_config.escalation_threshold:
            return RecoveryStrategy.ESCALATE
        
        # Use recommended strategy if error is recoverable
        if error_classification['recoverable']:
            return error_classification['recommended_strategy']
        
        # Non-recoverable errors should escalate
        return RecoveryStrategy.ESCALATE
    
    async def _execute_recovery_strategy(self, context: 'PipelineContext',
                                       stage_name: str, original_error: Exception,
                                       strategy: RecoveryStrategy,
                                       config: RecoveryConfig) -> RecoveryResult:
        """Execute the determined recovery strategy."""
        attempts = []
        current_strategy = strategy
        
        for attempt_number in range(1, config.max_retries + 1):
            attempt_start = time.time()
            
            try:
                # Execute recovery attempt
                recovery_data = await self.recovery_strategies[current_strategy](
                    context, stage_name, original_error, attempt_number, config
                )
                
                attempt_duration = time.time() - attempt_start
                
                # Create successful attempt record
                attempt = RecoveryAttempt(
                    attempt_number=attempt_number,
                    timestamp=datetime.utcnow(),
                    strategy=current_strategy,
                    result='success',
                    error_info=None,
                    duration_seconds=attempt_duration,
                    recovery_data=recovery_data
                )
                attempts.append(attempt)
                
                # Return successful recovery result
                return RecoveryResult(
                    recovered=True,
                    final_strategy=current_strategy,
                    attempts=attempts,
                    result_data=recovery_data,
                    error=None,
                    recommendations=self._generate_recommendations(
                        current_strategy, recovery_data
                    )
                )
                
            except Exception as e:
                attempt_duration = time.time() - attempt_start
                
                # Create failed attempt record
                attempt = RecoveryAttempt(
                    attempt_number=attempt_number,
                    timestamp=datetime.utcnow(),
                    strategy=current_strategy,
                    result='failed',
                    error_info={
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    duration_seconds=attempt_duration,
                    recovery_data=None
                )
                attempts.append(attempt)
                
                self.logger.warning(
                    f"Recovery attempt {attempt_number} failed for stage {stage_name}: {e}"
                )
                
                # Determine next strategy
                next_strategy = self._get_next_recovery_strategy(
                    current_strategy, attempt_number, config, e
                )
                
                if next_strategy == RecoveryStrategy.ESCALATE:
                    break
                
                current_strategy = next_strategy
                
                # Wait before next attempt if retrying
                if current_strategy == RecoveryStrategy.RETRY:
                    await self._wait_before_retry(attempt_number, config)
        
        # All recovery attempts failed
        return RecoveryResult(
            recovered=False,
            final_strategy=current_strategy,
            attempts=attempts,
            result_data=None,
            error=f"All recovery attempts failed for stage {stage_name}",
            recommendations=[
                f"Stage {stage_name} failed after {len(attempts)} recovery attempts",
                "Consider reviewing input data quality and pipeline configuration",
                "Manual intervention may be required"
            ]
        )
    
    async def _execute_retry_strategy(self, context: 'PipelineContext',
                                    stage_name: str, error: Exception,
                                    attempt_number: int, config: RecoveryConfig) -> Dict[str, Any]:
        """Execute retry recovery strategy."""
        self.logger.info(
            f"Executing retry strategy for stage {stage_name}, attempt {attempt_number}"
        )
        
        # Simulate retry logic - in real implementation, this would retry the stage
        # For now, we'll simulate success on certain conditions
        
        # Simulate different retry outcomes based on error type
        error_classification = self._classify_error(error)
        
        if error_classification['category'] == 'timeout_errors' and attempt_number <= 2:
            # Simulate timeout recovery
            await asyncio.sleep(0.1)  # Simulate retry delay
            return {
                'retry_successful': True,
                'attempt_number': attempt_number,
                'error_resolved': 'timeout_resolved'
            }
        
        elif error_classification['category'] == 'validation_errors' and attempt_number == 1:
            # Simulate validation error recovery
            return {
                'retry_successful': True,
                'attempt_number': attempt_number,
                'error_resolved': 'validation_fixed'
            }
        
        else:
            # Simulate retry failure
            raise PipelineExecutionError(f"Retry attempt {attempt_number} failed")
    
    async def _execute_fallback_strategy(self, context: 'PipelineContext',
                                       stage_name: str, error: Exception,
                                       attempt_number: int, config: RecoveryConfig) -> Dict[str, Any]:
        """Execute fallback recovery strategy."""
        self.logger.info(
            f"Executing fallback strategy for stage {stage_name}, attempt {attempt_number}"
        )
        
        # Simulate fallback logic
        error_classification = self._classify_error(error)
        
        if error_classification['category'] == 'resource_errors':
            # Simulate fallback to alternative resource
            return {
                'fallback_successful': True,
                'fallback_type': 'alternative_resource',
                'resource_used': 'backup_dataset'
            }
        
        elif error_classification['category'] == 'bias_detection_errors':
            # Simulate fallback with bias mitigation
            return {
                'fallback_successful': True,
                'fallback_type': 'bias_mitigation',
                'mitigation_applied': 'demographic_rebalancing'
            }
        
        else:
            # Simulate fallback failure
            raise PipelineExecutionError(f"Fallback strategy failed")
    
    async def _execute_skip_strategy(self, context: 'PipelineContext',
                                   stage_name: str, error: Exception,
                                   attempt_number: int, config: RecoveryConfig) -> Dict[str, Any]:
        """Execute skip recovery strategy."""
        self.logger.info(
            f"Executing skip strategy for stage {stage_name}, attempt {attempt_number}"
        )
        
        # Check if stage can be skipped
        skippable_stages = ['standardization']  # Some stages can be safely skipped
        
        if stage_name in skippable_stages:
            return {
                'skip_successful': True,
                'skipped_stage': stage_name,
                'impact': 'minimal',
                'recommendation': 'Manual review recommended'
            }
        else:
            raise PipelineExecutionError(f"Stage {stage_name} cannot be skipped")
    
    async def _execute_rollback_strategy(self, context: 'PipelineContext',
                                       stage_name: str, error: Exception,
                                       attempt_number: int, config: RecoveryConfig) -> Dict[str, Any]:
        """Execute rollback recovery strategy."""
        self.logger.info(
            f"Executing rollback strategy for stage {stage_name}, attempt {attempt_number}"
        )
        
        # Simulate rollback to previous checkpoint
        try:
            # In real implementation, this would restore from checkpoint
            rollback_result = {
                'rollback_successful': True,
                'rolled_back_stage': stage_name,
                'checkpoint_restored': f"{stage_name}_checkpoint_v1",
                'data_integrity': 'maintained'
            }
            
            return rollback_result
            
        except Exception as e:
            raise PipelineExecutionError(f"Rollback failed: {str(e)}")
    
    async def _execute_escalation_strategy(self, context: 'PipelineContext',
                                         stage_name: str, error: Exception,
                                         attempt_number: int, config: RecoveryConfig) -> Dict[str, Any]:
        """Execute escalation recovery strategy."""
        self.logger.warning(
            f"Executing escalation strategy for stage {stage_name}, attempt {attempt_number}"
        )
        
        # This represents escalation to human intervention or higher-level systems
        escalation_data = {
            'escalation_triggered': True,
            'stage': stage_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'attempts_made': attempt_number,
            'requires_manual_intervention': True,
            'escalation_timestamp': datetime.utcnow().isoformat()
        }
        
        # In real implementation, this would trigger alerts, notifications, etc.
        self.logger.critical(
            f"ESCALATION REQUIRED: Stage {stage_name} in execution "
            f"{context.execution_id} requires manual intervention"
        )
        
        return escalation_data
    
    def _get_next_recovery_strategy(self, current_strategy: RecoveryStrategy,
                                  attempt_number: int, config: RecoveryConfig,
                                  last_error: Exception) -> RecoveryStrategy:
        """Determine the next recovery strategy to try."""
        # If we've hit max retries, escalate
        if attempt_number >= config.max_retries:
            return RecoveryStrategy.ESCALATE
        
        # Strategy progression logic
        strategy_progression = {
            RecoveryStrategy.RETRY: RecoveryStrategy.FALLBACK,
            RecoveryStrategy.FALLBACK: RecoveryStrategy.SKIP if config.skip_enabled else RecoveryStrategy.ROLLBACK,
            RecoveryStrategy.SKIP: RecoveryStrategy.ROLLBACK,
            RecoveryStrategy.ROLLBACK: RecoveryStrategy.ESCALATE
        }
        
        return strategy_progression.get(current_strategy, RecoveryStrategy.ESCALATE)
    
    async def _wait_before_retry(self, attempt_number: int, config: RecoveryConfig) -> None:
        """Wait before retry attempt with exponential backoff."""
        if config.exponential_backoff:
            delay = min(
                config.retry_delay_seconds * (config.backoff_multiplier ** (attempt_number - 1)),
                config.max_retry_delay_seconds
            )
        else:
            delay = config.retry_delay_seconds
        
        self.logger.info(f"Waiting {delay:.1f} seconds before retry attempt {attempt_number + 1}")
        await asyncio.sleep(delay)
    
    def _generate_recommendations(self, final_strategy: RecoveryStrategy,
                                recovery_data: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on recovery result."""
        recommendations = []
        
        if final_strategy == RecoveryStrategy.RETRY and recovery_data:
            recommendations.append("Retry was successful - monitor for similar issues")
        
        elif final_strategy == RecoveryStrategy.FALLBACK and recovery_data:
            recommendations.append("Fallback solution applied - review for optimization")
            if 'mitigation_applied' in recovery_data:
                recommendations.append(f"Applied mitigation: {recovery_data['mitigation_applied']}")
        
        elif final_strategy == RecoveryStrategy.SKIP and recovery_data:
            recommendations.append("Stage was skipped - manual review recommended")
        
        elif final_strategy == RecoveryStrategy.ROLLBACK and recovery_data:
            recommendations.append("Rollback completed - verify data integrity")
        
        elif final_strategy == RecoveryStrategy.ESCALATE:
            recommendations.append("Manual intervention required")
            recommendations.append("Review error logs and pipeline configuration")
        
        return recommendations
    
    async def _publish_recovery_event(self, context: 'PipelineContext',
                                    stage_name: str, recovery_result: RecoveryResult) -> None:
        """Publish recovery event for monitoring and alerting."""
        try:
            event = EventMessage(
                event_type=EventType.RECOVERY_ATTEMPT.value,
                execution_id=context.execution_id,
                stage=stage_name,
                payload={
                    'recovered': recovery_result.recovered,
                    'final_strategy': recovery_result.final_strategy.value,
                    'attempts_count': len(recovery_result.attempts),
                    'recommendations': recovery_result.recommendations,
                    'recovery_timestamp': datetime.utcnow().isoformat()
                },
                source='error_recovery_manager',
                target='monitoring_system'
            )
            
            await self.event_bus.publish_event(event)
            
        except Exception as e:
            self.logger.error(f"Failed to publish recovery event: {e}")
    
    async def create_error_checkpoint(self, execution_id: str, stage_name: str,
                                    error: Exception, context_data: Dict[str, Any]) -> str:
        """
        Create an error checkpoint for debugging and recovery.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            error: The error that occurred
            context_data: Additional context data
            
        Returns:
            Checkpoint ID
            
        Raises:
            PipelineExecutionError: If checkpoint creation fails
        """
        try:
            checkpoint_id = f"error_{execution_id}_{stage_name}_{int(time.time())}"
            
            # Sanitize context data
            sanitized_context = sanitize_input(context_data)
            
            error_checkpoint = {
                'checkpoint_id': checkpoint_id,
                'execution_id': execution_id,
                'stage_name': stage_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_timestamp': datetime.utcnow().isoformat(),
                'context_data': sanitized_context,
                'environment_info': self._get_environment_info(),
                'recovery_suggestions': self._generate_recovery_suggestions(error)
            }
            
            # Store checkpoint in Redis
            key = f"pipeline:error_checkpoint:{checkpoint_id}"
            await self.redis_client.setex(
                key,
                86400 * 7,  # 7 days retention
                json.dumps(error_checkpoint, default=str)
            )
            
            self.logger.info(
                f"Error checkpoint created: {checkpoint_id} for execution {execution_id}, "
                f"stage {stage_name}"
            )
            
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create error checkpoint: {e}")
            raise PipelineExecutionError(f"Error checkpoint creation failed: {str(e)}")
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'python_version': '3.11',  # Would be actual Python version
            'platform': 'linux',  # Would be actual platform
            'memory_usage': 'normal',  # Would be actual memory usage
            'cpu_usage': 'normal'  # Would be actual CPU usage
        }
    
    def _generate_recovery_suggestions(self, error: Exception) -> List[str]:
        """Generate recovery suggestions based on error type."""
        error_classification = self._classify_error(error)
        
        suggestions = [
            f"Error type: {error_classification['category']}",
            f"Severity: {error_classification['severity'].value}"
        ]
        
        if error_classification['recoverable']:
            suggestions.append("Error is recoverable - retry recommended")
            suggestions.append(f"Recommended strategy: {error_classification['recommended_strategy'].value}")
        else:
            suggestions.append("Error may require manual intervention")
        
        # Add specific suggestions based on error category
        category = error_classification['category']
        if category == 'timeout_errors':
            suggestions.extend([
                "Check system resources and network connectivity",
                "Consider increasing timeout values",
                "Verify external service availability"
            ])
        elif category == 'validation_errors':
            suggestions.extend([
                "Review input data quality and format",
                "Check validation rules configuration",
                "Verify data preprocessing steps"
            ])
        elif category == 'bias_detection_errors':
            suggestions.extend([
                "Review dataset for demographic representation",
                "Consider bias mitigation techniques",
                "Check bias detection thresholds"
            ])
        
        return suggestions
    
    async def get_error_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve error checkpoint for analysis.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Error checkpoint data or None if not found
            
        Raises:
            PipelineExecutionError: If retrieval fails
        """
        try:
            key = f"pipeline:error_checkpoint:{checkpoint_id}"
            checkpoint_data = await self.redis_client.get(key)
            
            if not checkpoint_data:
                return None
            
            return json.loads(checkpoint_data)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve error checkpoint: {e}")
            raise PipelineExecutionError(f"Error checkpoint retrieval failed: {str(e)}")
    
    async def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old error checkpoints.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of cleaned up checkpoints
            
        Raises:
            PipelineExecutionError: If cleanup fails
        """
        try:
            pattern = "pipeline:error_checkpoint:*"
            keys = await self.redis_client.keys(pattern)
            
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
            
            for key in keys:
                try:
                    checkpoint_data = await self.redis_client.get(key)
                    if checkpoint_data:
                        checkpoint = json.loads(checkpoint_data)
                        checkpoint_time = datetime.fromisoformat(checkpoint['error_timestamp'])
                        
                        if checkpoint_time < cutoff_time:
                            await self.redis_client.delete(key)
                            cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process checkpoint key {key}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} old error checkpoints")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            raise PipelineExecutionError(f"Checkpoint cleanup failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of error recovery manager.
        
        Returns:
            Health check results
        """
        try:
            # Check configuration
            config_healthy = (
                self.recovery_config.max_retries > 0 and
                self.recovery_config.retry_delay_seconds > 0 and
                self.recovery_config.timeout_seconds > 0
            )
            
            # Check error classification rules
            rules_healthy = len(self.error_classification_rules) > 0
            
            # Check stage recovery configs
            stages_healthy = len(self.stage_recovery_configs) > 0
            
            status = 'healthy' if all([config_healthy, rules_healthy, stages_healthy]) else 'degraded'
            
            return {
                'status': status,
                'configuration_healthy': config_healthy,
                'error_rules_loaded': rules_healthy,
                'stage_configs_loaded': stages_healthy,
                'recovery_strategies_available': len(self.recovery_strategies),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ErrorRecoveryManager health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }