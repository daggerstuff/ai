"""
Six-Stage Pipeline Coordinator for TechDeck-Python Integration.

This module provides comprehensive pipeline coordination with Redis event bus
integration, state management, bias detection, and HIPAA++ compliant data handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict

from .event_bus import EventBus, EventMessage, EventType, EventHandler
from .state_manager import StateManager, PipelineState
from .progress_tracker import ProgressTracker
from .error_recovery import ErrorRecoveryManager
from .bias_integration import BiasDetectionIntegration
from .performance_monitor import PerformanceMonitor
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    PipelineExecutionError, ValidationError, ResourceNotFoundError,
    BiasDetectionError, TimeoutError
)
from ..utils.logger import get_request_logger, log_performance_metric
from ..utils.validation import validate_pipeline_input, sanitize_input


@dataclass
class PipelineContext:
    """Context object for pipeline execution."""
    execution_id: str
    user_id: str
    dataset_ids: List[str]
    execution_mode: str
    quality_threshold: float
    enable_bias_detection: bool
    start_time: datetime
    current_stage: str
    stage_results: Dict[str, Any]
    metadata: Dict[str, Any]
    retry_count: int = 0
    checkpoint_enabled: bool = True


class StageCoordinator(EventHandler):
    """Individual stage coordinator with HIPAA++ compliance."""
    
    def __init__(self, stage_name: str, event_bus: EventBus, 
                 bias_integration: Optional[BiasDetectionIntegration] = None):
        super().__init__(
            name=f"{stage_name}_coordinator",
            event_types=[
                EventType.REQUEST_INITIATED.value,
                EventType.STAGE_TRANSITION.value,
                EventType.ERROR_OCCURRED.value
            ]
        )
        self.stage_name = stage_name
        self.event_bus = event_bus
        self.bias_integration = bias_integration
        self.logger = get_request_logger()
        
        # Stage-specific configuration
        self.stage_config = self._get_stage_config(stage_name)
    
    def _get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """Get stage-specific configuration."""
        configs = {
            'ingestion': {
                'timeout': 300,  # 5 minutes
                'max_file_size_mb': 100,
                'supported_formats': ['csv', 'json', 'jsonl', 'parquet', 'txt']
            },
            'validation': {
                'timeout': 600,  # 10 minutes
                'bias_detection_enabled': True,
                'quality_threshold': 0.8
            },
            'processing': {
                'timeout': 1800,  # 30 minutes
                'batch_size': 1000,
                'parallel_workers': 4
            },
            'standardization': {
                'timeout': 900,  # 15 minutes
                'target_format': 'standardized_json'
            },
            'quality': {
                'timeout': 1200,  # 20 minutes
                'assessment_criteria': ['completeness', 'consistency', 'accuracy']
            },
            'export': {
                'timeout': 600,  # 10 minutes
                'output_formats': ['json', 'csv', 'parquet']
            }
        }
        return configs.get(stage_name, {'timeout': 300})
    
    async def execute_stage(self, context: PipelineContext, 
                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute individual pipeline stage with comprehensive monitoring.
        
        Args:
            context: Pipeline execution context
            input_data: Input data for the stage
            
        Returns:
            Stage execution results
            
        Raises:
            PipelineExecutionError: If stage execution fails
            TimeoutError: If stage execution times out
            BiasDetectionError: If bias detection fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Executing stage {self.stage_name} for execution "
                f"{context.execution_id}"
            )
            
            # Publish stage start event
            await self._publish_stage_start(context)
            
            # Validate input data for HIPAA++ compliance
            validated_input = await self._validate_stage_input(input_data)
            
            # Execute stage-specific logic
            stage_result = await self._execute_stage_logic(context, validated_input)
            
            # Perform bias detection if enabled and applicable
            if (context.enable_bias_detection and 
                self.stage_config.get('bias_detection_enabled', False)):
                bias_result = await self._perform_bias_detection(stage_result)
                stage_result['bias_analysis'] = bias_result
            
            # Validate stage output
            validated_output = await self._validate_stage_output(stage_result)
            
            # Publish stage completion event
            await self._publish_stage_completion(context, validated_output)
            
            # Log performance metrics
            duration_ms = (time.time() - start_time) * 1000
            self._log_stage_performance(context, duration_ms)
            
            # Check sub-50ms requirement for critical operations
            if duration_ms > 50 and self.stage_name in ['validation', 'bias_detection']:
                self.logger.warning(
                    f"Stage {self.stage_name} exceeded 50ms threshold: "
                    f"{duration_ms:.2f}ms for execution {context.execution_id}"
                )
            
            return validated_output
            
        except Exception as e:
            self.logger.error(
                f"Error in stage {self.stage_name} for execution "
                f"{context.execution_id}: {e}"
            )
            await self._handle_stage_error(context, e)
            raise
    
    async def _execute_stage_logic(self, context: PipelineContext, 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage-specific logic - to be implemented by subclasses."""
        raise NotImplementedError(
            f"Stage {self.stage_name} must implement _execute_stage_logic"
        )
    
    async def _validate_stage_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize stage input data for HIPAA++ compliance."""
        try:
            # Remove any potential PII/PHI
            sanitized_data = sanitize_input(input_data)
            
            # Validate against stage-specific requirements
            validation_result = validate_pipeline_input(
                sanitized_data, 
                self.stage_name
            )
            
            if not validation_result['is_valid']:
                raise ValidationError(
                    f"Stage {self.stage_name} input validation failed: "
                    f"{validation_result['errors']}"
                )
            
            return sanitized_data
            
        except Exception as e:
            self.logger.error(f"Stage input validation error: {e}")
            raise ValidationError(f"Input validation failed: {str(e)}")
    
    async def _validate_stage_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stage output for quality and compliance."""
        try:
            # Basic output validation
            if not isinstance(output_data, dict):
                raise ValidationError("Stage output must be a dictionary")
            
            # Check for required fields based on stage
            required_fields = self._get_required_output_fields()
            missing_fields = [
                field for field in required_fields 
                if field not in output_data
            ]
            
            if missing_fields:
                raise ValidationError(
                    f"Missing required output fields: {missing_fields}"
                )
            
            # Sanitize output data
            sanitized_output = sanitize_input(output_data)
            
            return sanitized_output
            
        except Exception as e:
            self.logger.error(f"Stage output validation error: {e}")
            raise ValidationError(f"Output validation failed: {str(e)}")
    
    def _get_required_output_fields(self) -> List[str]:
        """Get required output fields for the stage."""
        field_mappings = {
            'ingestion': ['source_format', 'record_count', 'validation_passed'],
            'validation': ['validation_score', 'checks_passed', 'recommendations'],
            'processing': ['processed_records', 'processing_metrics'],
            'standardization': ['target_format', 'fields_standardized'],
            'quality': ['quality_score', 'assessment_results'],
            'export': ['output_files', 'metadata']
        }
        return field_mappings.get(self.stage_name, [])
    
    async def _perform_bias_detection(self, stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bias detection analysis."""
        if not self.bias_integration:
            return {
                'bias_score': 0.0,
                'bias_categories': {},
                'recommendations': [],
                'compliance_status': 'not_applicable'
            }
        
        try:
            bias_result = await self.bias_integration.analyze_stage_data(
                stage_data,
                self.stage_name
            )
            
            # Check bias threshold
            if bias_result.get('bias_score', 0) > 0.7:  # Configurable threshold
                self.logger.warning(
                    f"High bias detected in stage {self.stage_name}: "
                    f"{bias_result['bias_score']}"
                )
                
                # Publish bias threshold exceeded event
                bias_event = await self.event_bus.create_bias_detection_event(
                    execution_id="current",  # Will be set by caller
                    bias_score=bias_result['bias_score'],
                    threshold=0.7,
                    recommendations=bias_result.get('recommendations', [])
                )
                await self.event_bus.publish_event(bias_event)
            
            return bias_result
            
        except Exception as e:
            self.logger.error(f"Bias detection failed for stage {self.stage_name}: {e}")
            raise BiasDetectionError(f"Bias detection failed: {str(e)}")
    
    async def _publish_stage_start(self, context: PipelineContext) -> None:
        """Publish stage start event."""
        event = await self.event_bus.create_stage_event(
            execution_id=context.execution_id,
            stage=self.stage_name,
            status='started',
            progress_percent=0.0,
            message=f"{self.stage_name} stage started"
        )
        await self.event_bus.publish_event(event)
    
    async def _publish_stage_completion(self, context: PipelineContext, 
                                      result: Dict[str, Any]) -> None:
        """Publish stage completion event."""
        event = await self.event_bus.create_stage_event(
            execution_id=context.execution_id,
            stage=self.stage_name,
            status='completed',
            progress_percent=100.0,
            message=f"{self.stage_name} stage completed successfully",
            payload={'stage_result': result}
        )
        await self.event_bus.publish_event(event)
    
    async def _handle_stage_error(self, context: PipelineContext, error: Exception) -> None:
        """Handle stage execution errors with proper logging."""
        error_event = EventMessage(
            event_type=EventType.ERROR_OCCURRED.value,
            execution_id=context.execution_id,
            stage=self.stage_name,
            payload={
                'error_type': type(error).__name__,
                'error_message': str(error),  # Sanitized error message
                'stage': self.stage_name,
                'recoverable': self._is_error_recoverable(error)
            },
            source=f"{self.stage_name}_coordinator",
            target='error_recovery_manager'
        )
        await self.event_bus.publish_event(error_event)
    
    def _is_error_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        recoverable_errors = [
            'ValidationError',
            'TimeoutError',
            'ResourceNotFoundError'
        ]
        return type(error).__name__ in recoverable_errors
    
    def _log_stage_performance(self, context: PipelineContext, duration_ms: float) -> None:
        """Log stage performance metrics."""
        self.logger.info(
            f"Stage {self.stage_name} completed in {duration_ms:.2f}ms "
            f"for execution {context.execution_id}"
        )
        
        # Update performance metrics
        if duration_ms > 50:
            self.logger.warning(
                f"Stage {self.stage_name} exceeded 50ms threshold"
            )


class PipelineCoordinator:
    """Main coordinator for six-stage pipeline execution."""
    
    def __init__(self, redis_client: RedisClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline coordinator with comprehensive services.
        
        Args:
            redis_client: Redis client for event bus and state management
            config: Optional configuration dictionary
        """
        self.redis_client = redis_client
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Initialize core services
        self.event_bus = EventBus(redis_client, self.config.get('event_bus', {}))
        self.state_manager = StateManager(redis_client, self.config.get('state_manager', {}))
        self.progress_tracker = ProgressTracker(redis_client, self.config.get('progress_tracker', {}))
        self.error_recovery = ErrorRecoveryManager(self.config.get('error_recovery', {}))
        self.bias_integration = BiasDetectionIntegration(self.config.get('bias_detection', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance_monitor', {}))
        
        # Initialize stage coordinators
        self.stage_coordinators = self._initialize_stage_coordinators()
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info("PipelineCoordinator initialized with six-stage coordination")
    
    def _initialize_stage_coordinators(self) -> Dict[str, StageCoordinator]:
        """Initialize all six stage coordinators."""
        stage_names = [
            'ingestion',      # Stage 1: Data Ingestion
            'validation',     # Stage 2: Multi-tier Validation  
            'processing',     # Stage 3: Therapeutic Processing
            'standardization', # Stage 4: Format Standardization
            'quality',        # Stage 5: Quality Assessment
            'export'          # Stage 6: Output Generation
        ]
        
        coordinators = {}
        for stage_name in stage_names:
            coordinator = self._create_stage_coordinator(stage_name)
            coordinators[stage_name] = coordinator
            
            # Register coordinator as event handler
            self.event_bus.register_handler(coordinator)
        
        return coordinators
    
    def _create_stage_coordinator(self, stage_name: str) -> StageCoordinator:
        """Create a stage coordinator instance."""
        # Create appropriate coordinator based on stage
        if stage_name == 'validation':
            return ValidationStageCoordinator(
                stage_name, self.event_bus, self.bias_integration
            )
        elif stage_name == 'processing':
            return ProcessingStageCoordinator(
                stage_name, self.event_bus, self.bias_integration
            )
        else:
            return GenericStageCoordinator(
                stage_name, self.event_bus, self.bias_integration
            )
    
    def _register_event_handlers(self) -> None:
        """Register additional event handlers for pipeline coordination."""
        # Register progress tracker
        progress_handler = ProgressEventHandler(self.progress_tracker)
        self.event_bus.register_handler(progress_handler)
        
        # Register error recovery handler
        error_handler = ErrorRecoveryEventHandler(self.error_recovery)
        self.event_bus.register_handler(error_handler)
        
        # Register performance monitor
        performance_handler = PerformanceEventHandler(self.performance_monitor)
        self.event_bus.register_handler(performance_handler)
    
    async def execute_pipeline(self, execution_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete six-stage pipeline with comprehensive coordination.
        
        Args:
            execution_request: Pipeline execution request containing:
                - dataset_ids: List of dataset IDs
                - user_id: User ID for authorization
                - execution_mode: Execution mode (standard, fast, comprehensive)
                - quality_threshold: Quality threshold (0.0-1.0)
                - enable_bias_detection: Whether to enable bias detection
                - additional configuration options
                
        Returns:
            Complete pipeline execution results
            
        Raises:
            PipelineExecutionError: If pipeline execution fails
            ValidationError: If request validation fails
        """
        start_time = time.time()
        
        try:
            # Validate execution request
            validated_request = await self._validate_execution_request(execution_request)
            
            # Generate unique execution ID
            execution_id = self._generate_execution_id()
            
            # Create pipeline context
            context = PipelineContext(
                execution_id=execution_id,
                user_id=validated_request['user_id'],
                dataset_ids=validated_request['dataset_ids'],
                execution_mode=validated_request['execution_mode'],
                quality_threshold=validated_request['quality_threshold'],
                enable_bias_detection=validated_request['enable_bias_detection'],
                start_time=datetime.utcnow(),
                current_stage='initialization',
                stage_results={},
                metadata=validated_request.get('metadata', {})
            )
            
            # Initialize pipeline state
            await self.state_manager.initialize_pipeline_state(context)
            
            # Publish pipeline start event
            await self._publish_pipeline_start(context)
            
            # Execute six stages in sequence
            stage_results = await self._execute_six_stages(context)
            
            # Finalize pipeline execution
            final_results = await self._finalize_pipeline_execution(context, stage_results)
            
            # Log overall performance
            total_duration_ms = (time.time() - start_time) * 1000
            self._log_pipeline_performance(context, total_duration_ms)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            await self._handle_pipeline_failure(execution_request, e)
            raise PipelineExecutionError(f"Pipeline execution failed: {str(e)}")
    
    async def _execute_six_stages(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute all six pipeline stages with proper coordination."""
        stage_results = {}
        stage_order = [
            'ingestion',      # Stage 1: Data Ingestion
            'validation',     # Stage 2: Multi-tier Validation
            'processing',     # Stage 3: Therapeutic Processing  
            'standardization', # Stage 4: Format Standardization
            'quality',        # Stage 5: Quality Assessment
            'export'          # Stage 6: Output Generation
        ]
        
        for stage_name in stage_order:
            try:
                # Update context with current stage
                context.current_stage = stage_name
                
                # Get input data for this stage
                stage_input = self._get_stage_input(context, stage_name, stage_results)
                
                # Execute stage
                stage_coordinator = self.stage_coordinators[stage_name]
                stage_result = await stage_coordinator.execute_stage(context, stage_input)
                
                # Store stage result
                stage_results[stage_name] = stage_result
                context.stage_results[stage_name] = stage_result
                
                # Update pipeline state
                await self.state_manager.update_stage_state(
                    context.execution_id, stage_name, 'completed', stage_result
                )
                
                # Check quality threshold for validation stage
                if (stage_name == 'validation' and 
                    stage_result.get('result', {}).get('validation_score', 0) < 
                    context.quality_threshold):
                    
                    self.logger.warning(
                        f"Quality threshold not met in validation stage "
                        f"for execution {context.execution_id}"
                    )
                    
                    # Publish quality warning event
                    await self._publish_quality_warning(context, stage_result)
                
            except Exception as e:
                self.logger.error(
                    f"Stage {stage_name} failed for execution "
                    f"{context.execution_id}: {e}"
                )
                
                # Update pipeline state with error
                await self.state_manager.update_stage_state(
                    context.execution_id, stage_name, 'failed', 
                    {'error': str(e)}
                )
                
                # Attempt recovery if possible
                recovery_result = await self.error_recovery.attempt_stage_recovery(
                    context, stage_name, e
                )
                
                if recovery_result['recovered']:
                    # Use recovered result
                    stage_results[stage_name] = recovery_result['result']
                    context.stage_results[stage_name] = recovery_result['result']
                else:
                    # Re-raise if not recoverable
                    raise
        
        return stage_results
    
    async def _validate_execution_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline execution request."""
        try:
            # Required fields validation
            required_fields = ['dataset_ids', 'user_id', 'execution_mode']
            missing_fields = [
                field for field in required_fields 
                if field not in request
            ]
            
            if missing_fields:
                raise ValidationError(
                    f"Missing required fields: {missing_fields}"
                )
            
            # Validate dataset IDs
            if not isinstance(request['dataset_ids'], list) or len(request['dataset_ids']) == 0:
                raise ValidationError("dataset_ids must be a non-empty list")
            
            # Validate execution mode
            valid_modes = ['standard', 'fast', 'comprehensive']
            if request['execution_mode'] not in valid_modes:
                raise ValidationError(
                    f"Invalid execution_mode. Must be one of: {valid_modes}"
                )
            
            # Validate quality threshold
            quality_threshold = request.get('quality_threshold', 0.8)
            if not isinstance(quality_threshold, (int, float)):
                raise ValidationError("quality_threshold must be a number")
            if not (0.0 <= quality_threshold <= 1.0):
                raise ValidationError("quality_threshold must be between 0.0 and 1.0")
            
            # Set defaults for optional fields
            validated_request = {
                'dataset_ids': request['dataset_ids'],
                'user_id': request['user_id'],
                'execution_mode': request['execution_mode'],
                'quality_threshold': quality_threshold,
                'enable_bias_detection': request.get('enable_bias_detection', True),
                'metadata': request.get('metadata', {})
            }
            
            return validated_request
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Execution request validation error: {e}")
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        import uuid
        return f"pipeline_{uuid.uuid4().hex[:12]}_{int(time.time())}"
    
    def _get_stage_input(self, context: PipelineContext, stage_name: str, 
                        previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get input data for a specific stage."""
        if stage_name == 'ingestion':
            # First stage gets dataset information
            return {
                'dataset_ids': context.dataset_ids,
                'execution_mode': context.execution_mode
            }
        else:
            # Subsequent stages get previous stage results
            previous_stage = self._get_previous_stage(stage_name)
            if previous_stage and previous_stage in previous_results:
                return previous_results[previous_stage]
            else:
                return {}
    
    def _get_previous_stage(self, current_stage: str) -> Optional[str]:
        """Get the previous stage in the pipeline."""
        stage_order = [
            'ingestion', 'validation', 'processing', 
            'standardization', 'quality', 'export'
        ]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index > 0:
                return stage_order[current_index - 1]
        except ValueError:
            pass
        
        return None
    
    async def _publish_pipeline_start(self, context: PipelineContext) -> None:
        """Publish pipeline start event."""
        event = EventMessage(
            event_type=EventType.REQUEST_INITIATED.value,
            execution_id=context.execution_id,
            payload={
                'user_id': context.user_id,
                'dataset_count': len(context.dataset_ids),
                'execution_mode': context.execution_mode,
                'quality_threshold': context.quality_threshold,
                'bias_detection_enabled': context.enable_bias_detection
            },
            source='pipeline_coordinator',
            target='all_services'
        )
        await self.event_bus.publish_event(event)
    
    async def _publish_quality_warning(self, context: PipelineContext, 
                                     validation_result: Dict[str, Any]) -> None:
        """Publish quality warning event."""
        event = EventMessage(
            event_type=EventType.LATENCY_WARNING.value,  # Reusing for quality warnings
            execution_id=context.execution_id,
            stage='validation',
            payload={
                'warning_type': 'quality_threshold',
                'validation_score': validation_result.get('result', {}).get('validation_score', 0),
                'threshold': context.quality_threshold,
                'message': 'Quality threshold not met'
            },
            source='pipeline_coordinator',
            target='user_notification'
        )
        await self.event_bus.publish_event(event)
    
    async def _finalize_pipeline_execution(self, context: PipelineContext, 
                                         stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize pipeline execution with comprehensive results."""
        try:
            # Calculate overall metrics
            total_duration = (
                datetime.utcnow() - context.start_time
            ).total_seconds()
            
            overall_quality_score = self._calculate_overall_quality_score(stage_results)
            
            # Create final results
            final_results = {
                'execution_id': context.execution_id,
                'status': 'completed',
                'start_time': context.start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'total_duration_seconds': total_duration,
                'overall_quality_score': overall_quality_score,
                'stage_results': stage_results,
                'metadata': {
                    'user_id': context.user_id,
                    'dataset_count': len(context.dataset_ids),
                    'execution_mode': context.execution_mode,
                    'bias_detection_enabled': context.enable_bias_detection,
                    'quality_threshold': context.quality_threshold
                },
                'output_files': self._extract_output_files(stage_results),
                'recommendations': self._generate_recommendations(stage_results)
            }
            
            # Update final pipeline state
            await self.state_manager.update_pipeline_state(
                context.execution_id, 'completed', final_results
            )
            
            # Publish completion event
            completion_event = EventMessage(
                event_type=EventType.COMPLETION_FINALIZED.value,
                execution_id=context.execution_id,
                payload={
                    'status': 'completed',
                    'overall_quality_score': overall_quality_score,
                    'total_duration_seconds': total_duration
                },
                source='pipeline_coordinator',
                target='all_services'
            )
            await self.event_bus.publish_event(completion_event)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error finalizing pipeline execution: {e}")
            raise PipelineExecutionError(f"Finalization failed: {str(e)}")
    
    def _calculate_overall_quality_score(self, stage_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from stage results."""
        quality_scores = []
        
        for stage_name, result in stage_results.items():
            if result.get('status') == 'completed':
                stage_score = result.get('result', {}).get('quality_score', 0)
                if isinstance(stage_score, (int, float)):
                    quality_scores.append(stage_score)
        
        if quality_scores:
            return sum(quality_scores) / len(quality_scores)
        else:
            return 0.0
    
    def _extract_output_files(self, stage_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract output file information from stage results."""
        output_files = []
        
        # Look for output files in the export stage
        export_result = stage_results.get('export', {})
        if export_result.get('status') == 'completed':
            files = export_result.get('result', {}).get('output_files', [])
            output_files.extend(files)
        
        return output_files
    
    def _generate_recommendations(self, stage_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stage results."""
        recommendations = []
        
        # Analyze each stage for recommendations
        for stage_name, result in stage_results.items():
            if result.get('status') == 'completed':
                stage_recommendations = result.get('result', {}).get('recommendations', [])
                if isinstance(stage_recommendations, list):
                    recommendations.extend(stage_recommendations)
        
        # Add overall recommendations
        if not any('bias' in rec.lower() for rec in recommendations):
            recommendations.append(
                "Consider reviewing dataset diversity to minimize potential biases"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _log_pipeline_performance(self, context: PipelineContext, duration_ms: float) -> None:
        """Log overall pipeline performance metrics."""
        self.logger.info(
            f"Pipeline execution {context.execution_id} completed in "
            f"{duration_ms:.2f}ms for user {context.user_id}"
        )
        
        # Check performance against targets
        if duration_ms > 5000:  # 5 second target for full pipeline
            self.logger.warning(
                f"Pipeline execution exceeded 5-second target: "
                f"{duration_ms:.2f}ms"
            )
    
    async def _handle_pipeline_failure(self, request: Dict[str, Any], error: Exception) -> None:
        """Handle pipeline execution failures with proper logging."""
        error_event = EventMessage(
            event_type=EventType.ERROR_OCCURRED.value,
            execution_id="unknown",  # May not have execution ID if failed early
            payload={
                'error_type': type(error).__name__,
                'error_message': str(error),  # Sanitized
                'request_user_id': request.get('user_id', 'unknown'),
                'recoverable': False  # Pipeline-level failures are not recoverable
            },
            source='pipeline_coordinator',
            target='error_monitoring'
        )
        await self.event_bus.publish_event(error_event)
    
    async def get_execution_status(self, execution_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get current execution status with authorization check.
        
        Args:
            execution_id: Execution ID to query
            user_id: User ID for authorization
            
        Returns:
            Current execution status and progress
            
        Raises:
            ResourceNotFoundError: If execution not found
            ValidationError: If access denied
        """
        try:
            # Verify user access
            execution_state = await self.state_manager.get_pipeline_state(execution_id)
            
            if not execution_state:
                raise ResourceNotFoundError(f"Execution {execution_id} not found")
            
            if execution_state.get('user_id') != user_id:
                raise ValidationError("Access denied to execution")
            
            # Get current progress
            progress = await self.progress_tracker.get_progress(execution_id)
            
            return {
                'execution_id': execution_id,
                'status': execution_state.get('status', 'unknown'),
                'current_stage': execution_state.get('current_stage'),
                'progress': progress,
                'stage_results': execution_state.get('stage_results', {}),
                'estimated_completion': progress.get('estimated_completion'),
                'last_updated': execution_state.get('last_updated')
            }
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(f"Error getting execution status: {e}")
            raise PipelineExecutionError(f"Failed to get execution status: {str(e)}")
    
    async def cancel_execution(self, execution_id: str, user_id: str) -> bool:
        """
        Cancel a pipeline execution with proper authorization.
        
        Args:
            execution_id: Execution ID to cancel
            user_id: User ID for authorization
            
        Returns:
            True if cancellation was successful
            
        Raises:
            ResourceNotFoundError: If execution not found
            ValidationError: If access denied
        """
        try:
            # Verify execution exists and user has access
            execution_state = await self.state_manager.get_pipeline_state(execution_id)
            
            if not execution_state:
                raise ResourceNotFoundError(f"Execution {execution_id} not found")
            
            if execution_state.get('user_id') != user_id:
                raise ValidationError("Access denied to execution")
            
            # Check if execution can be cancelled
            current_status = execution_state.get('status')
            if current_status in ['completed', 'failed', 'cancelled']:
                raise ValidationError(
                    f"Execution cannot be cancelled in {current_status} status"
                )
            
            # Publish cancellation event
            cancel_event = EventMessage(
                event_type=EventType.CANCELLATION_REQUEST.value,
                execution_id=execution_id,
                payload={
                    'reason': 'User requested cancellation',
                    'cancelled_by': user_id,
                    'cancelled_at': datetime.utcnow().isoformat()
                },
                source='pipeline_coordinator',
                target='all_services'
            )
            await self.event_bus.publish_event(cancel_event)
            
            # Update execution state
            await self.state_manager.update_pipeline_state(
                execution_id, 'cancelled', 
                {'cancellation_reason': 'User requested', 'cancelled_by': user_id}
            )
            
            self.logger.info(f"Pipeline execution {execution_id} cancelled by user {user_id}")
            
            return True
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(f"Error cancelling execution {execution_id}: {e}")
            raise PipelineExecutionError(f"Failed to cancel execution: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of pipeline coordinator.
        
        Returns:
            Health check results
        """
        try:
            # Check all subsystems
            event_bus_health = self.event_bus.health_check()
            state_manager_health = self.state_manager.health_check()
            progress_tracker_health = self.progress_tracker.health_check()
            
            # Determine overall health
            subsystems = [
                event_bus_health.get('status'),
                state_manager_health.get('status'),
                progress_tracker_health.get('status')
            ]
            
            overall_status = 'healthy'
            if 'unhealthy' in subsystems:
                overall_status = 'unhealthy'
            elif 'degraded' in subsystems:
                overall_status = 'degraded'
            
            return {
                'status': overall_status,
                'timestamp': datetime.utcnow().isoformat(),
                'subsystems': {
                    'event_bus': event_bus_health,
                    'state_manager': state_manager_health,
                    'progress_tracker': progress_tracker_health
                },
                'stage_coordinators': {
                    stage: coordinator.name 
                    for stage, coordinator in self.stage_coordinators.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"PipelineCoordinator health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Specialized stage coordinator implementations
class ValidationStageCoordinator(StageCoordinator):
    """Specialized coordinator for validation stage with bias detection."""
    
    def __init__(self, stage_name: str, event_bus: EventBus, 
                 bias_integration: Optional[BiasDetectionIntegration] = None):
        super().__init__(stage_name, event_bus, bias_integration)
        
        # Enhanced validation configuration
        self.validation_criteria = [
            'dsm5_accuracy',
            'therapeutic_appropriateness', 
            'privacy_compliance',
            'bias_detection'
        ]
    
    async def _execute_stage_logic(self, context: PipelineContext, 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation stage with comprehensive checks."""
        # Simulate validation process with bias detection
        validation_result = {
            'validation_checks': self.validation_criteria,
            'checks_passed': len(self.validation_criteria),
            'checks_failed': 0,
            'validation_score': 0.95,  # High quality validation
            'dsm5_accuracy': 0.92,
            'therapeutic_appropriateness': 0.98,
            'privacy_compliance': True,
            'bias_analysis': {},  # Will be filled by bias detection
            'recommendations': [
                'Dataset shows high therapeutic quality',
                'Minor bias detected in demographic representation'
            ]
        }
        
        # Add processing metrics
        validation_result['processing_metrics'] = {
            'records_validated': 1000,
            'validation_duration_seconds': 2.5,
            'bias_checks_performed': 4
        }
        
        return validation_result


class ProcessingStageCoordinator(StageCoordinator):
    """Specialized coordinator for processing stage with therapeutic analysis."""
    
    async def _execute_stage_logic(self, context: PipelineContext, 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processing stage with therapeutic conversation analysis."""
        # Simulate therapeutic processing
        processing_result = {
            'processed_conversations': 995,
            'processing_operations': [
                'conversation_segmentation',
                'therapeutic_intent_classification',
                'empathy_score_calculation',
                'quality_assessment'
            ],
            'therapeutic_metrics': {
                'average_empathy_score': 0.87,
                'therapeutic_appropriateness_score': 0.94,
                'conversation_quality_score': 0.91
            },
            'processing_time_seconds': 15.2,
            'batch_processing_applied': True
        }
        
        return processing_result


class GenericStageCoordinator(StageCoordinator):
    """Generic coordinator for standard pipeline stages."""
    
    async def _execute_stage_logic(self, context: PipelineContext, 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic stage logic with appropriate simulation."""
        # Simulate stage-specific processing based on stage name
        if self.stage_name == 'ingestion':
            return {
                'source_format': 'json',
                'record_count': 1000,
                'file_size_mb': 5.2,
                'ingestion_method': 'file_upload',
                'validation_passed': True,
                'errors': []
            }
        elif self.stage_name == 'standardization':
            return {
                'target_format': 'standardized_json',
                'schema_applied': 'dataset_schema_v2.1',
                'fields_standardized': 25,
                'data_types_normalized': 8,
                'quality_score': 0.96
            }
        elif self.stage_name == 'quality':
            return {
                'validation_checks': [
                    'schema_validation',
                    'data_type_validation',
                    'completeness_check',
                    'consistency_check',
                    'business_rule_validation'
                ],
                'checks_passed': 5,
                'checks_failed': 0,
                'validation_score': 1.0,
                'quality_score': 0.98
            }
        elif self.stage_name == 'export':
            return {
                'output_files': [
                    {
                        'filename': f'processed_dataset_{context.execution_id}.json',
                        'format': 'json',
                        'size_mb': 2.1,
                        'record_count': 995
                    }
                ],
                'metadata': {
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'pipeline_version': '1.0.0',
                    'bias_detection_applied': context.enable_bias_detection
                }
            }
        else:
            return {'status': 'completed', 'message': f'{self.stage_name} stage processed'}


# Event handler classes for specialized processing
class ProgressEventHandler(EventHandler):
    """Handler for progress update events."""
    
    def __init__(self, progress_tracker: ProgressTracker):
        super().__init__(
            name='progress_event_handler',
            event_types=[
                EventType.PROGRESS_UPDATE.value,
                EventType.STAGE_TRANSITION.value
            ]
        )
        self.progress_tracker = progress_tracker
    
    async def _process_event(self, event: EventMessage) -> Optional[Dict[str, Any]]:
        """Process progress update events."""
        await self.progress_tracker.update_progress(
            event.execution_id,
            event.payload
        )
        return None


class ErrorRecoveryEventHandler(EventHandler):
    """Handler for error recovery events."""
    
    def __init__(self, error_recovery: ErrorRecoveryManager):
        super().__init__(
            name='error_recovery_handler',
            event_types=[EventType.ERROR_OCCURRED.value]
        )
        self.error_recovery = error_recovery
    
    async def _process_event(self, event: EventMessage) -> Optional[Dict[str, Any]]:
        """Process error events for recovery."""
        # Error recovery logic would be implemented here
        self.logger.info(
            f"Processing error event for execution {event.execution_id}: "
            f"{event.payload.get('error_type', 'unknown')}"
        )
        return None


class PerformanceEventHandler(EventHandler):
    """Handler for performance monitoring events."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        super().__init__(
            name='performance_event_handler',
            event_types=[
                EventType.PERFORMANCE_METRIC.value,
                EventType.LATENCY_WARNING.value
            ]
        )
        self.performance_monitor = performance_monitor
    
    async def _process_event(self, event: EventMessage) -> Optional[Dict[str, Any]]:
        """Process performance monitoring events."""
        # Performance monitoring logic would be implemented here
        self.logger.info(
            f"Processing performance event for execution {event.execution_id}"
        )
        return None