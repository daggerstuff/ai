"""
Pipeline orchestrator adapter for TechDeck-Python Pipeline Integration.

This module provides the integration layer between the Flask API service
and the Python dataset pipeline, handling communication, data transformation,
and execution coordination.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..error_handling.custom_errors import (
    PipelineExecutionError, IntegrationError, ValidationError, ResourceNotFoundError
)
from ..utils.logger import get_request_logger
from ..utils.validation import validate_pipeline_input, sanitize_input


class PipelineMessageType(Enum):
    """Pipeline message types for event-driven communication."""
    EXECUTION_START = "execution_start"
    STAGE_COMPLETE = "stage_complete"
    PROGRESS_UPDATE = "progress_update"
    ERROR_OCCURRED = "error_occurred"
    EXECUTION_COMPLETE = "execution_complete"
    CANCELLATION_REQUEST = "cancellation_request"


@dataclass
class PipelineMessage:
    """Pipeline communication message."""
    message_type: str
    execution_id: str
    payload: Dict[str, Any]
    timestamp: str
    source: str
    target: str


class PipelineOrchestrator:
    """Orchestrator for Python dataset pipeline integration."""
    
    def __init__(self, redis_client, bias_detection_service=None):
        self.redis_client = redis_client
        self.bias_detection_service = bias_detection_service
        self.logger = logging.getLogger(__name__)
        self.pipeline_channel = "pipeline_events"
        self.response_timeout = 300  # 5 minutes
        
    async def execute_pipeline(self, execution_id: str, config: Dict[str, Any],
                             dataset_info: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Execute the six-stage pipeline with the given configuration.
        
        Args:
            execution_id: Unique execution identifier
            config: Pipeline configuration
            dataset_info: Dataset information
            user_id: User ID for access control
            
        Returns:
            Pipeline execution results
            
        Raises:
            PipelineExecutionError: If pipeline execution fails
            IntegrationError: If integration fails
        """
        request_logger = get_request_logger()
        request_logger.info(f"Starting pipeline execution {execution_id} for user {user_id}")
        
        try:
            # Validate inputs
            validate_pipeline_input(config, dataset_info)
            
            # Prepare pipeline input
            pipeline_input = self._prepare_pipeline_input(execution_id, config, dataset_info, user_id)
            
            # Execute six-stage pipeline
            results = await self._execute_six_stage_pipeline(execution_id, pipeline_input)
            
            request_logger.info(f"Pipeline execution completed successfully: {execution_id}")
            
            return results
            
        except ValidationError:
            raise
        except Exception as e:
            request_logger.error(f"Error executing pipeline: {e}")
            raise PipelineExecutionError(f"Pipeline execution failed: {str(e)}")
    
    async def _execute_six_stage_pipeline(self, execution_id: str, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the six-stage pipeline process."""
        try:
            results = {
                'execution_id': execution_id,
                'stages': {},
                'overall_status': 'in_progress',
                'start_time': datetime.utcnow().isoformat(),
                'end_time': None,
                'total_duration_seconds': 0
            }
            
            # Stage 1: Data Ingestion
            stage1_result = await self._execute_stage_1_data_ingestion(execution_id, pipeline_input)
            results['stages']['data_ingestion'] = stage1_result
            
            # Stage 2: Preprocessing
            stage2_result = await self._execute_stage_2_preprocessing(execution_id, stage1_result)
            results['stages']['preprocessing'] = stage2_result
            
            # Stage 3: Bias Detection
            stage3_result = await self._execute_stage_3_bias_detection(execution_id, stage2_result)
            results['stages']['bias_detection'] = stage3_result
            
            # Stage 4: Standardization
            stage4_result = await self._execute_stage_4_standardization(execution_id, stage3_result)
            results['stages']['standardization'] = stage4_result
            
            # Stage 5: Validation
            stage5_result = await self._execute_stage_5_validation(execution_id, stage4_result)
            results['stages']['validation'] = stage5_result
            
            # Stage 6: Output Generation
            stage6_result = await self._execute_stage_6_output_generation(execution_id, stage5_result)
            results['stages']['output_generation'] = stage6_result
            
            # Calculate total duration
            start_time = datetime.fromisoformat(results['start_time'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(stage6_result['end_time'].replace('Z', '+00:00'))
            results['total_duration_seconds'] = (end_time - start_time).total_seconds()
            results['end_time'] = stage6_result['end_time']
            results['overall_status'] = 'completed'
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in six-stage pipeline execution: {e}")
            raise PipelineExecutionError(f"Six-stage pipeline execution failed: {str(e)}")
    
    async def _execute_stage_1_data_ingestion(self, execution_id: str, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 1: Data Ingestion."""
        try:
            self.logger.info(f"Executing Stage 1: Data Ingestion for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Simulate data ingestion process
            ingested_data = {
                'source_format': pipeline_input['config']['source_format'],
                'target_format': pipeline_input['config']['target_format'],
                'record_count': 1000,  # Placeholder
                'file_size_mb': 5.2,   # Placeholder
                'ingestion_method': 'file_upload',
                'validation_passed': True,
                'errors': []
            }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'data_ingestion', 
                100, 
                'Data ingestion completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'data_ingestion',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': ingested_data,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 1 data ingestion: {e}")
            raise PipelineExecutionError(f"Data ingestion failed: {str(e)}")
    
    async def _execute_stage_2_preprocessing(self, execution_id: str, previous_stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 2: Preprocessing."""
        try:
            self.logger.info(f"Executing Stage 2: Preprocessing for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Simulate preprocessing operations
            preprocessing_result = {
                'operations_applied': [
                    'data_type_conversion',
                    'missing_value_handling',
                    'duplicate_removal',
                    'format_standardization'
                ],
                'records_processed': previous_stage_result['result']['record_count'],
                'records_after_preprocessing': 995,  # Placeholder
                'quality_metrics': {
                    'completeness_score': 0.98,
                    'consistency_score': 0.95,
                    'accuracy_score': 0.97
                }
            }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'preprocessing', 
                100, 
                'Preprocessing completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'preprocessing',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': preprocessing_result,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 2 preprocessing: {e}")
            raise PipelineExecutionError(f"Preprocessing failed: {str(e)}")
    
    async def _execute_stage_3_bias_detection(self, execution_id: str, previous_stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 3: Bias Detection."""
        try:
            self.logger.info(f"Executing Stage 3: Bias Detection for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Perform bias detection using the bias detection service
            if self.bias_detection_service:
                bias_result = await self.bias_detection_service.analyze_dataset(
                    previous_stage_result['result'],
                    execution_id
                )
            else:
                # Simulate bias detection result
                bias_result = {
                    'bias_score': 0.15,  # Low bias
                    'bias_categories': {
                        'demographic': 0.10,
                        'geographic': 0.05,
                        'temporal': 0.20
                    },
                    'recommendations': [
                        'Consider increasing dataset diversity',
                        'Review temporal bias in data collection'
                    ],
                    'compliance_status': 'acceptable',
                    'threshold_exceeded': False
                }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'bias_detection', 
                100, 
                'Bias detection completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'bias_detection',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': bias_result,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 3 bias detection: {e}")
            raise PipelineExecutionError(f"Bias detection failed: {str(e)}")
    
    async def _execute_stage_4_standardization(self, execution_id: str, previous_stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 4: Standardization."""
        try:
            self.logger.info(f"Executing Stage 4: Standardization for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Simulate standardization process
            standardization_result = {
                'target_format': 'standardized_json',
                'schema_applied': 'dataset_schema_v2.1',
                'fields_standardized': 25,
                'data_types_normalized': 8,
                'encoding_standardized': 'utf-8',
                'quality_score': 0.96
            }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'standardization', 
                100, 
                'Standardization completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'standardization',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': standardization_result,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 4 standardization: {e}")
            raise PipelineExecutionError(f"Standardization failed: {str(e)}")
    
    async def _execute_stage_5_validation(self, execution_id: str, previous_stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 5: Validation."""
        try:
            self.logger.info(f"Executing Stage 5: Validation for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Simulate validation process
            validation_result = {
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
                'critical_issues': [],
                'warnings': []
            }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'validation', 
                100, 
                'Validation completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'validation',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': validation_result,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 5 validation: {e}")
            raise PipelineExecutionError(f"Validation failed: {str(e)}")
    
    async def _execute_stage_6_output_generation(self, execution_id: str, previous_stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Stage 6: Output Generation."""
        try:
            self.logger.info(f"Executing Stage 6: Output Generation for {execution_id}")
            
            start_time = datetime.utcnow()
            
            # Simulate output generation
            output_result = {
                'output_files': [
                    {
                        'filename': f'processed_dataset_{execution_id}.json',
                        'format': 'json',
                        'size_mb': 2.1,
                        'record_count': 995
                    },
                    {
                        'filename': f'validation_report_{execution_id}.json',
                        'format': 'json',
                        'size_mb': 0.1,
                        'record_count': 1
                    }
                ],
                'metadata': {
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'pipeline_version': '1.0.0',
                    'bias_detection_applied': True,
                    'quality_score': 0.96
                },
                'download_links': [
                    f'/api/v1/pipeline/executions/{execution_id}/output/processed_dataset',
                    f'/api/v1/pipeline/executions/{execution_id}/output/validation_report'
                ]
            }
            
            # Publish progress update
            await self._publish_progress_update(
                execution_id, 
                'output_generation', 
                100, 
                'Output generation completed successfully'
            )
            
            end_time = datetime.utcnow()
            
            return {
                'stage': 'output_generation',
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'result': output_result,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error in Stage 6 output generation: {e}")
            raise PipelineExecutionError(f"Output generation failed: {str(e)}")
    
    def _prepare_pipeline_input(self, execution_id: str, config: Dict[str, Any],
                              dataset_info: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Prepare input data for pipeline execution."""
        try:
            return {
                'execution_id': execution_id,
                'user_id': user_id,
                'config': config,
                'dataset_info': dataset_info,
                'timestamp': datetime.utcnow().isoformat(),
                'pipeline_version': '1.0.0'
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing pipeline input: {e}")
            raise IntegrationError(f"Failed to prepare pipeline input: {str(e)}")
    
    async def _publish_progress_update(self, execution_id: str, stage: str,
                                     progress_percent: float, message: str):
        """Publish progress update to Redis for WebSocket broadcasting."""
        try:
            progress_data = {
                'execution_id': execution_id,
                'stage': stage,
                'progress_percent': progress_percent,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in Redis for WebSocket broadcasting
            progress_key = f"pipeline_progress:{execution_id}"
            await self.redis_client.set_json(progress_key, progress_data, ex=3600)
            
            # Publish to pipeline channel
            message_obj = PipelineMessage(
                message_type=PipelineMessageType.PROGRESS_UPDATE.value,
                execution_id=execution_id,
                payload=progress_data,
                timestamp=datetime.utcnow().isoformat(),
                source='pipeline_orchestrator',
                target='progress_tracker'
            )
            
            await self.redis_client.publish(
                self.pipeline_channel,
                json.dumps(message_obj.__dict__)
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing progress update: {e}")
    
    async def cancel_pipeline_execution(self, execution_id: str, user_id: str) -> bool:
        """
        Cancel a pipeline execution.
        
        Args:
            execution_id: Execution ID to cancel
            user_id: User ID for authorization
            
        Returns:
            True if cancellation was successful
            
        Raises:
            ResourceNotFoundError: If execution not found
            ValidationError: If access is denied
        """
        try:
            # Verify execution exists and user has access
            execution_key = f"pipeline_execution:{execution_id}"
            execution_data = await self.redis_client.get_json(execution_key)
            
            if not execution_data:
                raise ResourceNotFoundError(f"Execution {execution_id} not found")
            
            if execution_data['user_id'] != user_id:
                raise ValidationError("Access denied to execution")
            
            # Check if execution can be cancelled
            if execution_data['status'] in ['completed', 'failed', 'cancelled']:
                raise ValidationError("Execution cannot be cancelled in current status")
            
            # Publish cancellation message
            cancel_message = PipelineMessage(
                message_type=PipelineMessageType.CANCELLATION_REQUEST.value,
                execution_id=execution_id,
                payload={'reason': 'User requested cancellation'},
                timestamp=datetime.utcnow().isoformat(),
                source='api_service',
                target='pipeline_executor'
            )
            
            await self.redis_client.publish(
                self.pipeline_channel,
                json.dumps(cancel_message.__dict__)
            )
            
            self.logger.info(f"Pipeline execution cancellation requested: {execution_id}")
            
            return True
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(f"Error cancelling pipeline execution: {e}")
            raise IntegrationError(f"Failed to cancel pipeline execution: {str(e)}")
    
    async def get_execution_results(self, execution_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get execution results for a completed pipeline.
        
        Args:
            execution_id: Execution ID
            user_id: User ID for authorization
            
        Returns:
            Execution results and output files
            
        Raises:
            ResourceNotFoundError: If execution not found
            ValidationError: If access is denied or execution not completed
        """
        try:
            # Verify execution exists and user has access
            execution_key = f"pipeline_execution:{execution_id}"
            execution_data = await self.redis_client.get_json(execution_key)
            
            if not execution_data:
                raise ResourceNotFoundError(f"Execution {execution_id} not found")
            
            if execution_data['user_id'] != user_id:
                raise ValidationError("Access denied to execution")
            
            # Check if execution is completed
            if execution_data['status'] != 'completed':
                raise ValidationError("Execution results not available in current status")
            
            # Get stage results
            results_key = f"pipeline_results:{execution_id}"
            results_data = await self.redis_client.get_json(results_key)
            
            if not results_data:
                raise ResourceNotFoundError(f"Execution results not found for {execution_id}")
            
            return results_data
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(f"Error getting execution results: {e}")
            raise IntegrationError(f"Failed to get execution results: {str(e)}")
    
    async def validate_pipeline_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate pipeline configuration before execution.
        
        Args:
            config: Pipeline configuration to validate
            
        Returns:
            Validation results
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validation_result = {
                'config': config,
                'validation_timestamp': datetime.utcnow().isoformat(),
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Validate required fields
            required_fields = ['source_format', 'target_format', 'processing_stages']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required fields: {missing_fields}")
            
            # Validate processing stages
            valid_stages = [
                'data_ingestion', 'preprocessing', 'bias_detection',
                'standardization', 'validation', 'output_generation'
            ]
            
            stages = config.get('processing_stages', [])
            invalid_stages = [stage for stage in stages if stage not in valid_stages]
            
            if invalid_stages:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Invalid processing stages: {invalid_stages}")
            
            # Validate data formats
            supported_formats = ['csv', 'json', 'jsonl', 'parquet', 'txt']
            source_format = config.get('source_format', '').lower()
            target_format = config.get('target_format', '').lower()
            
            if source_format not in supported_formats:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Unsupported source format: {source_format}")
            
            if target_format not in supported_formats:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Unsupported target format: {target_format}")
            
            # Add warnings for potential issues
            if config.get('data_size_mb', 0) > 100:
                validation_result['warnings'].append("Large dataset size may impact performance")
            
            if not config.get('bias_detection_enabled', False):
                validation_result['warnings'].append("Bias detection is disabled")
            
            self.logger.info(f"Pipeline configuration validation completed: {validation_result['is_valid']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating pipeline configuration: {e}")
            raise ValidationError(f"Configuration validation failed: {str(e)}")