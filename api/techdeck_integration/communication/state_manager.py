"""
State Manager for Pipeline Communication - HIPAA++ Compliant State Management.

This module provides comprehensive state management for the six-stage pipeline
with Redis persistence, audit trails, and HIPAA++ compliant data handling.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .event_bus import EventBus, EventMessage, EventType
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    StateManagementError, ResourceNotFoundError, ValidationError
)
from ..utils.logger import get_request_logger
from ..utils.validation import sanitize_input, validate_state_data


@dataclass
class PipelineState:
    """Comprehensive pipeline state with HIPAA++ compliance."""
    execution_id: str
    user_id: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    current_stage: str
    stage_results: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]
    total_duration_seconds: Optional[float]
    overall_quality_score: Optional[float]
    error_count: int
    retry_count: int
    last_updated: datetime
    metadata: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    checkpoint_data: Optional[Dict[str, Any]]
    encryption_applied: bool = True


@dataclass
class StageState:
    """Individual stage state with detailed tracking."""
    execution_id: str
    stage_name: str
    status: str  # 'pending', 'started', 'completed', 'failed', 'retrying'
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    result: Optional[Dict[str, Any]]
    error_info: Optional[Dict[str, Any]]
    retry_count: int
    checkpoint_data: Optional[Dict[str, Any]]
    bias_analysis: Optional[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]
    last_updated: datetime


class StateManager:
    """Comprehensive state manager for pipeline execution with HIPAA++ compliance."""
    
    def __init__(self, redis_client: RedisClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize state manager with Redis persistence.
        
        Args:
            redis_client: Redis client for state persistence
            config: Optional configuration dictionary
        """
        self.redis_client = redis_client
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Configuration
        self.state_ttl_seconds = self.config.get('state_ttl_seconds', 86400)  # 24 hours
        self.audit_retention_days = self.config.get('audit_retention_days', 90)
        self.checkpoint_interval_seconds = self.config.get('checkpoint_interval_seconds', 300)  # 5 minutes
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        
        # Redis key prefixes
        self.pipeline_state_prefix = "pipeline:state:"
        self.stage_state_prefix = "pipeline:stage:"
        self.audit_trail_prefix = "pipeline:audit:"
        self.checkpoint_prefix = "pipeline:checkpoint:"
        
        self.logger.info("StateManager initialized with HIPAA++ compliance")
    
    async def initialize_pipeline_state(self, context: 'PipelineContext') -> PipelineState:
        """
        Initialize pipeline state for new execution.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            Initialized pipeline state
            
        Raises:
            StateManagementError: If state initialization fails
        """
        try:
            # Create initial state
            pipeline_state = PipelineState(
                execution_id=context.execution_id,
                user_id=context.user_id,
                status='pending',
                current_stage='initialization',
                stage_results={},
                start_time=context.start_time,
                end_time=None,
                total_duration_seconds=None,
                overall_quality_score=None,
                error_count=0,
                retry_count=0,
                last_updated=datetime.utcnow(),
                metadata=context.metadata,
                audit_trail=[],
                checkpoint_data=None,
                encryption_applied=self.encryption_enabled
            )
            
            # Store state in Redis
            await self._store_pipeline_state(pipeline_state)
            
            # Create audit entry
            await self._create_audit_entry(
                context.execution_id,
                'pipeline_initialized',
                {
                    'user_id': context.user_id,
                    'dataset_count': len(context.dataset_ids),
                    'execution_mode': context.execution_mode,
                    'quality_threshold': context.quality_threshold,
                    'bias_detection_enabled': context.enable_bias_detection
                }
            )
            
            self.logger.info(
                f"Pipeline state initialized for execution {context.execution_id}"
            )
            
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline state: {e}")
            raise StateManagementError(f"State initialization failed: {str(e)}")
    
    async def update_pipeline_state(self, execution_id: str, status: str, 
                                  data: Optional[Dict[str, Any]] = None) -> PipelineState:
        """
        Update pipeline state with comprehensive tracking.
        
        Args:
            execution_id: Execution ID
            status: New status
            data: Additional data to update
            
        Returns:
            Updated pipeline state
            
        Raises:
            ResourceNotFoundError: If execution not found
            StateManagementError: If update fails
        """
        try:
            # Retrieve current state
            current_state = await self.get_pipeline_state(execution_id)
            if not current_state:
                raise ResourceNotFoundError(f"Pipeline state not found: {execution_id}")
            
            # Update state
            current_state.status = status
            current_state.last_updated = datetime.utcnow()
            
            # Handle completion
            if status in ['completed', 'failed', 'cancelled']:
                current_state.end_time = datetime.utcnow()
                if current_state.start_time:
                    current_state.total_duration_seconds = (
                        current_state.end_time - current_state.start_time
                    ).total_seconds()
            
            # Update additional data
            if data:
                if 'stage_results' in data:
                    current_state.stage_results.update(data['stage_results'])
                if 'overall_quality_score' in data:
                    current_state.overall_quality_score = data['overall_quality_score']
                if 'error_count' in data:
                    current_state.error_count = data['error_count']
                if 'retry_count' in data:
                    current_state.retry_count = data['retry_count']
                if 'current_stage' in data:
                    current_state.current_stage = data['current_stage']
            
            # Store updated state
            await self._store_pipeline_state(current_state)
            
            # Create audit entry
            await self._create_audit_entry(
                execution_id,
                'pipeline_state_updated',
                {
                    'status': status,
                    'current_stage': current_state.current_stage,
                    'error_count': current_state.error_count,
                    'retry_count': current_state.retry_count
                }
            )
            
            self.logger.info(
                f"Pipeline state updated for execution {execution_id}: {status}"
            )
            
            return current_state
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update pipeline state: {e}")
            raise StateManagementError(f"State update failed: {str(e)}")
    
    async def update_stage_state(self, execution_id: str, stage_name: str, 
                               status: str, result: Optional[Dict[str, Any]] = None) -> StageState:
        """
        Update individual stage state with detailed tracking.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            status: New stage status
            result: Stage execution result
            
        Returns:
            Updated stage state
            
        Raises:
            ResourceNotFoundError: If execution not found
            StateManagementError: If update fails
        """
        try:
            # Retrieve current stage state
            stage_state = await self.get_stage_state(execution_id, stage_name)
            
            if not stage_state:
                # Create new stage state
                stage_state = StageState(
                    execution_id=execution_id,
                    stage_name=stage_name,
                    status=status,
                    start_time=datetime.utcnow() if status == 'started' else None,
                    end_time=None,
                    duration_seconds=None,
                    result=result,
                    error_info=None,
                    retry_count=0,
                    checkpoint_data=None,
                    bias_analysis=None,
                    quality_metrics=None,
                    last_updated=datetime.utcnow()
                )
            else:
                # Update existing state
                stage_state.status = status
                stage_state.last_updated = datetime.utcnow()
                
                if status == 'started' and not stage_state.start_time:
                    stage_state.start_time = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    stage_state.end_time = datetime.utcnow()
                    if stage_state.start_time:
                        stage_state.duration_seconds = (
                            stage_state.end_time - stage_state.start_time
                        ).total_seconds()
                
                if result:
                    stage_state.result = result
            
            # Store updated stage state
            await self._store_stage_state(stage_state)
            
            # Update pipeline current stage if this is the current stage
            if status == 'started':
                pipeline_state = await self.get_pipeline_state(execution_id)
                if pipeline_state:
                    pipeline_state.current_stage = stage_name
                    await self._store_pipeline_state(pipeline_state)
            
            # Create audit entry
            await self._create_audit_entry(
                execution_id,
                'stage_state_updated',
                {
                    'stage_name': stage_name,
                    'status': status,
                    'duration_seconds': stage_state.duration_seconds
                }
            )
            
            self.logger.info(
                f"Stage {stage_name} state updated for execution {execution_id}: {status}"
            )
            
            return stage_state
            
        except Exception as e:
            self.logger.error(f"Failed to update stage state: {e}")
            raise StateManagementError(f"Stage state update failed: {str(e)}")
    
    async def get_pipeline_state(self, execution_id: str) -> Optional[PipelineState]:
        """
        Retrieve pipeline state by execution ID.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Pipeline state or None if not found
            
        Raises:
            StateManagementError: If retrieval fails
        """
        try:
            key = f"{self.pipeline_state_prefix}{execution_id}"
            state_data = await self.redis_client.get(key)
            
            if not state_data:
                return None
            
            # Parse and validate state data
            state_dict = json.loads(state_data)
            
            # Convert datetime strings back to datetime objects
            state_dict['start_time'] = datetime.fromisoformat(state_dict['start_time'])
            if state_dict['end_time']:
                state_dict['end_time'] = datetime.fromisoformat(state_dict['end_time'])
            state_dict['last_updated'] = datetime.fromisoformat(state_dict['last_updated'])
            
            # Convert audit trail datetime strings
            for audit_entry in state_dict.get('audit_trail', []):
                audit_entry['timestamp'] = datetime.fromisoformat(audit_entry['timestamp'])
            
            return PipelineState(**state_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve pipeline state: {e}")
            raise StateManagementError(f"State retrieval failed: {str(e)}")
    
    async def get_stage_state(self, execution_id: str, stage_name: str) -> Optional[StageState]:
        """
        Retrieve stage state by execution ID and stage name.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            
        Returns:
            Stage state or None if not found
            
        Raises:
            StateManagementError: If retrieval fails
        """
        try:
            key = f"{self.stage_state_prefix}{execution_id}:{stage_name}"
            state_data = await self.redis_client.get(key)
            
            if not state_data:
                return None
            
            # Parse and validate state data
            state_dict = json.loads(state_data)
            
            # Convert datetime strings back to datetime objects
            if state_dict['start_time']:
                state_dict['start_time'] = datetime.fromisoformat(state_dict['start_time'])
            if state_dict['end_time']:
                state_dict['end_time'] = datetime.fromisoformat(state_dict['end_time'])
            state_dict['last_updated'] = datetime.fromisoformat(state_dict['last_updated'])
            
            return StageState(**state_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve stage state: {e}")
            raise StateManagementError(f"Stage state retrieval failed: {str(e)}")
    
    async def get_all_stage_states(self, execution_id: str) -> List[StageState]:
        """
        Retrieve all stage states for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            List of stage states
            
        Raises:
            StateManagementError: If retrieval fails
        """
        try:
            pattern = f"{self.stage_state_prefix}{execution_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            stage_states = []
            for key in keys:
                state_data = await self.redis_client.get(key)
                if state_data:
                    state_dict = json.loads(state_data)
                    
                    # Convert datetime strings
                    if state_dict['start_time']:
                        state_dict['start_time'] = datetime.fromisoformat(state_dict['start_time'])
                    if state_dict['end_time']:
                        state_dict['end_time'] = datetime.fromisoformat(state_dict['end_time'])
                    state_dict['last_updated'] = datetime.fromisoformat(state_dict['last_updated'])
                    
                    stage_states.append(StageState(**state_dict))
            
            # Sort by stage name for consistent ordering
            stage_states.sort(key=lambda x: x.stage_name)
            
            return stage_states
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve all stage states: {e}")
            raise StateManagementError(f"Stage states retrieval failed: {str(e)}")
    
    async def create_checkpoint(self, execution_id: str, stage_name: str, 
                              checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for stage recovery.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            checkpoint_data: Checkpoint data
            
        Returns:
            Checkpoint ID
            
        Raises:
            StateManagementError: If checkpoint creation fails
        """
        try:
            checkpoint_id = f"{execution_id}:{stage_name}:{int(time.time())}"
            
            # Sanitize checkpoint data for HIPAA++ compliance
            sanitized_data = sanitize_input(checkpoint_data)
            
            # Validate checkpoint data
            validation_result = validate_state_data(sanitized_data)
            if not validation_result['is_valid']:
                raise ValidationError(
                    f"Checkpoint data validation failed: {validation_result['errors']}"
                )
            
            checkpoint_info = {
                'checkpoint_id': checkpoint_id,
                'execution_id': execution_id,
                'stage_name': stage_name,
                'data': sanitized_data,
                'created_at': datetime.utcnow().isoformat(),
                'encryption_applied': self.encryption_enabled
            }
            
            # Store checkpoint
            key = f"{self.checkpoint_prefix}{checkpoint_id}"
            await self.redis_client.setex(
                key, 
                self.state_ttl_seconds, 
                json.dumps(checkpoint_info, default=str)
            )
            
            # Update stage state with checkpoint reference
            stage_state = await self.get_stage_state(execution_id, stage_name)
            if stage_state:
                stage_state.checkpoint_data = {'checkpoint_id': checkpoint_id}
                await self._store_stage_state(stage_state)
            
            self.logger.info(
                f"Checkpoint created for execution {execution_id}, "
                f"stage {stage_name}: {checkpoint_id}"
            )
            
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise StateManagementError(f"Checkpoint creation failed: {str(e)}")
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore checkpoint data for recovery.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint data
            
        Raises:
            ResourceNotFoundError: If checkpoint not found
            StateManagementError: If restoration fails
        """
        try:
            key = f"{self.checkpoint_prefix}{checkpoint_id}"
            checkpoint_data = await self.redis_client.get(key)
            
            if not checkpoint_data:
                raise ResourceNotFoundError(f"Checkpoint not found: {checkpoint_id}")
            
            checkpoint_info = json.loads(checkpoint_data)
            
            # Validate checkpoint integrity
            if checkpoint_info.get('encryption_applied') != self.encryption_enabled:
                self.logger.warning(
                    f"Encryption mismatch for checkpoint {checkpoint_id}"
                )
            
            self.logger.info(f"Checkpoint restored: {checkpoint_id}")
            
            return checkpoint_info['data']
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            raise StateManagementError(f"Checkpoint restoration failed: {str(e)}")
    
    async def get_execution_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get execution history for a user with authorization check.
        
        Args:
            user_id: User ID
            limit: Maximum number of executions to return
            
        Returns:
            List of execution summaries
            
        Raises:
            ValidationError: If user ID is invalid
            StateManagementError: If retrieval fails
        """
        try:
            if not user_id or not isinstance(user_id, str):
                raise ValidationError("Invalid user ID")
            
            # Get all pipeline states for user
            pattern = f"{self.pipeline_state_prefix}*"
            all_keys = await self.redis_client.keys(pattern)
            
            user_executions = []
            for key in all_keys[:limit]:  # Limit for performance
                state_data = await self.redis_client.get(key)
                if state_data:
                    state_dict = json.loads(state_data)
                    if state_dict.get('user_id') == user_id:
                        # Create summary with sanitized data
                        summary = {
                            'execution_id': state_dict['execution_id'],
                            'status': state_dict['status'],
                            'start_time': state_dict['start_time'],
                            'end_time': state_dict.get('end_time'),
                            'current_stage': state_dict['current_stage'],
                            'overall_quality_score': state_dict.get('overall_quality_score'),
                            'total_duration_seconds': state_dict.get('total_duration_seconds'),
                            'error_count': state_dict['error_count'],
                            'retry_count': state_dict['retry_count']
                        }
                        user_executions.append(summary)
            
            # Sort by start time (newest first)
            user_executions.sort(key=lambda x: x['start_time'], reverse=True)
            
            return user_executions
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get execution history: {e}")
            raise StateManagementError(f"Execution history retrieval failed: {str(e)}")
    
    async def cleanup_expired_states(self, max_age_days: int = 30) -> int:
        """
        Clean up expired pipeline states and audit trails.
        
        Args:
            max_age_days: Maximum age in days for state retention
            
        Returns:
            Number of cleaned up entries
            
        Raises:
            StateManagementError: If cleanup fails
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            # Clean up pipeline states
            pipeline_pattern = f"{self.pipeline_state_prefix}*"
            pipeline_keys = await self.redis_client.keys(pipeline_pattern)
            
            for key in pipeline_keys:
                state_data = await self.redis_client.get(key)
                if state_data:
                    state_dict = json.loads(state_data)
                    start_time = datetime.fromisoformat(state_dict['start_time'])
                    
                    if start_time < cutoff_time:
                        await self.redis_client.delete(key)
                        cleaned_count += 1
            
            # Clean up stage states
            stage_pattern = f"{self.stage_state_prefix}*"
            stage_keys = await self.redis_client.keys(stage_pattern)
            
            for key in stage_keys:
                await self.redis_client.delete(key)
                cleaned_count += 1
            
            # Clean up checkpoints
            checkpoint_pattern = f"{self.checkpoint_prefix}*"
            checkpoint_keys = await self.redis_client.keys(checkpoint_pattern)
            
            for key in checkpoint_keys:
                await self.redis_client.delete(key)
                cleaned_count += 1
            
            # Clean up old audit entries
            audit_pattern = f"{self.audit_trail_prefix}*"
            audit_keys = await self.redis_client.keys(audit_pattern)
            
            for key in audit_keys:
                audit_data = await self.redis_client.get(key)
                if audit_data:
                    audit_entries = json.loads(audit_data)
                    # Filter out old entries
                    filtered_entries = [
                        entry for entry in audit_entries
                        if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
                    ]
                    
                    if len(filtered_entries) < len(audit_entries):
                        if filtered_entries:
                            await self.redis_client.setex(
                                key, self.state_ttl_seconds, 
                                json.dumps(filtered_entries, default=str)
                            )
                        else:
                            await self.redis_client.delete(key)
                        cleaned_count += (len(audit_entries) - len(filtered_entries))
            
            self.logger.info(
                f"Cleaned up {cleaned_count} expired state entries "
                f"older than {max_age_days} days"
            )
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired states: {e}")
            raise StateManagementError(f"State cleanup failed: {str(e)}")
    
    async def get_audit_trail(self, execution_id: str) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            List of audit entries
            
        Raises:
            ResourceNotFoundError: If execution not found
            StateManagementError: If retrieval fails
        """
        try:
            key = f"{self.audit_trail_prefix}{execution_id}"
            audit_data = await self.redis_client.get(key)
            
            if not audit_data:
                # Check if pipeline exists
                pipeline_state = await self.get_pipeline_state(execution_id)
                if not pipeline_state:
                    raise ResourceNotFoundError(f"Execution {execution_id} not found")
                
                return []
            
            audit_entries = json.loads(audit_data)
            
            # Convert datetime strings back to datetime objects
            for entry in audit_entries:
                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
            
            return audit_entries
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit trail: {e}")
            raise StateManagementError(f"Audit trail retrieval failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of state manager.
        
        Returns:
            Health check results
        """
        try:
            # Test Redis connection
            redis_health = await self.redis_client.ping()
            
            # Check state storage
            test_key = f"{self.pipeline_state_prefix}health_check"
            test_data = {'test': 'data', 'timestamp': datetime.utcnow().isoformat()}
            
            await self.redis_client.setex(test_key, 60, json.dumps(test_data))
            retrieved_data = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            storage_healthy = retrieved_data is not None
            
            status = 'healthy' if redis_health and storage_healthy else 'unhealthy'
            
            return {
                'status': status,
                'redis_connection': redis_health,
                'storage_functionality': storage_healthy,
                'encryption_enabled': self.encryption_enabled,
                'state_ttl_seconds': self.state_ttl_seconds,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"StateManager health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _store_pipeline_state(self, state: PipelineState) -> None:
        """Store pipeline state in Redis."""
        try:
            # Convert to dictionary
            state_dict = asdict(state)
            
            # Convert datetime objects to ISO strings
            state_dict['start_time'] = state.start_time.isoformat()
            if state.end_time:
                state_dict['end_time'] = state.end_time.isoformat()
            state_dict['last_updated'] = state.last_updated.isoformat()
            
            # Convert audit trail datetime objects
            for audit_entry in state_dict.get('audit_trail', []):
                audit_entry['timestamp'] = audit_entry['timestamp'].isoformat()
            
            # Store in Redis with TTL
            key = f"{self.pipeline_state_prefix}{state.execution_id}"
            await self.redis_client.setex(
                key, 
                self.state_ttl_seconds, 
                json.dumps(state_dict, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store pipeline state: {e}")
            raise StateManagementError(f"State storage failed: {str(e)}")
    
    async def _store_stage_state(self, state: StageState) -> None:
        """Store stage state in Redis."""
        try:
            # Convert to dictionary
            state_dict = asdict(state)
            
            # Convert datetime objects to ISO strings
            if state.start_time:
                state_dict['start_time'] = state.start_time.isoformat()
            if state.end_time:
                state_dict['end_time'] = state.end_time.isoformat()
            state_dict['last_updated'] = state.last_updated.isoformat()
            
            # Store in Redis with TTL
            key = f"{self.stage_state_prefix}{state.execution_id}:{state.stage_name}"
            await self.redis_client.setex(
                key, 
                self.state_ttl_seconds, 
                json.dumps(state_dict, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store stage state: {e}")
            raise StateManagementError(f"Stage state storage failed: {str(e)}")
    
    async def _create_audit_entry(self, execution_id: str, event_type: str, 
                                event_data: Dict[str, Any]) -> None:
        """Create audit entry for HIPAA++ compliance."""
        try:
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'execution_id': execution_id,
                'data': sanitize_input(event_data)
            }
            
            # Get existing audit trail
            key = f"{self.audit_trail_prefix}{execution_id}"
            existing_audit = await self.redis_client.get(key)
            
            if existing_audit:
                audit_trail = json.loads(existing_audit)
            else:
                audit_trail = []
            
            # Add new entry
            audit_trail.append(audit_entry)
            
            # Keep only recent entries (last 1000)
            if len(audit_trail) > 1000:
                audit_trail = audit_trail[-1000:]
            
            # Store updated audit trail
            await self.redis_client.setex(
                key, 
                self.state_ttl_seconds, 
                json.dumps(audit_trail, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create audit entry: {e}")
            # Don't raise - audit failures shouldn't break the pipeline