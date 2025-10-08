"""
Progress Tracker for Pipeline Communication - Real-time WebSocket Integration.

This module provides comprehensive progress tracking with WebSocket support,
real-time updates, and HIPAA++ compliant data handling for the six-stage pipeline.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict

from .event_bus import EventBus, EventMessage, EventType
from .state_manager import StateManager
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    ProgressTrackingError, ValidationError, ResourceNotFoundError
)
from ..utils.logger import get_request_logger
from ..utils.validation import sanitize_input


@dataclass
class ProgressUpdate:
    """Progress update with comprehensive tracking."""
    execution_id: str
    stage_name: str
    status: str  # 'pending', 'started', 'in_progress', 'completed', 'failed'
    progress_percent: float  # 0.0 to 100.0
    message: str
    timestamp: datetime
    estimated_completion: Optional[datetime]
    stage_duration_seconds: Optional[float]
    current_operation: Optional[str]
    operations_completed: int
    total_operations: int
    metadata: Dict[str, Any]


@dataclass
class WebSocketConnection:
    """WebSocket connection information."""
    connection_id: str
    user_id: str
    execution_id: Optional[str]
    subscribed_stages: Set[str]
    last_heartbeat: datetime
    connection_established: datetime


class ProgressTracker:
    """Comprehensive progress tracker with WebSocket integration."""
    
    def __init__(self, redis_client: RedisClient, event_bus: EventBus, 
                 state_manager: StateManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize progress tracker with WebSocket support.
        
        Args:
            redis_client: Redis client for state persistence
            event_bus: Event bus for event coordination
            state_manager: State manager for pipeline state
            config: Optional configuration dictionary
        """
        self.redis_client = redis_client
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Configuration
        self.progress_update_interval = self.config.get('progress_update_interval', 1.0)  # seconds
        self.websocket_heartbeat_timeout = self.config.get('websocket_heartbeat_timeout', 30.0)  # seconds
        self.progress_retention_hours = self.config.get('progress_retention_hours', 24)
        self.enable_real_time_updates = self.config.get('enable_real_time_updates', True)
        
        # Redis key prefixes
        self.progress_prefix = "pipeline:progress:"
        self.websocket_prefix = "pipeline:websocket:"
        self.subscriptions_prefix = "pipeline:subscriptions:"
        
        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocketConnection] = {}
        
        # Progress tracking for active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Stage progress mappings
        self.stage_progress_mapping = {
            'ingestion': {'start': 0, 'end': 15},
            'validation': {'start': 15, 'end': 30},
            'processing': {'start': 30, 'end': 50},
            'standardization': {'start': 50, 'end': 70},
            'quality': {'start': 70, 'end': 85},
            'export': {'start': 85, 'end': 100}
        }
        
        self.logger.info("ProgressTracker initialized with WebSocket support")
    
    async def initialize_progress_tracking(self, execution_id: str, user_id: str,
                                         dataset_count: int, execution_mode: str) -> Dict[str, Any]:
        """
        Initialize progress tracking for a new pipeline execution.
        
        Args:
            execution_id: Execution ID
            user_id: User ID
            dataset_count: Number of datasets
            execution_mode: Execution mode
            
        Returns:
            Initial progress configuration
            
        Raises:
            ProgressTrackingError: If initialization fails
        """
        try:
            # Validate inputs
            if not execution_id or not user_id:
                raise ValidationError("Execution ID and User ID are required")
            
            if dataset_count <= 0:
                raise ValidationError("Dataset count must be positive")
            
            # Create initial progress configuration
            progress_config = {
                'execution_id': execution_id,
                'user_id': user_id,
                'start_time': datetime.utcnow().isoformat(),
                'total_stages': 6,
                'current_stage': 'initialization',
                'overall_progress': 0.0,
                'stage_progress': {},
                'estimated_total_duration': self._estimate_total_duration(dataset_count, execution_mode),
                'status': 'initialized',
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Initialize stage progress
            for stage_name in self.stage_progress_mapping.keys():
                progress_config['stage_progress'][stage_name] = {
                    'status': 'pending',
                    'progress_percent': 0.0,
                    'start_time': None,
                    'end_time': None,
                    'duration_seconds': None,
                    'operations_completed': 0,
                    'total_operations': 0,
                    'current_operation': None,
                    'error_count': 0,
                    'retry_count': 0
                }
            
            # Store progress configuration
            key = f"{self.progress_prefix}{execution_id}"
            await self.redis_client.setex(
                key,
                self.progress_retention_hours * 3600,  # Convert to seconds
                json.dumps(progress_config, default=str)
            )
            
            # Track active execution
            self.active_executions[execution_id] = {
                'user_id': user_id,
                'start_time': datetime.utcnow(),
                'last_update': datetime.utcnow(),
                'subscribers': set()
            }
            
            # Publish initialization event
            if self.enable_real_time_updates:
                await self._publish_progress_event(
                    execution_id,
                    'initialization',
                    'initialized',
                    0.0,
                    "Pipeline execution initialized"
                )
            
            self.logger.info(
                f"Progress tracking initialized for execution {execution_id}"
            )
            
            return progress_config
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize progress tracking: {e}")
            raise ProgressTrackingError(f"Progress initialization failed: {str(e)}")
    
    async def update_stage_progress(self, execution_id: str, stage_name: str,
                                  status: str, progress_percent: float,
                                  message: str, current_operation: Optional[str] = None,
                                  operations_completed: Optional[int] = None,
                                  total_operations: Optional[int] = None) -> Dict[str, Any]:
        """
        Update progress for a specific stage.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            status: Stage status
            progress_percent: Progress percentage (0.0-100.0)
            message: Progress message
            current_operation: Current operation description
            operations_completed: Number of completed operations
            total_operations: Total number of operations
            
        Returns:
            Updated progress configuration
            
        Raises:
            ProgressTrackingError: If update fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not execution_id or not stage_name:
                raise ValidationError("Execution ID and stage name are required")
            
            if not (0.0 <= progress_percent <= 100.0):
                raise ValidationError("Progress percent must be between 0.0 and 100.0")
            
            if stage_name not in self.stage_progress_mapping:
                raise ValidationError(f"Invalid stage name: {stage_name}")
            
            # Retrieve current progress
            progress_config = await self.get_progress(execution_id)
            if not progress_config:
                raise ResourceNotFoundError(f"Progress tracking not found for execution {execution_id}")
            
            # Update stage progress
            stage_progress = progress_config['stage_progress'][stage_name]
            
            # Update status and timing
            if status == 'started' and stage_progress['status'] == 'pending':
                stage_progress['start_time'] = datetime.utcnow().isoformat()
            elif status in ['completed', 'failed'] and not stage_progress['end_time']:
                stage_progress['end_time'] = datetime.utcnow().isoformat()
                if stage_progress['start_time']:
                    start_time = datetime.fromisoformat(stage_progress['start_time'])
                    stage_progress['duration_seconds'] = (
                        datetime.utcnow() - start_time
                    ).total_seconds()
            
            stage_progress['status'] = status
            stage_progress['progress_percent'] = progress_percent
            stage_progress['current_operation'] = current_operation
            
            if operations_completed is not None:
                stage_progress['operations_completed'] = operations_completed
            if total_operations is not None:
                stage_progress['total_operations'] = total_operations
            
            # Calculate overall progress
            progress_config['overall_progress'] = self._calculate_overall_progress(
                progress_config['stage_progress']
            )
            
            # Update current stage if this is the active stage
            if status == 'started':
                progress_config['current_stage'] = stage_name
            
            # Update last updated timestamp
            progress_config['last_updated'] = datetime.utcnow().isoformat()
            
            # Store updated progress
            key = f"{self.progress_prefix}{execution_id}"
            await self.redis_client.setex(
                key,
                self.progress_retention_hours * 3600,
                json.dumps(progress_config, default=str)
            )
            
            # Update active execution tracking
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['last_update'] = datetime.utcnow()
            
            # Publish progress event
            if self.enable_real_time_updates:
                await self._publish_progress_event(
                    execution_id,
                    stage_name,
                    status,
                    progress_percent,
                    message,
                    progress_config['overall_progress']
                )
            
            # Notify WebSocket subscribers
            await self._notify_subscribers(execution_id, stage_name, progress_config)
            
            self.logger.info(
                f"Stage {stage_name} progress updated for execution {execution_id}: "
                f"{progress_percent:.1f}% - {message}"
            )
            
            return progress_config
            
        except (ValidationError, ResourceNotFoundError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to update stage progress: {e}")
            raise ProgressTrackingError(f"Progress update failed: {str(e)}")
    
    async def update_operation_progress(self, execution_id: str, stage_name: str,
                                      operation_name: str, operation_progress: float,
                                      operation_message: str) -> Dict[str, Any]:
        """
        Update progress for a specific operation within a stage.
        
        Args:
            execution_id: Execution ID
            stage_name: Stage name
            operation_name: Operation name
            operation_progress: Operation progress (0.0-100.0)
            operation_message: Operation message
            
        Returns:
            Updated progress configuration
            
        Raises:
            ProgressTrackingError: If update fails
        """
        try:
            # Retrieve current progress
            progress_config = await self.get_progress(execution_id)
            if not progress_config:
                raise ResourceNotFoundError(f"Progress tracking not found for execution {execution_id}")
            
            # Update operation progress
            stage_progress = progress_config['stage_progress'][stage_name]
            stage_progress['current_operation'] = operation_name
            
            # Calculate stage progress based on operation progress
            if stage_name in self.stage_progress_mapping:
                stage_range = self.stage_progress_mapping[stage_name]
                stage_progress['progress_percent'] = (
                    stage_range['start'] + 
                    (stage_range['end'] - stage_range['start']) * (operation_progress / 100.0)
                )
            
            # Update overall progress
            progress_config['overall_progress'] = self._calculate_overall_progress(
                progress_config['stage_progress']
            )
            
            progress_config['last_updated'] = datetime.utcnow().isoformat()
            
            # Store updated progress
            key = f"{self.progress_prefix}{execution_id}"
            await self.redis_client.setex(
                key,
                self.progress_retention_hours * 3600,
                json.dumps(progress_config, default=str)
            )
            
            # Publish operation progress event
            if self.enable_real_time_updates:
                await self._publish_operation_event(
                    execution_id,
                    stage_name,
                    operation_name,
                    operation_progress,
                    operation_message
                )
            
            self.logger.info(
                f"Operation {operation_name} progress updated for execution {execution_id}, "
                f"stage {stage_name}: {operation_progress:.1f}% - {operation_message}"
            )
            
            return progress_config
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update operation progress: {e}")
            raise ProgressTrackingError(f"Operation progress update failed: {str(e)}")
    
    async def get_progress(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Progress configuration or None if not found
            
        Raises:
            ProgressTrackingError: If retrieval fails
        """
        try:
            key = f"{self.progress_prefix}{execution_id}"
            progress_data = await self.redis_client.get(key)
            
            if not progress_data:
                return None
            
            return json.loads(progress_data)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve progress: {e}")
            raise ProgressTrackingError(f"Progress retrieval failed: {str(e)}")
    
    async def get_detailed_progress(self, execution_id: str) -> Dict[str, Any]:
        """
        Get detailed progress with stage information and estimates.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Detailed progress information
            
        Raises:
            ResourceNotFoundError: If execution not found
            ProgressTrackingError: If retrieval fails
        """
        try:
            # Get basic progress
            progress_config = await self.get_progress(execution_id)
            if not progress_config:
                raise ResourceNotFoundError(f"Progress not found for execution {execution_id}")
            
            # Get pipeline state for additional details
            pipeline_state = await self.state_manager.get_pipeline_state(execution_id)
            
            # Get stage states
            stage_states = await self.state_manager.get_all_stage_states(execution_id)
            
            # Calculate estimates
            estimates = self._calculate_completion_estimates(progress_config, stage_states)
            
            # Create detailed progress
            detailed_progress = {
                'execution_id': execution_id,
                'overall_progress': progress_config['overall_progress'],
                'current_stage': progress_config['current_stage'],
                'status': progress_config['status'],
                'start_time': progress_config['start_time'],
                'estimated_completion': estimates.get('estimated_completion'),
                'estimated_remaining_seconds': estimates.get('estimated_remaining_seconds'),
                'stage_details': {},
                'quality_metrics': {},
                'performance_metrics': {}
            }
            
            # Add stage details
            for stage_name, stage_progress in progress_config['stage_progress'].items():
                stage_state = next(
                    (ss for ss in stage_states if ss.stage_name == stage_name), 
                    None
                )
                
                detailed_progress['stage_details'][stage_name] = {
                    'status': stage_progress['status'],
                    'progress_percent': stage_progress['progress_percent'],
                    'start_time': stage_progress.get('start_time'),
                    'end_time': stage_progress.get('end_time'),
                    'duration_seconds': stage_progress.get('duration_seconds'),
                    'current_operation': stage_progress.get('current_operation'),
                    'operations_completed': stage_progress.get('operations_completed', 0),
                    'total_operations': stage_progress.get('total_operations', 0),
                    'error_count': stage_progress.get('error_count', 0),
                    'retry_count': stage_progress.get('retry_count', 0),
                    'result_summary': self._get_stage_result_summary(stage_state)
                }
            
            # Add quality metrics if available
            if pipeline_state and pipeline_state.overall_quality_score is not None:
                detailed_progress['quality_metrics'] = {
                    'overall_quality_score': pipeline_state.overall_quality_score,
                    'stage_quality_scores': self._extract_quality_scores(stage_states)
                }
            
            # Add performance metrics
            detailed_progress['performance_metrics'] = self._calculate_performance_metrics(
                progress_config, stage_states
            )
            
            return detailed_progress
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get detailed progress: {e}")
            raise ProgressTrackingError(f"Detailed progress retrieval failed: {str(e)}")
    
    async def register_websocket_connection(self, connection_id: str, user_id: str,
                                          execution_id: Optional[str] = None) -> bool:
        """
        Register a new WebSocket connection.
        
        Args:
            connection_id: WebSocket connection ID
            user_id: User ID
            execution_id: Optional execution ID to subscribe to
            
        Returns:
            True if registration successful
            
        Raises:
            ProgressTrackingError: If registration fails
        """
        try:
            # Validate inputs
            if not connection_id or not user_id:
                raise ValidationError("Connection ID and User ID are required")
            
            # Create connection object
            connection = WebSocketConnection(
                connection_id=connection_id,
                user_id=user_id,
                execution_id=execution_id,
                subscribed_stages=set(),
                last_heartbeat=datetime.utcnow(),
                connection_established=datetime.utcnow()
            )
            
            # Store connection
            self.active_connections[connection_id] = connection
            
            # Store in Redis for persistence
            key = f"{self.websocket_prefix}{connection_id}"
            connection_dict = asdict(connection)
            connection_dict['subscribed_stages'] = list(connection.subscribed_stages)
            
            await self.redis_client.setex(
                key,
                int(self.websocket_heartbeat_timeout * 2),  # 2x heartbeat timeout
                json.dumps(connection_dict, default=str)
            )
            
            # Subscribe to execution if provided
            if execution_id:
                await self.subscribe_to_execution(connection_id, execution_id)
            
            self.logger.info(
                f"WebSocket connection registered: {connection_id} for user {user_id}"
            )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to register WebSocket connection: {e}")
            raise ProgressTrackingError(f"WebSocket registration failed: {str(e)}")
    
    async def unregister_websocket_connection(self, connection_id: str) -> bool:
        """
        Unregister a WebSocket connection.
        
        Args:
            connection_id: WebSocket connection ID
            
        Returns:
            True if unregistration successful
            
        Raises:
            ProgressTrackingError: If unregistration fails
        """
        try:
            # Remove from active connections
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                
                # Unsubscribe from all executions
                if connection.execution_id:
                    await self.unsubscribe_from_execution(connection_id, connection.execution_id)
                
                del self.active_connections[connection_id]
            
            # Remove from Redis
            key = f"{self.websocket_prefix}{connection_id}"
            await self.redis_client.delete(key)
            
            self.logger.info(f"WebSocket connection unregistered: {connection_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister WebSocket connection: {e}")
            raise ProgressTrackingError(f"WebSocket unregistration failed: {str(e)}")
    
    async def subscribe_to_execution(self, connection_id: str, execution_id: str) -> bool:
        """
        Subscribe WebSocket connection to execution progress.
        
        Args:
            connection_id: WebSocket connection ID
            execution_id: Execution ID to subscribe to
            
        Returns:
            True if subscription successful
            
        Raises:
            ProgressTrackingError: If subscription fails
        """
        try:
            # Validate connection exists
            if connection_id not in self.active_connections:
                raise ResourceNotFoundError(f"WebSocket connection not found: {connection_id}")
            
            # Update connection
            connection = self.active_connections[connection_id]
            connection.execution_id = execution_id
            
            # Add to execution subscribers
            if execution_id not in self.active_executions:
                self.active_executions[execution_id] = {
                    'subscribers': set()
                }
            
            self.active_executions[execution_id]['subscribers'].add(connection_id)
            
            # Update Redis
            key = f"{self.websocket_prefix}{connection_id}"
            connection_dict = asdict(connection)
            connection_dict['subscribed_stages'] = list(connection.subscribed_stages)
            
            await self.redis_client.setex(
                key,
                int(self.websocket_heartbeat_timeout * 2),
                json.dumps(connection_dict, default=str)
            )
            
            self.logger.info(
                f"WebSocket connection {connection_id} subscribed to execution {execution_id}"
            )
            
            return True
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to subscribe to execution: {e}")
            raise ProgressTrackingError(f"Execution subscription failed: {str(e)}")
    
    async def unsubscribe_from_execution(self, connection_id: str, execution_id: str) -> bool:
        """
        Unsubscribe WebSocket connection from execution progress.
        
        Args:
            connection_id: WebSocket connection ID
            execution_id: Execution ID to unsubscribe from
            
        Returns:
            True if unsubscription successful
            
        Raises:
            ProgressTrackingError: If unsubscription fails
        """
        try:
            # Remove from execution subscribers
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['subscribers'].discard(connection_id)
                
                # Clean up if no more subscribers
                if not self.active_executions[execution_id]['subscribers']:
                    del self.active_executions[execution_id]
            
            self.logger.info(
                f"WebSocket connection {connection_id} unsubscribed from execution {execution_id}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from execution: {e}")
            raise ProgressTrackingError(f"Execution unsubscription failed: {str(e)}")
    
    async def update_websocket_heartbeat(self, connection_id: str) -> bool:
        """
        Update WebSocket connection heartbeat.
        
        Args:
            connection_id: WebSocket connection ID
            
        Returns:
            True if heartbeat updated
            
        Raises:
            ProgressTrackingError: If heartbeat update fails
        """
        try:
            if connection_id not in self.active_connections:
                raise ResourceNotFoundError(f"WebSocket connection not found: {connection_id}")
            
            connection = self.active_connections[connection_id]
            connection.last_heartbeat = datetime.utcnow()
            
            # Update Redis
            key = f"{self.websocket_prefix}{connection_id}"
            connection_dict = asdict(connection)
            connection_dict['subscribed_stages'] = list(connection.subscribed_stages)
            
            await self.redis_client.setex(
                key,
                int(self.websocket_heartbeat_timeout * 2),
                json.dumps(connection_dict, default=str)
            )
            
            return True
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update WebSocket heartbeat: {e}")
            raise ProgressTrackingError(f"Heartbeat update failed: {str(e)}")
    
    async def cleanup_stale_connections(self) -> int:
        """
        Clean up stale WebSocket connections.
        
        Returns:
            Number of cleaned up connections
            
        Raises:
            ProgressTrackingError: If cleanup fails
        """
        try:
            stale_count = 0
            current_time = datetime.utcnow()
            
            # Check active connections
            connections_to_remove = []
            for connection_id, connection in self.active_connections.items():
                time_since_heartbeat = (current_time - connection.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.websocket_heartbeat_timeout:
                    connections_to_remove.append(connection_id)
            
            # Remove stale connections
            for connection_id in connections_to_remove:
                await self.unregister_websocket_connection(connection_id)
                stale_count += 1
            
            # Check Redis for stale connections (in case of service restart)
            pattern = f"{self.websocket_prefix}*"
            redis_keys = await self.redis_client.keys(pattern)
            
            for key in redis_keys:
                connection_data = await self.redis_client.get(key)
                if connection_data:
                    connection_dict = json.loads(connection_data)
                    last_heartbeat = datetime.fromisoformat(connection_dict['last_heartbeat'])
                    time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.websocket_heartbeat_timeout:
                        connection_id = connection_dict['connection_id']
                        await self.redis_client.delete(key)
                        stale_count += 1
            
            if stale_count > 0:
                self.logger.info(f"Cleaned up {stale_count} stale WebSocket connections")
            
            return stale_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup stale connections: {e}")
            raise ProgressTrackingError(f"Stale connection cleanup failed: {str(e)}")
    
    async def _publish_progress_event(self, execution_id: str, stage_name: str,
                                    status: str, progress_percent: float,
                                    message: str, overall_progress: Optional[float] = None) -> None:
        """Publish progress update event."""
        try:
            event = EventMessage(
                event_type=EventType.PROGRESS_UPDATE.value,
                execution_id=execution_id,
                stage=stage_name,
                payload={
                    'status': status,
                    'progress_percent': progress_percent,
                    'message': message,
                    'overall_progress': overall_progress,
                    'timestamp': datetime.utcnow().isoformat()
                },
                source='progress_tracker',
                target='all_services'
            )
            await self.event_bus.publish_event(event)
            
        except Exception as e:
            self.logger.error(f"Failed to publish progress event: {e}")
    
    async def _publish_operation_event(self, execution_id: str, stage_name: str,
                                     operation_name: str, operation_progress: float,
                                     operation_message: str) -> None:
        """Publish operation progress event."""
        try:
            event = EventMessage(
                event_type=EventType.OPERATION_PROGRESS.value,
                execution_id=execution_id,
                stage=stage_name,
                payload={
                    'operation_name': operation_name,
                    'operation_progress': operation_progress,
                    'operation_message': operation_message,
                    'timestamp': datetime.utcnow().isoformat()
                },
                source='progress_tracker',
                target='all_services'
            )
            await self.event_bus.publish_event(event)
            
        except Exception as e:
            self.logger.error(f"Failed to publish operation event: {e}")
    
    async def _notify_subscribers(self, execution_id: str, stage_name: str,
                                progress_config: Dict[str, Any]) -> None:
        """Notify WebSocket subscribers of progress updates."""
        try:
            if execution_id not in self.active_executions:
                return
            
            # Get subscribers for this execution
            subscribers = self.active_executions[execution_id].get('subscribers', set())
            
            if not subscribers:
                return
            
            # Create progress update message
            progress_update = {
                'type': 'progress_update',
                'execution_id': execution_id,
                'stage_name': stage_name,
                'stage_progress': progress_config['stage_progress'][stage_name],
                'overall_progress': progress_config['overall_progress'],
                'current_stage': progress_config['current_stage'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to all subscribers (WebSocket implementation would go here)
            # This is a placeholder for actual WebSocket message sending
            for connection_id in subscribers:
                if connection_id in self.active_connections:
                    self.logger.debug(
                        f"Sending progress update to connection {connection_id}: "
                        f"{progress_update['overall_progress']:.1f}%"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to notify subscribers: {e}")
    
    def _calculate_overall_progress(self, stage_progress: Dict[str, Any]) -> float:
        """Calculate overall progress from stage progress."""
        try:
            total_weight = 0
            weighted_progress = 0
            
            for stage_name, stage_info in stage_progress.items():
                if stage_name in self.stage_progress_mapping:
                    stage_range = self.stage_progress_mapping[stage_name]
                    stage_weight = stage_range['end'] - stage_range['start']
                    
                    # Use actual progress if available, otherwise use status-based estimate
                    if stage_info['progress_percent'] > 0:
                        stage_contribution = stage_info['progress_percent'] / 100.0
                    else:
                        # Estimate based on status
                        if stage_info['status'] == 'completed':
                            stage_contribution = 1.0
                        elif stage_info['status'] == 'started':
                            stage_contribution = 0.5
                        else:
                            stage_contribution = 0.0
                    
                    weighted_progress += stage_weight * stage_contribution
                    total_weight += stage_weight
            
            if total_weight > 0:
                return min(100.0, weighted_progress)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating overall progress: {e}")
            return 0.0
    
    def _calculate_completion_estimates(self, progress_config: Dict[str, Any], 
                                      stage_states: List[Any]) -> Dict[str, Any]:
        """Calculate completion time estimates."""
        try:
            current_time = datetime.utcnow()
            start_time = datetime.fromisoformat(progress_config['start_time'])
            elapsed_seconds = (current_time - start_time).total_seconds()
            
            overall_progress = progress_config['overall_progress'] / 100.0
            
            if overall_progress > 0:
                estimated_total_seconds = elapsed_seconds / overall_progress
                remaining_seconds = estimated_total_seconds - elapsed_seconds
                
                estimated_completion = current_time + timedelta(seconds=remaining_seconds)
                
                return {
                    'estimated_completion': estimated_completion.isoformat(),
                    'estimated_remaining_seconds': remaining_seconds,
                    'estimated_total_duration_seconds': estimated_total_seconds
                }
            else:
                return {
                    'estimated_completion': None,
                    'estimated_remaining_seconds': None,
                    'estimated_total_duration_seconds': None
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating completion estimates: {e}")
            return {
                'estimated_completion': None,
                'estimated_remaining_seconds': None,
                'estimated_total_duration_seconds': None
            }
    
    def _estimate_total_duration(self, dataset_count: int, execution_mode: str) -> float:
        """Estimate total pipeline duration in seconds."""
        try:
            # Base duration estimates per stage (in seconds)
            base_durations = {
                'standard': {
                    'ingestion': 30,
                    'validation': 120,
                    'processing': 300,
                    'standardization': 60,
                    'quality': 90,
                    'export': 30
                },
                'fast': {
                    'ingestion': 15,
                    'validation': 60,
                    'processing': 150,
                    'standardization': 30,
                    'quality': 45,
                    'export': 15
                },
                'comprehensive': {
                    'ingestion': 60,
                    'validation': 240,
                    'processing': 600,
                    'standardization': 120,
                    'quality': 180,
                    'export': 60
                }
            }
            
            mode_durations = base_durations.get(execution_mode, base_durations['standard'])
            base_total = sum(mode_durations.values())
            
            # Adjust for dataset count (linear scaling up to a point)
            dataset_factor = min(2.0, 1.0 + (dataset_count - 1) * 0.1)
            
            return base_total * dataset_factor
            
        except Exception as e:
            self.logger.error(f"Error estimating total duration: {e}")
            return 600.0  # Default 10 minutes
    
    def _get_stage_result_summary(self, stage_state: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Get summary of stage execution results."""
        if not stage_state or not stage_state.result:
            return None
        
        try:
            # Extract key metrics from stage result
            result = stage_state.result
            
            summary = {
                'status': stage_state.status,
                'duration_seconds': stage_state.duration_seconds
            }
            
            # Add stage-specific metrics
            if stage_state.stage_name == 'validation':
                summary['validation_score'] = result.get('validation_score')
                summary['checks_passed'] = result.get('checks_passed')
            elif stage_state.stage_name == 'processing':
                summary['processed_records'] = result.get('processed_records')
                summary['processing_time_seconds'] = result.get('processing_time_seconds')
            elif stage_state.stage_name == 'quality':
                summary['quality_score'] = result.get('quality_score')
                summary['assessment_results'] = result.get('assessment_results')
            elif stage_state.stage_name == 'export':
                summary['output_files'] = result.get('output_files', [])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting stage result summary: {e}")
            return None
    
    def _extract_quality_scores(self, stage_states: List[Any]) -> Dict[str, float]:
        """Extract quality scores from stage states."""
        quality_scores = {}
        
        try:
            for stage_state in stage_states:
                if stage_state.result and 'quality_score' in stage_state.result:
                    quality_scores[stage_state.stage_name] = stage_state.result['quality_score']
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Error extracting quality scores: {e}")
            return {}
    
    def _calculate_performance_metrics(self, progress_config: Dict[str, Any], 
                                     stage_states: List[Any]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            metrics = {
                'stages_completed': 0,
                'stages_failed': 0,
                'total_stage_duration_seconds': 0,
                'average_stage_duration_seconds': 0,
                'current_stage_duration_seconds': 0
            }
            
            current_time = datetime.utcnow()
            start_time = datetime.fromisoformat(progress_config['start_time'])
            
            for stage_state in stage_states:
                if stage_state.status == 'completed':
                    metrics['stages_completed'] += 1
                    if stage_state.duration_seconds:
                        metrics['total_stage_duration_seconds'] += stage_state.duration_seconds
                elif stage_state.status == 'failed':
                    metrics['stages_failed'] += 1
                
                # Current stage duration
                if (stage_state.stage_name == progress_config['current_stage'] and 
                    stage_state.start_time and not stage_state.end_time):
                    stage_start = datetime.fromisoformat(stage_state.start_time)
                    metrics['current_stage_duration_seconds'] = (
                        current_time - stage_start
                    ).total_seconds()
            
            # Calculate averages
            if metrics['stages_completed'] > 0:
                metrics['average_stage_duration_seconds'] = (
                    metrics['total_stage_duration_seconds'] / metrics['stages_completed']
                )
            
            # Add overall metrics
            metrics['total_elapsed_seconds'] = (current_time - start_time).total_seconds()
            metrics['overall_progress_percent'] = progress_config['overall_progress']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of progress tracker.
        
        Returns:
            Health check results
        """
        try:
            # Test Redis connection
            redis_health = await self.redis_client.ping()
            
            # Check active connections
            active_connections = len(self.active_connections)
            active_executions = len(self.active_executions)
            
            # Test progress retrieval
            test_key = f"{self.progress_prefix}health_check"
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
                'active_connections': active_connections,
                'active_executions': active_executions,
                'real_time_updates_enabled': self.enable_real_time_updates,
                'websocket_heartbeat_timeout': self.websocket_heartbeat_timeout,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ProgressTracker health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }