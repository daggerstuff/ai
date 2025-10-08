"""
WebSocket progress tracking for TechDeck-Python Pipeline Integration.

This module implements real-time progress tracking using WebSocket connections,
providing live updates for pipeline execution, file uploads, and system operations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, asdict
from enum import Enum

from flask import request, g
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from flask_jwt_extended import jwt_required, get_jwt_identity

from ..utils.logger import get_request_logger
from ..error_handling.custom_errors import ValidationError, WebSocketError
from ..integration.redis_client import RedisClient


class ProgressStatus(Enum):
    """Progress status enumeration."""
    PENDING = "pending"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressType(Enum):
    """Progress type enumeration."""
    PIPELINE_EXECUTION = "pipeline_execution"
    FILE_UPLOAD = "file_upload"
    DATA_VALIDATION = "data_validation"
    BIAS_DETECTION = "bias_detection"
    SYSTEM_OPERATION = "system_operation"


@dataclass
class ProgressEvent:
    """Progress event data structure."""
    event_id: str
    user_id: str
    operation_type: str
    status: str
    progress_percent: float
    current_step: str
    total_steps: int
    current_step_number: int
    message: str
    details: Dict[str, Any]
    timestamp: str
    estimated_time_remaining: Optional[int] = None


class ProgressTracker:
    """WebSocket-based progress tracking manager."""
    
    def __init__(self, socketio: SocketIO, redis_client: RedisClient):
        self.socketio = socketio
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.active_connections: Set[str] = set()
        self.user_rooms: Dict[str, str] = {}
        
    def initialize_handlers(self):
        """Initialize WebSocket event handlers."""
        self.logger.info("Initializing WebSocket progress tracking handlers")
        
        @self.socketio.on('connect')
        @jwt_required()
        def handle_connect():
            """Handle client connection."""
            try:
                user_id = get_jwt_identity()
                request_id = getattr(g, 'request_id', 'unknown')
                
                self.logger.info(f"WebSocket connection established for user {user_id}")
                
                # Join user-specific room
                user_room = f"user_{user_id}"
                join_room(user_room)
                
                # Track connection
                self.active_connections.add(request.sid)
                self.user_rooms[request.sid] = user_room
                
                # Send connection confirmation
                emit('connected', {
                    'success': True,
                    'data': {
                        'user_id': user_id,
                        'connection_id': request.sid,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                })
                
                # Send any pending progress updates
                self._send_pending_updates(user_id)
                
            except Exception as e:
                self.logger.error(f"Error handling WebSocket connection: {e}")
                emit('connection_error', {
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            try:
                request_id = getattr(g, 'request_id', 'unknown')
                
                self.logger.info(f"WebSocket connection closed for session {request.sid}")
                
                # Remove from tracking
                self.active_connections.discard(request.sid)
                user_room = self.user_rooms.pop(request.sid, None)
                
                if user_room:
                    leave_room(user_room)
                
            except Exception as e:
                self.logger.error(f"Error handling WebSocket disconnection: {e}")
        
        @self.socketio.on('subscribe_progress')
        @jwt_required()
        def handle_subscribe_progress(data):
            """Subscribe to progress updates for specific operations."""
            try:
                user_id = get_jwt_identity()
                operation_id = data.get('operation_id')
                
                if not operation_id:
                    raise ValidationError("operation_id is required")
                
                # Join operation-specific room
                operation_room = f"operation_{operation_id}"
                join_room(operation_room)
                
                self.logger.info(f"User {user_id} subscribed to progress for operation {operation_id}")
                
                # Send current progress status
                current_progress = self._get_current_progress(operation_id)
                
                emit('progress_subscribed', {
                    'success': True,
                    'data': {
                        'operation_id': operation_id,
                        'current_progress': current_progress,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                })
                
            except ValidationError as e:
                emit('subscription_error', {
                    'success': False,
                    'error': str(e)
                })
            except Exception as e:
                self.logger.error(f"Error handling progress subscription: {e}")
                emit('subscription_error', {
                    'success': False,
                    'error': 'Internal server error'
                })
        
        @self.socketio.on('unsubscribe_progress')
        @jwt_required()
        def handle_unsubscribe_progress(data):
            """Unsubscribe from progress updates."""
            try:
                user_id = get_jwt_identity()
                operation_id = data.get('operation_id')
                
                if not operation_id:
                    raise ValidationError("operation_id is required")
                
                # Leave operation-specific room
                operation_room = f"operation_{operation_id}"
                leave_room(operation_room)
                
                self.logger.info(f"User {user_id} unsubscribed from progress for operation {operation_id}")
                
                emit('progress_unsubscribed', {
                    'success': True,
                    'data': {
                        'operation_id': operation_id,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                })
                
            except ValidationError as e:
                emit('unsubscription_error', {
                    'success': False,
                    'error': str(e)
                })
            except Exception as e:
                self.logger.error(f"Error handling progress unsubscription: {e}")
                emit('unsubscription_error', {
                    'success': False,
                    'error': 'Internal server error'
                })
        
        @self.socketio.on('cancel_operation')
        @jwt_required()
        def handle_cancel_operation(data):
            """Cancel an ongoing operation."""
            try:
                user_id = get_jwt_identity()
                operation_id = data.get('operation_id')
                
                if not operation_id:
                    raise ValidationError("operation_id is required")
                
                # Verify operation ownership
                if not self._verify_operation_ownership(operation_id, user_id):
                    raise ValidationError("Operation not found or access denied")
                
                # Cancel operation
                self.cancel_operation(operation_id, user_id)
                
                self.logger.info(f"User {user_id} cancelled operation {operation_id}")
                
                emit('operation_cancelled', {
                    'success': True,
                    'data': {
                        'operation_id': operation_id,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                })
                
            except ValidationError as e:
                emit('cancellation_error', {
                    'success': False,
                    'error': str(e)
                })
            except Exception as e:
                self.logger.error(f"Error handling operation cancellation: {e}")
                emit('cancellation_error', {
                    'success': False,
                    'error': 'Internal server error'
                })
    
    def start_progress_tracking(self, operation_id: str, user_id: str, operation_type: str, total_steps: int = 1):
        """Start progress tracking for an operation."""
        try:
            progress_key = f"progress:{operation_id}"
            
            # Initialize progress in Redis
            progress_data = {
                'operation_id': operation_id,
                'user_id': user_id,
                'operation_type': operation_type,
                'status': ProgressStatus.STARTED.value,
                'progress_percent': 0.0,
                'current_step': 'Initializing',
                'total_steps': total_steps,
                'current_step_number': 0,
                'message': 'Operation started',
                'details': {},
                'timestamp': datetime.utcnow().isoformat(),
                'estimated_time_remaining': None
            }
            
            self.redis_client.set_json(progress_key, progress_data, ex=3600)  # 1 hour TTL
            
            # Create progress event
            event = ProgressEvent(
                event_id=f"{operation_id}_start",
                user_id=user_id,
                operation_type=operation_type,
                status=ProgressStatus.STARTED.value,
                progress_percent=0.0,
                current_step='Initializing',
                total_steps=total_steps,
                current_step_number=0,
                message='Operation started',
                details={},
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Broadcast progress update
            self._broadcast_progress(operation_id, event)
            
            self.logger.info(f"Started progress tracking for operation {operation_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting progress tracking: {e}")
            raise WebSocketError(f"Failed to start progress tracking: {str(e)}")
    
    def update_progress(self, operation_id: str, user_id: str, status: str, progress_percent: float,
                       current_step: str, message: str, details: Optional[Dict[str, Any]] = None,
                       estimated_time_remaining: Optional[int] = None):
        """Update progress for an operation."""
        try:
            progress_key = f"progress:{operation_id}"
            
            # Get current progress
            current_progress = self.redis_client.get_json(progress_key)
            if not current_progress:
                raise ValidationError(f"Progress tracking not found for operation {operation_id}")
            
            # Verify ownership
            if current_progress['user_id'] != user_id:
                raise ValidationError("Access denied to operation progress")
            
            # Update progress data
            current_progress.update({
                'status': status,
                'progress_percent': min(100.0, max(0.0, progress_percent)),
                'current_step': current_step,
                'message': message,
                'details': details or {},
                'timestamp': datetime.utcnow().isoformat(),
                'estimated_time_remaining': estimated_time_remaining
            })
            
            # Update step number if step changed
            if current_step != current_progress.get('current_step'):
                current_progress['current_step_number'] = current_progress.get('current_step_number', 0) + 1
            
            # Save updated progress
            self.redis_client.set_json(progress_key, current_progress, ex=3600)
            
            # Create progress event
            event = ProgressEvent(
                event_id=f"{operation_id}_update_{int(time.time() * 1000)}",
                user_id=user_id,
                operation_type=current_progress['operation_type'],
                status=status,
                progress_percent=current_progress['progress_percent'],
                current_step=current_step,
                total_steps=current_progress['total_steps'],
                current_step_number=current_progress['current_step_number'],
                message=message,
                details=details or {},
                timestamp=datetime.utcnow().isoformat(),
                estimated_time_remaining=estimated_time_remaining
            )
            
            # Broadcast progress update
            self._broadcast_progress(operation_id, event)
            
            self.logger.debug(f"Updated progress for operation {operation_id}: {progress_percent}% - {message}")
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")
            raise WebSocketError(f"Failed to update progress: {str(e)}")
    
    def complete_operation(self, operation_id: str, user_id: str, message: str = "Operation completed successfully",
                          details: Optional[Dict[str, Any]] = None):
        """Mark an operation as completed."""
        try:
            self.update_progress(
                operation_id=operation_id,
                user_id=user_id,
                status=ProgressStatus.COMPLETED.value,
                progress_percent=100.0,
                current_step="Completed",
                message=message,
                details=details or {}
            )
            
            self.logger.info(f"Operation {operation_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error completing operation: {e}")
            raise WebSocketError(f"Failed to complete operation: {str(e)}")
    
    def fail_operation(self, operation_id: str, user_id: str, error_message: str,
                      error_details: Optional[Dict[str, Any]] = None):
        """Mark an operation as failed."""
        try:
            self.update_progress(
                operation_id=operation_id,
                user_id=user_id,
                status=ProgressStatus.FAILED.value,
                progress_percent=0.0,
                current_step="Failed",
                message=error_message,
                details=error_details or {}
            )
            
            self.logger.error(f"Operation {operation_id} failed: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error failing operation: {e}")
            raise WebSocketError(f"Failed to fail operation: {str(e)}")
    
    def cancel_operation(self, operation_id: str, user_id: str, message: str = "Operation cancelled"):
        """Cancel an ongoing operation."""
        try:
            self.update_progress(
                operation_id=operation_id,
                user_id=user_id,
                status=ProgressStatus.CANCELLED.value,
                progress_percent=0.0,
                current_step="Cancelled",
                message=message
            )
            
            self.logger.info(f"Operation {operation_id} cancelled by user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling operation: {e}")
            raise WebSocketError(f"Failed to cancel operation: {str(e)}")
    
    def get_progress(self, operation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for an operation."""
        try:
            progress_key = f"progress:{operation_id}"
            progress_data = self.redis_client.get_json(progress_key)
            
            if not progress_data:
                return None
            
            # Verify ownership
            if progress_data['user_id'] != user_id:
                raise ValidationError("Access denied to operation progress")
            
            return progress_data
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting progress: {e}")
            raise WebSocketError(f"Failed to get progress: {str(e)}")
    
    def get_user_operations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operations for a user."""
        try:
            # Get operation IDs from Redis (using pattern matching)
            pattern = f"progress:*"
            keys = self.redis_client.keys(pattern)
            
            user_operations = []
            
            for key in keys[:limit]:  # Limit to prevent excessive queries
                progress_data = self.redis_client.get_json(key)
                if progress_data and progress_data['user_id'] == user_id:
                    user_operations.append({
                        'operation_id': progress_data['operation_id'],
                        'operation_type': progress_data['operation_type'],
                        'status': progress_data['status'],
                        'progress_percent': progress_data['progress_percent'],
                        'current_step': progress_data['current_step'],
                        'timestamp': progress_data['timestamp']
                    })
            
            # Sort by timestamp (newest first)
            user_operations.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return user_operations
            
        except Exception as e:
            self.logger.error(f"Error getting user operations: {e}")
            raise WebSocketError(f"Failed to get user operations: {str(e)}")
    
    def _broadcast_progress(self, operation_id: str, event: ProgressEvent):
        """Broadcast progress update to connected clients."""
        try:
            operation_room = f"operation_{operation_id}"
            user_room = f"user_{event.user_id}"
            
            # Convert event to dictionary
            event_data = asdict(event)
            
            # Emit to operation room
            self.socketio.emit('progress_update', {
                'success': True,
                'data': event_data,
                'timestamp': datetime.utcnow().isoformat()
            }, room=operation_room)
            
            # Also emit to user room for general updates
            self.socketio.emit('user_progress_update', {
                'success': True,
                'data': event_data,
                'timestamp': datetime.utcnow().isoformat()
            }, room=user_room)
            
            self.logger.debug(f"Broadcasted progress update for operation {operation_id}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting progress: {e}")
    
    def _send_pending_updates(self, user_id: str):
        """Send any pending progress updates to user."""
        try:
            # Get user's room
            user_room = f"user_{user_id}"
            
            # Get recent operations for user
            operations = self.get_user_operations(user_id, limit=5)
            
            if operations:
                # Emit current operations status
                self.socketio.emit('current_operations', {
                    'success': True,
                    'data': {
                        'operations': operations,
                        'count': len(operations)
                    }
                }, room=user_room)
                
                self.logger.debug(f"Sent pending updates to user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Error sending pending updates: {e}")
    
    def _get_current_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for an operation."""
        try:
            progress_key = f"progress:{operation_id}"
            return self.redis_client.get_json(progress_key)
        except Exception as e:
            self.logger.error(f"Error getting current progress: {e}")
            return None
    
    def _verify_operation_ownership(self, operation_id: str, user_id: str) -> bool:
        """Verify that a user owns an operation."""
        try:
            progress_data = self.get_progress(operation_id, user_id)
            return progress_data is not None
        except ValidationError:
            return False
        except Exception as e:
            self.logger.error(f"Error verifying operation ownership: {e}")
            return False


# Convenience functions for different operation types
def track_pipeline_execution(progress_tracker: ProgressTracker, operation_id: str, user_id: str,
                           pipeline_config: Dict[str, Any]):
    """Start tracking pipeline execution progress."""
    total_steps = 6  # Six-stage pipeline
    progress_tracker.start_progress_tracking(
        operation_id=operation_id,
        user_id=user_id,
        operation_type=ProgressType.PIPELINE_EXECUTION.value,
        total_steps=total_steps
    )


def track_file_upload(progress_tracker: ProgressTracker, operation_id: str, user_id: str,
                     file_info: Dict[str, Any]):
    """Start tracking file upload progress."""
    total_steps = 3  # Upload, validation, processing
    progress_tracker.start_progress_tracking(
        operation_id=operation_id,
        user_id=user_id,
        operation_type=ProgressType.FILE_UPLOAD.value,
        total_steps=total_steps
    )


def track_data_validation(progress_tracker: ProgressTracker, operation_id: str, user_id: str,
                         validation_config: Dict[str, Any]):
    """Start tracking data validation progress."""
    total_steps = 4  # Schema check, quality check, bias check, compliance check
    progress_tracker.start_progress_tracking(
        operation_id=operation_id,
        user_id=user_id,
        operation_type=ProgressType.DATA_VALIDATION.value,
        total_steps=total_steps
    )


def track_bias_detection(progress_tracker: ProgressTracker, operation_id: str, user_id: str,
                        detection_config: Dict[str, Any]):
    """Start tracking bias detection progress."""
    total_steps = 3  # Preprocessing, detection, analysis
    progress_tracker.start_progress_tracking(
        operation_id=operation_id,
        user_id=user_id,
        operation_type=ProgressType.BIAS_DETECTION.value,
        total_steps=total_steps
    )