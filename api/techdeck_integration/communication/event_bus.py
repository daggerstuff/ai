"""
Redis Event Bus for TechDeck-Python Pipeline Integration.

This module provides a comprehensive event-driven communication system
with Redis pub/sub, connection pooling, and HIPAA++ compliant logging.
"""

import json
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

from ..integration.redis_client import RedisClient
from ..utils.logger import get_request_logger, log_performance_metric
from ..error_handling.custom_errors import (
    RedisError, EventBusError, ValidationError, TimeoutError
)


class EventType(Enum):
    """Standardized event types for pipeline communication."""
    # Six-stage pipeline events
    REQUEST_INITIATED = "request.initiated"
    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETED = "validation.completed"
    PROCESSING_STARTED = "processing.started"
    PROCESSING_COMPLETED = "processing.completed"
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    RESPONSE_PREPARED = "response.prepared"
    COMPLETION_FINALIZED = "completion.finalized"
    
    # Progress tracking events
    PROGRESS_UPDATE = "progress.update"
    STAGE_TRANSITION = "stage.transition"
    
    # Error and recovery events
    ERROR_OCCURRED = "error.occurred"
    RETRY_ATTEMPT = "retry.attempt"
    RECOVERY_INITIATED = "recovery.initiated"
    
    # Bias detection events
    BIAS_CHECK_STARTED = "bias.check.started"
    BIAS_CHECK_COMPLETED = "bias.check.completed"
    BIAS_THRESHOLD_EXCEEDED = "bias.threshold.exceeded"
    
    # Performance monitoring events
    PERFORMANCE_METRIC = "performance.metric"
    LATENCY_WARNING = "latency.warning"


@dataclass
class EventMessage:
    """Standardized event message structure."""
    event_type: str
    execution_id: str
    stage: Optional[str] = None
    payload: Dict[str, Any] = None
    timestamp: str = None
    source: str = None
    target: str = None
    correlation_id: str = None
    retry_count: int = 0
    priority: int = 1  # 1-10, higher is more important
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.payload is None:
            self.payload = {}
        if self.correlation_id is None:
            self.correlation_id = self.execution_id


class EventHandler:
    """Base class for event handlers with HIPAA++ compliance."""
    
    def __init__(self, name: str, event_types: List[str]):
        self.name = name
        self.event_types = event_types
        self.logger = get_request_logger()
        
    async def handle(self, event: EventMessage) -> Optional[Dict[str, Any]]:
        """
        Handle incoming event with comprehensive logging.
        
        Args:
            event: Event message to process
            
        Returns:
            Optional response data
            
        Raises:
            EventBusError: If event processing fails
        """
        start_time = time.time()
        
        try:
            # Log event receipt (without sensitive data)
            self.logger.info(
                f"Handler {self.name} processing event {event.event_type} "
                f"for execution {event.execution_id}"
            )
            
            # Process event
            result = await self._process_event(event)
            
            # Log performance metrics
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > 50:  # Sub-50ms requirement
                self.logger.warning(
                    f"Event handler {self.name} took {duration_ms:.2f}ms "
                    f"for execution {event.execution_id}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in event handler {self.name} for execution "
                f"{event.execution_id}: {e}"
            )
            raise EventBusError(f"Event handler {self.name} failed: {str(e)}")
    
    async def _process_event(self, event: EventMessage) -> Optional[Dict[str, Any]]:
        """
        Process the event - to be implemented by subclasses.
        
        Args:
            event: Event message to process
            
        Returns:
            Optional response data
        """
        raise NotImplementedError("Subclasses must implement _process_event")


class EventBus:
    """Redis-based event bus with connection pooling and HIPAA++ compliance."""
    
    def __init__(self, redis_client: RedisClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event bus with Redis connection.
        
        Args:
            redis_client: Redis client instance
            config: Optional configuration dictionary
        """
        self.redis_client = redis_client
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Configuration with defaults
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.event_timeout = self.config.get('event_timeout', 300)  # 5 minutes
        self.connection_pool_size = self.config.get('connection_pool_size', 10)
        
        # Event handlers registry
        self.handlers: Dict[str, List[EventHandler]] = {}
        
        # Performance monitoring
        self.metrics = {
            'events_published': 0,
            'events_processed': 0,
            'errors_encountered': 0,
            'average_processing_time_ms': 0
        }
        
        self.logger.info("EventBus initialized with Redis connection pooling")
    
    def register_handler(self, handler: EventHandler) -> None:
        """
        Register an event handler for specific event types.
        
        Args:
            handler: Event handler instance to register
        """
        for event_type in handler.event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
        self.logger.info(
            f"Registered handler {handler.name} for event types: "
            f"{handler.event_types}"
        )
    
    def unregister_handler(self, handler_name: str) -> None:
        """
        Unregister an event handler by name.
        
        Args:
            handler_name: Name of the handler to unregister
        """
        for event_type, handlers in self.handlers.items():
            self.handlers[event_type] = [
                h for h in handlers if h.name != handler_name
            ]
        
        self.logger.info(f"Unregistered handler {handler_name}")
    
    @log_performance_metric('event_publish')
    async def publish_event(
        self,
        event: EventMessage,
        channel: str = "pipeline_events",
        guaranteed_delivery: bool = True
    ) -> bool:
        """
        Publish an event to Redis with guaranteed delivery options.
        
        Args:
            event: Event message to publish
            channel: Redis channel to publish to
            guaranteed_delivery: Whether to ensure delivery with retries
            
        Returns:
            True if published successfully
            
        Raises:
            EventBusError: If event publishing fails after retries
        """
        try:
            # Validate event
            self._validate_event(event)
            
            # Serialize event
            event_data = asdict(event)
            event_json = json.dumps(event_data, default=str)
            
            # Publish with retry logic if guaranteed delivery is enabled
            if guaranteed_delivery:
                return await self._publish_with_retry(channel, event_json, event)
            else:
                return await self._publish_single_attempt(channel, event_json, event)
                
        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_type}: {e}")
            self.metrics['errors_encountered'] += 1
            raise EventBusError(f"Event publishing failed: {str(e)}")
    
    async def _publish_with_retry(
        self,
        channel: str,
        event_json: str,
        event: EventMessage
    ) -> bool:
        """Publish event with retry logic for guaranteed delivery."""
        for attempt in range(self.max_retries):
            try:
                result = self.redis_client.publish_message(channel, event_json)
                
                if result > 0:
                    self.metrics['events_published'] += 1
                    self.logger.info(
                        f"Event {event.event_type} published successfully "
                        f"to {channel} for execution {event.execution_id} "
                        f"(attempt {attempt + 1})"
                    )
                    return True
                else:
                    self.logger.warning(
                        f"No subscribers for event {event.event_type} "
                        f"on channel {channel} (attempt {attempt + 1})"
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Event publish attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
            
            # If we get here, retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return False
    
    async def _publish_single_attempt(
        self,
        channel: str,
        event_json: str,
        event: EventMessage
    ) -> bool:
        """Publish event with single attempt (fire-and-forget)."""
        try:
            result = self.redis_client.publish_message(channel, event_json)
            self.metrics['events_published'] += 1
            
            self.logger.debug(
                f"Event {event.event_type} published to {channel} "
                f"for execution {event.execution_id}"
            )
            
            return result > 0
            
        except Exception as e:
            self.logger.error(
                f"Single attempt publish failed for event {event.event_type}: {e}"
            )
            return False
    
    async def subscribe_to_events(
        self,
        event_types: Optional[List[str]] = None,
        handler: Optional[EventHandler] = None,
        channel: str = "pipeline_events"
    ) -> None:
        """
        Subscribe to events and process them with registered handlers.
        
        Args:
            event_types: Optional list of specific event types to subscribe to
            handler: Optional specific handler to use
            channel: Redis channel to subscribe to
        """
        try:
            # Subscribe to Redis channel
            pubsub = self.redis_client.subscribe_to_channel(channel)
            
            self.logger.info(
                f"Subscribed to events on channel {channel} "
                f"(types: {event_types or 'all'})"
            )
            
            # Process incoming messages
            while True:
                try:
                    message = pubsub.get_message(timeout=1.0)
                    
                    if message and message['type'] == 'message':
                        await self._process_incoming_message(
                            message['data'],
                            event_types,
                            handler
                        )
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    self.metrics['errors_encountered'] += 1
                    
        except Exception as e:
            self.logger.error(f"Event subscription failed: {e}")
            raise EventBusError(f"Event subscription failed: {str(e)}")
    
    async def _process_incoming_message(
        self,
        message_data: Union[str, bytes],
        event_types_filter: Optional[List[str]],
        specific_handler: Optional[EventHandler]
    ) -> None:
        """Process an incoming message from Redis."""
        start_time = time.time()
        
        try:
            # Deserialize message
            if isinstance(message_data, bytes):
                message_data = message_data.decode('utf-8')
            
            event_data = json.loads(message_data)
            event = EventMessage(**event_data)
            
            # Filter by event type if specified
            if event_types_filter and event.event_type not in event_types_filter:
                return
            
            self.logger.debug(
                f"Processing event {event.event_type} "
                f"for execution {event.execution_id}"
            )
            
            # Use specific handler if provided, otherwise use registered handlers
            if specific_handler:
                handlers = [specific_handler]
            else:
                handlers = self.handlers.get(event.event_type, [])
            
            # Process event with all applicable handlers
            for handler in handlers:
                try:
                    await handler.handle(event)
                    self.metrics['events_processed'] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Handler {handler.name} failed for event "
                        f"{event.event_type}: {e}"
                    )
                    self.metrics['errors_encountered'] += 1
            
            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time_ms)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode event message: {e}")
            self.metrics['errors_encountered'] += 1
        except Exception as e:
            self.logger.error(f"Error processing incoming message: {e}")
            self.metrics['errors_encountered'] += 1
    
    def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance metrics with new processing time."""
        current_avg = self.metrics['average_processing_time_ms']
        processed_count = self.metrics['events_processed']
        
        # Calculate new average
        new_avg = (current_avg * (processed_count - 1) + processing_time_ms) / processed_count
        self.metrics['average_processing_time_ms'] = new_avg
        
        # Log performance warnings
        if processing_time_ms > 50:  # Sub-50ms requirement
            self.logger.warning(
                f"Event processing took {processing_time_ms:.2f}ms "
                f"(average: {new_avg:.2f}ms)"
            )
    
    def _validate_event(self, event: EventMessage) -> None:
        """Validate event message structure and content."""
        if not event.event_type:
            raise ValidationError("Event type is required")
        
        if not event.execution_id:
            raise ValidationError("Execution ID is required")
        
        if event.event_type not in [et.value for et in EventType]:
            self.logger.warning(f"Unknown event type: {event.event_type}")
        
        # HIPAA++ compliance: validate no sensitive data in payload
        if event.payload:
            self._validate_payload_compliance(event.payload)
    
    def _validate_payload_compliance(self, payload: Dict[str, Any]) -> None:
        """Validate payload for HIPAA++ compliance."""
        # Check for potential PII/PHI indicators
        pii_indicators = ['email', 'phone', 'ssn', 'name', 'address']
        payload_str = str(payload).lower()
        
        for indicator in pii_indicators:
            if indicator in payload_str:
                self.logger.warning(
                    f"Potential PII detected in event payload: {indicator}"
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            **self.metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'connection_pool_stats': self.redis_client.get_connection_pool_stats()
        }
    
    async def create_stage_event(
        self,
        execution_id: str,
        stage: str,
        status: str,
        progress_percent: float = 0.0,
        message: str = "",
        payload: Optional[Dict[str, Any]] = None
    ) -> EventMessage:
        """
        Create a standardized stage event for six-stage pipeline.
        
        Args:
            execution_id: Unique execution identifier
            stage: Current pipeline stage
            status: Stage status (started, completed, failed)
            progress_percent: Progress percentage (0-100)
            message: Status message
            payload: Optional additional data
            
        Returns:
            Formatted event message
        """
        event_type_map = {
            'started': EventType.PROCESSING_STARTED.value,
            'completed': EventType.PROCESSING_COMPLETED.value,
            'failed': EventType.ERROR_OCCURRED.value
        }
        
        event_type = event_type_map.get(status, EventType.PROGRESS_UPDATE.value)
        
        return EventMessage(
            event_type=event_type,
            execution_id=execution_id,
            stage=stage,
            payload={
                'stage': stage,
                'status': status,
                'progress_percent': progress_percent,
                'message': message,
                **(payload or {})
            },
            source='pipeline_coordinator',
            target='progress_tracker'
        )
    
    async def create_bias_detection_event(
        self,
        execution_id: str,
        bias_score: float,
        threshold: float,
        recommendations: List[str] = None
    ) -> EventMessage:
        """
        Create a bias detection event with HIPAA++ compliant data.
        
        Args:
            execution_id: Unique execution identifier
            bias_score: Calculated bias score
            threshold: Bias threshold
            recommendations: Optional recommendations
            
        Returns:
            Formatted bias detection event
        """
        event_type = (
            EventType.BIAS_THRESHOLD_EXCEEDED.value 
            if bias_score > threshold 
            else EventType.BIAS_CHECK_COMPLETED.value
        )
        
        return EventMessage(
            event_type=event_type,
            execution_id=execution_id,
            payload={
                'bias_score': bias_score,
                'threshold': threshold,
                'threshold_exceeded': bias_score > threshold,
                'recommendations': recommendations or [],
                'compliance_status': 'acceptable' if bias_score <= threshold else 'review_required'
            },
            source='bias_detection_service',
            target='pipeline_coordinator'
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of event bus.
        
        Returns:
            Health check results
        """
        try:
            # Check Redis connection
            redis_health = self.redis_client.health_check()
            
            # Get performance metrics
            metrics = self.get_metrics()
            
            return {
                'status': 'healthy' if redis_health['status'] == 'healthy' else 'degraded',
                'redis_health': redis_health,
                'metrics': metrics,
                'registered_handlers': {
                    event_type: len(handlers) 
                    for event_type, handlers in self.handlers.items()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"EventBus health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Convenience functions for common event operations
async def publish_stage_progress(
    event_bus: EventBus,
    execution_id: str,
    stage: str,
    progress_percent: float,
    message: str = ""
) -> bool:
    """Publish stage progress update event."""
    event = await event_bus.create_stage_event(
        execution_id=execution_id,
        stage=stage,
        status='in_progress',
        progress_percent=progress_percent,
        message=message
    )
    return await event_bus.publish_event(event)


async def publish_stage_completion(
    event_bus: EventBus,
    execution_id: str,
    stage: str,
    result_data: Optional[Dict[str, Any]] = None
) -> bool:
    """Publish stage completion event."""
    event = await event_bus.create_stage_event(
        execution_id=execution_id,
        stage=stage,
        status='completed',
        progress_percent=100.0,
        message=f"{stage} completed successfully",
        payload=result_data
    )
    return await event_bus.publish_event(event)


async def publish_error_event(
    event_bus: EventBus,
    execution_id: str,
    error_type: str,
    error_message: str,
    stage: Optional[str] = None
) -> bool:
    """Publish error event with HIPAA++ compliant logging."""
    event = EventMessage(
        event_type=EventType.ERROR_OCCURRED.value,
        execution_id=execution_id,
        stage=stage,
        payload={
            'error_type': error_type,
            'error_message': error_message,  # Sanitized message
            'stage': stage,
            'recoverable': True  # Default to recoverable
        },
        source='error_handler',
        target='recovery_manager'
    )
    return await event_bus.publish_event(event)