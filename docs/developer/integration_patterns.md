# Integration Patterns: Pixelated Empathy AI

**Common integration patterns and best practices for building applications with Pixelated Empathy AI.**

## Table of Contents

1. [Microservices Integration](#microservices-integration)
2. [Event-Driven Architecture](#event-driven-architecture)
3. [Batch Processing Patterns](#batch-processing-patterns)
4. [Real-Time Integration](#real-time-integration)
5. [ML Pipeline Integration](#ml-pipeline-integration)
6. [Enterprise Integration](#enterprise-integration)

---

## Microservices Integration

### Service-to-Service Communication

```python
# Conversation service with caching and circuit breaker
import asyncio
import aioredis
from circuit_breaker import CircuitBreaker

class ConversationService:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.cache = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=PixelatedEmpathyAPIError
        )
    
    async def initialize(self):
        """Initialize async components."""
        self.cache = await aioredis.from_url("redis://localhost:6379")
    
    @circuit_breaker
    async def get_conversation_with_fallback(self, conversation_id):
        """Get conversation with caching and fallback."""
        
        # Check cache first
        cache_key = f"conv:{conversation_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # Fetch from API with circuit breaker protection
        conversation = self.api.get_conversation(conversation_id)
        
        # Cache for 1 hour
        await self.cache.setex(cache_key, 3600, json.dumps(conversation))
        
        return conversation
    
    async def bulk_get_conversations(self, conversation_ids, batch_size=10):
        """Efficiently fetch multiple conversations."""
        
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(conversation_ids), batch_size):
            batch = conversation_ids[i:i + batch_size]
            
            # Create tasks for concurrent execution
            tasks = [
                self.get_conversation_with_fallback(conv_id)
                for conv_id in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = [
                result for result in batch_results
                if not isinstance(result, Exception)
            ]
            
            results.extend(successful_results)
            
            # Rate limiting between batches
            await asyncio.sleep(0.1)
        
        return results
```

### API Gateway Integration

```python
# API Gateway with request routing and load balancing
from fastapi import FastAPI, Request, HTTPException
import httpx
import random

class APIGateway:
    def __init__(self):
        self.app = FastAPI(title="Pixelated Empathy Gateway")
        self.backend_services = [
            "http://service-1:8000",
            "http://service-2:8000",
            "http://service-3:8000"
        ]
        self.health_status = {}
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.middleware("http")
        async def route_requests(request: Request, call_next):
            # Health check bypass
            if request.url.path == "/health":
                return await call_next(request)
            
            # Route to backend service
            backend = self.select_healthy_backend()
            if not backend:
                raise HTTPException(status_code=503, detail="No healthy backends")
            
            # Forward request
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=f"{backend}{request.url.path}",
                    headers=dict(request.headers),
                    content=await request.body()
                )
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
    
    def select_healthy_backend(self):
        """Select a healthy backend using weighted round-robin."""
        
        healthy_backends = [
            backend for backend, status in self.health_status.items()
            if status.get('healthy', True)
        ]
        
        if not healthy_backends:
            return None
        
        # Simple round-robin selection
        return random.choice(healthy_backends)
    
    async def health_check_backends(self):
        """Periodically check backend health."""
        
        while True:
            for backend in self.backend_services:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{backend}/health", timeout=5)
                        self.health_status[backend] = {
                            'healthy': response.status_code == 200,
                            'last_check': datetime.now()
                        }
                except Exception:
                    self.health_status[backend] = {
                        'healthy': False,
                        'last_check': datetime.now()
                    }
            
            await asyncio.sleep(30)  # Check every 30 seconds
```

---

## Event-Driven Architecture

### Event Publisher/Subscriber Pattern

```python
# Event-driven conversation processing
import asyncio
from typing import Dict, Any, Callable
import json

class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.event_queue = asyncio.Queue()
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        
        await self.event_queue.put(event)
    
    async def process_events(self):
        """Process events from the queue."""
        
        while True:
            event = await self.event_queue.get()
            event_type = event['type']
            
            if event_type in self.subscribers:
                # Process all subscribers concurrently
                tasks = [
                    handler(event) for handler in self.subscribers[event_type]
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

class ConversationEventProcessor:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.event_bus = EventBus()
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Set up event handlers for different event types."""
        
        self.event_bus.subscribe('conversation.created', self.handle_conversation_created)
        self.event_bus.subscribe('conversation.updated', self.handle_conversation_updated)
        self.event_bus.subscribe('quality.validation.requested', self.handle_quality_validation)
        self.event_bus.subscribe('export.requested', self.handle_export_request)
    
    async def handle_conversation_created(self, event):
        """Handle new conversation creation."""
        
        conversation_id = event['data']['conversation_id']
        
        # Trigger quality validation
        await self.event_bus.publish('quality.validation.requested', {
            'conversation_id': conversation_id,
            'priority': 'normal'
        })
        
        # Update statistics
        await self.update_conversation_statistics()
    
    async def handle_quality_validation(self, event):
        """Handle quality validation requests."""
        
        conversation_id = event['data']['conversation_id']
        
        try:
            # Get conversation data
            conversation = self.api.get_conversation(conversation_id)
            
            # Validate quality
            validation_result = self.api.validate_conversation_quality(conversation)
            
            # Publish validation completed event
            await self.event_bus.publish('quality.validation.completed', {
                'conversation_id': conversation_id,
                'validation_result': validation_result
            })
            
        except Exception as e:
            # Publish validation failed event
            await self.event_bus.publish('quality.validation.failed', {
                'conversation_id': conversation_id,
                'error': str(e)
            })
```

### Message Queue Integration

```python
# Celery integration for background processing
from celery import Celery
from kombu import Queue

# Celery configuration
celery_app = Celery(
    'pixelated_empathy_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure queues
celery_app.conf.task_routes = {
    'tasks.quality_validation': {'queue': 'quality'},
    'tasks.data_export': {'queue': 'export'},
    'tasks.analytics': {'queue': 'analytics'}
}

celery_app.conf.task_queues = (
    Queue('quality', routing_key='quality'),
    Queue('export', routing_key='export'),
    Queue('analytics', routing_key='analytics'),
)

@celery_app.task(bind=True, max_retries=3)
def validate_conversation_quality(self, conversation_data):
    """Background task for conversation quality validation."""
    
    try:
        api = PixelatedEmpathyAPI(os.getenv('PIXELATED_EMPATHY_API_KEY'))
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 25})
        
        # Validate conversation
        validation_result = api.validate_conversation_quality(conversation_data)
        
        self.update_state(state='PROGRESS', meta={'progress': 75})
        
        # Store results in database
        store_validation_results(conversation_data['id'], validation_result)
        
        self.update_state(state='PROGRESS', meta={'progress': 100})
        
        return {
            'status': 'completed',
            'conversation_id': conversation_data['id'],
            'validation_result': validation_result
        }
        
    except Exception as exc:
        # Retry with exponential backoff
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)

@celery_app.task(bind=True)
def export_dataset(self, dataset_name, export_format, filters):
    """Background task for dataset export."""
    
    try:
        api = PixelatedEmpathyAPI(os.getenv('PIXELATED_EMPATHY_API_KEY'))
        
        # Submit export job
        export_job = api.export_data(
            dataset=dataset_name,
            format=export_format,
            **filters
        )
        
        # Monitor export progress
        while True:
            job_status = api.get_job_status(export_job['job_id'])
            
            self.update_state(
                state='PROGRESS',
                meta={'progress': job_status.get('progress', 0)}
            )
            
            if job_status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(10)
        
        return {
            'status': job_status['status'],
            'export_id': export_job['export_id'],
            'download_url': job_status.get('download_url')
        }
        
    except Exception as exc:
        self.update_state(state='FAILURE', meta={'error': str(exc)})
        raise
```

---

## Batch Processing Patterns

### ETL Pipeline Integration

```python
# ETL pipeline for conversation data processing
import pandas as pd
from sqlalchemy import create_engine
import logging

class ConversationETLPipeline:
    def __init__(self, api_key, database_url):
        self.api = PixelatedEmpathyAPI(api_key)
        self.engine = create_engine(database_url)
        self.logger = logging.getLogger(__name__)
    
    async def extract_conversations(self, dataset_name, batch_size=1000):
        """Extract conversations from Pixelated Empathy AI."""
        
        self.logger.info(f"Starting extraction for dataset: {dataset_name}")
        
        conversations = []
        offset = 0
        
        while True:
            # Get batch of conversations
            batch = self.api.get_conversations(
                dataset=dataset_name,
                limit=batch_size,
                offset=offset
            )
            
            if not batch['conversations']:
                break
            
            # Get full conversation data
            for conv_summary in batch['conversations']:
                full_conversation = self.api.get_conversation(conv_summary['id'])
                conversations.append(full_conversation)
            
            offset += len(batch['conversations'])
            self.logger.info(f"Extracted {len(conversations)} conversations so far...")
            
            # Break if we got fewer results than requested
            if len(batch['conversations']) < batch_size:
                break
        
        self.logger.info(f"Extraction completed. Total: {len(conversations)} conversations")
        return conversations
    
    def transform_conversations(self, conversations):
        """Transform conversation data for analysis."""
        
        self.logger.info("Starting data transformation...")
        
        transformed_data = []
        
        for conv in conversations:
            # Extract conversation-level features
            conv_features = {
                'conversation_id': conv['id'],
                'dataset': conv['metadata'].get('dataset'),
                'tier': conv['metadata'].get('tier'),
                'message_count': len(conv['messages']),
                'total_length': sum(len(msg['content']) for msg in conv['messages']),
                'avg_message_length': sum(len(msg['content']) for msg in conv['messages']) / len(conv['messages']),
                'user_messages': len([msg for msg in conv['messages'] if msg['role'] == 'user']),
                'assistant_messages': len([msg for msg in conv['messages'] if msg['role'] == 'assistant']),
                'quality_score': conv['quality_metrics']['overall_quality'],
                'therapeutic_accuracy': conv['quality_metrics']['therapeutic_accuracy'],
                'safety_score': conv['quality_metrics']['safety_score']
            }
            
            # Extract message-level features
            for i, message in enumerate(conv['messages']):
                msg_features = {
                    'conversation_id': conv['id'],
                    'message_index': i,
                    'role': message['role'],
                    'content_length': len(message['content']),
                    'word_count': len(message['content'].split()),
                    'sentiment': self.analyze_sentiment(message['content']),
                    'empathy_score': self.calculate_empathy_score(message['content'])
                }
                
                transformed_data.append({**conv_features, **msg_features})
        
        self.logger.info(f"Transformation completed. {len(transformed_data)} records created")
        return pd.DataFrame(transformed_data)
    
    def load_to_warehouse(self, df, table_name):
        """Load transformed data to data warehouse."""
        
        self.logger.info(f"Loading data to table: {table_name}")
        
        # Load data to PostgreSQL
        df.to_sql(
            table_name,
            self.engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        
        self.logger.info(f"Data loading completed. {len(df)} records loaded")
    
    async def run_pipeline(self, dataset_name, table_name):
        """Run the complete ETL pipeline."""
        
        try:
            # Extract
            conversations = await self.extract_conversations(dataset_name)
            
            # Transform
            df = self.transform_conversations(conversations)
            
            # Load
            self.load_to_warehouse(df, table_name)
            
            self.logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            raise
```

### Streaming Data Processing

```python
# Apache Kafka integration for real-time processing
from kafka import KafkaProducer, KafkaConsumer
import json

class ConversationStreamProcessor:
    def __init__(self, api_key, kafka_config):
        self.api = PixelatedEmpathyAPI(api_key)
        self.kafka_config = kafka_config
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    async def stream_conversations(self, dataset_name):
        """Stream conversations to Kafka topic."""
        
        topic = f"conversations-{dataset_name}"
        
        # Use iterator for memory-efficient streaming
        async for conversation in self.api.iter_conversations(
            dataset=dataset_name,
            batch_size=50
        ):
            # Enrich conversation with additional metadata
            enriched_conversation = await self.enrich_conversation(conversation)
            
            # Send to Kafka
            self.producer.send(topic, enriched_conversation)
            
        self.producer.flush()
    
    async def enrich_conversation(self, conversation):
        """Enrich conversation with additional analysis."""
        
        # Add real-time quality validation
        validation = self.api.validate_conversation_quality(conversation)
        
        enriched = conversation.copy()
        enriched['real_time_validation'] = validation
        enriched['processing_timestamp'] = datetime.now().isoformat()
        
        return enriched
    
    def consume_and_process(self, topic):
        """Consume conversations from Kafka and process them."""
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='conversation-processors'
        )
        
        for message in consumer:
            conversation = message.value
            
            try:
                # Process conversation
                self.process_conversation(conversation)
                
            except Exception as e:
                self.logger.error(f"Error processing conversation: {e}")
                # Send to dead letter queue
                self.send_to_dlq(conversation, str(e))
    
    def process_conversation(self, conversation):
        """Process individual conversation."""
        
        # Extract insights
        insights = self.extract_insights(conversation)
        
        # Store in database
        self.store_insights(conversation['id'], insights)
        
        # Trigger alerts if needed
        if insights['risk_score'] > 0.8:
            self.trigger_alert(conversation['id'], insights)
```

---

## Real-Time Integration

### WebSocket Integration

```python
# Real-time conversation analysis with WebSockets
import asyncio
import websockets
import json

class RealTimeConversationAnalyzer:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.active_connections = set()
        self.analysis_cache = {}
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections."""
        
        self.active_connections.add(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_connections.remove(websocket)
    
    async def handle_message(self, websocket, data):
        """Handle incoming WebSocket messages."""
        
        message_type = data.get('type')
        
        if message_type == 'analyze_conversation':
            await self.analyze_conversation_real_time(websocket, data)
        elif message_type == 'subscribe_to_updates':
            await self.subscribe_to_updates(websocket, data)
        elif message_type == 'get_quality_suggestions':
            await self.get_quality_suggestions(websocket, data)
    
    async def analyze_conversation_real_time(self, websocket, data):
        """Analyze conversation in real-time."""
        
        conversation = data.get('conversation')
        request_id = data.get('request_id')
        
        try:
            # Validate conversation quality
            validation = self.api.validate_conversation_quality(conversation)
            
            # Calculate additional metrics
            metrics = await self.calculate_additional_metrics(conversation)
            
            # Send results back
            response = {
                'type': 'analysis_result',
                'request_id': request_id,
                'validation': validation,
                'additional_metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'request_id': request_id,
                'error': str(e)
            }
            await websocket.send(json.dumps(error_response))
    
    async def broadcast_update(self, update_data):
        """Broadcast updates to all connected clients."""
        
        if self.active_connections:
            message = json.dumps({
                'type': 'broadcast_update',
                'data': update_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all connected clients
            await asyncio.gather(
                *[ws.send(message) for ws in self.active_connections],
                return_exceptions=True
            )
```

### Server-Sent Events (SSE)

```python
# Server-Sent Events for real-time updates
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

class SSEConversationUpdates:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/stream/conversation-updates")
        async def stream_conversation_updates(user_id: str):
            return StreamingResponse(
                self.conversation_update_generator(user_id),
                media_type="text/plain"
            )
        
        @self.app.get("/stream/quality-updates")
        async def stream_quality_updates():
            return StreamingResponse(
                self.quality_update_generator(),
                media_type="text/plain"
            )
    
    async def conversation_update_generator(self, user_id: str):
        """Generate conversation updates for a specific user."""
        
        while True:
            try:
                # Get latest conversations for user
                updates = await self.get_user_conversation_updates(user_id)
                
                if updates:
                    for update in updates:
                        yield f"data: {json.dumps(update)}\n\n"
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                error_data = {'error': str(e), 'timestamp': datetime.now().isoformat()}
                yield f"data: {json.dumps(error_data)}\n\n"
                await asyncio.sleep(10)  # Wait longer on error
    
    async def quality_update_generator(self):
        """Generate quality metric updates."""
        
        while True:
            try:
                # Get latest quality metrics
                metrics = self.api.get_quality_metrics()
                
                update_data = {
                    'type': 'quality_update',
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(update_data)}\n\n"
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                error_data = {'error': str(e), 'timestamp': datetime.now().isoformat()}
                yield f"data: {json.dumps(error_data)}\n\n"
                await asyncio.sleep(60)  # Wait longer on error
```

---

**For more integration examples and patterns, see our [Architecture Documentation](architecture.md) and [SDK Reference](sdk_reference.md).**

*Integration patterns are updated regularly based on community feedback and best practices. Last updated: 2025-08-17*
