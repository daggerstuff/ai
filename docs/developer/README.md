# Developer Documentation: Pixelated Empathy AI

**Comprehensive technical documentation for developers building with Pixelated Empathy AI.**

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SDK Reference](#sdk-reference)
3. [Integration Patterns](#integration-patterns)
4. [Advanced Topics](#advanced-topics)
5. [Development Tools](#development-tools)
6. [Contributing](#contributing)

---

## Architecture Overview

### System Architecture

Pixelated Empathy AI is built on a modern, scalable architecture designed for enterprise-grade performance:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Dashboard │    │   Mobile Apps   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │  (Rate Limiting │
                    │  Authentication)│
                    └─────────┬───────┘
                              │
                 ┌────────────┼────────────┐
                 │                         │
        ┌─────────────────┐      ┌─────────────────┐
        │  FastAPI Server │      │  Processing     │
        │  (REST Endpoints│      │  Workers        │
        │   WebSockets)   │      │  (Async Jobs)   │
        └─────────┬───────┘      └─────────┬───────┘
                  │                        │
                  └────────────┬───────────┘
                               │
                  ┌─────────────────┐
                  │   Data Layer    │
                  │  - PostgreSQL   │
                  │  - Redis Cache  │
                  │  - File Storage │
                  └─────────────────┘
```

### Core Components

#### 1. API Gateway
- **Authentication**: JWT-based API key validation
- **Rate Limiting**: Configurable limits per user tier
- **Request Routing**: Intelligent routing to backend services
- **Monitoring**: Real-time metrics and logging

#### 2. FastAPI Application
- **RESTful Endpoints**: 15+ endpoints for data access and processing
- **WebSocket Support**: Real-time updates for long-running operations
- **Async Processing**: Non-blocking operations for better performance
- **Auto-Documentation**: OpenAPI/Swagger integration

#### 3. Processing Engine
- **Quality Validation**: Real NLP-based assessment using spaCy and transformers
- **Distributed Processing**: Scalable job processing with worker pools
- **Export Pipeline**: Multi-format data export (JSONL, CSV, Parquet, etc.)
- **Search Engine**: Advanced full-text search with semantic ranking

#### 4. Data Layer
- **Conversation Storage**: 2.59M+ conversations with metadata
- **Quality Metrics**: Real-time quality scores and validation results
- **Caching**: Redis-based caching for performance optimization
- **File Storage**: Secure storage for exports and processed data

---

## SDK Reference

### Python SDK

#### Installation
```bash
pip install pixelated-empathy-api
```

#### Quick Start
```python
from pixelated_empathy_api import PixelatedEmpathyAPI

# Initialize client
api = PixelatedEmpathyAPI("your_api_key")

# Get conversations
conversations = api.get_conversations(tier="professional", limit=10)
print(f"Retrieved {len(conversations['conversations'])} conversations")
```

#### Advanced Configuration
```python
from pixelated_empathy_api import PixelatedEmpathyAPI, APIConfig

# Custom configuration
config = APIConfig(
    api_key="your_api_key",
    base_url="https://api.pixelatedempathy.com/v1",
    timeout=60,
    max_retries=5,
    rate_limit_per_hour=2000
)

api = PixelatedEmpathyAPI(config=config)
```

### JavaScript SDK

#### Installation
```bash
npm install pixelated-empathy-api
```

#### Quick Start
```javascript
const { PixelatedEmpathyAPI } = require('pixelated-empathy-api');

// Initialize client
const api = new PixelatedEmpathyAPI('your_api_key');

// Get conversations
const conversations = await api.getConversations({ 
    tier: 'professional', 
    limit: 10 
});
console.log(`Retrieved ${conversations.conversations.length} conversations`);
```

#### Advanced Configuration
```javascript
const api = new PixelatedEmpathyAPI('your_api_key', {
    baseUrl: 'https://api.pixelatedempathy.com/v1',
    timeout: 60000,
    maxRetries: 5,
    rateLimitPerHour: 2000
});
```

---

## Integration Patterns

### 1. Microservices Integration

```python
# Service-to-service communication
class ConversationService:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.cache = RedisCache()
    
    async def get_conversation_with_cache(self, conversation_id):
        # Check cache first
        cached = await self.cache.get(f"conv:{conversation_id}")
        if cached:
            return json.loads(cached)
        
        # Fetch from API
        conversation = self.api.get_conversation(conversation_id)
        
        # Cache for 1 hour
        await self.cache.setex(
            f"conv:{conversation_id}", 
            3600, 
            json.dumps(conversation)
        )
        
        return conversation
```

### 2. Event-Driven Architecture

```python
# Event-driven processing
import asyncio
from typing import Dict, Any

class ConversationProcessor:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.event_queue = asyncio.Queue()
    
    async def process_conversation_event(self, event: Dict[str, Any]):
        """Process conversation-related events."""
        
        event_type = event.get('type')
        conversation_id = event.get('conversation_id')
        
        if event_type == 'quality_validation_requested':
            await self._validate_conversation_quality(conversation_id)
        elif event_type == 'export_requested':
            await self._export_conversation(conversation_id, event.get('format'))
        elif event_type == 'analysis_requested':
            await self._analyze_conversation(conversation_id)
    
    async def _validate_conversation_quality(self, conversation_id):
        conversation = self.api.get_conversation(conversation_id)
        validation = self.api.validate_conversation_quality(conversation)
        
        # Emit quality validation completed event
        await self._emit_event({
            'type': 'quality_validation_completed',
            'conversation_id': conversation_id,
            'quality_score': validation['validation_results']['overall_quality']
        })
```

### 3. Batch Processing Pipeline

```python
# Batch processing for large datasets
class BatchProcessor:
    def __init__(self, api_key, batch_size=100):
        self.api = PixelatedEmpathyAPI(api_key)
        self.batch_size = batch_size
    
    async def process_dataset(self, dataset_name, processor_func):
        """Process entire dataset in batches."""
        
        offset = 0
        processed_count = 0
        
        while True:
            # Get batch
            batch = self.api.get_conversations(
                dataset=dataset_name,
                limit=self.batch_size,
                offset=offset
            )
            
            if not batch['conversations']:
                break
            
            # Process batch concurrently
            tasks = []
            for conv_summary in batch['conversations']:
                task = self._process_single_conversation(
                    conv_summary['id'], 
                    processor_func
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            successful = [r for r in results if not isinstance(r, Exception)]
            processed_count += len(successful)
            
            print(f"Processed {processed_count} conversations...")
            
            offset += len(batch['conversations'])
            
            # Break if we got fewer results than requested
            if len(batch['conversations']) < self.batch_size:
                break
        
        return processed_count
    
    async def _process_single_conversation(self, conversation_id, processor_func):
        conversation = self.api.get_conversation(conversation_id)
        return await processor_func(conversation)
```

---

## Advanced Topics

### 1. Custom Quality Validators

```python
# Implementing custom quality validation
from typing import Dict, List, Any
import spacy
from transformers import pipeline

class CustomQualityValidator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.empathy_classifier = pipeline(
            "text-classification",
            model="your-custom-empathy-model"
        )
    
    def validate_conversation(self, conversation: Dict[str, Any]) -> Dict[str, float]:
        """Custom quality validation with domain-specific metrics."""
        
        messages = conversation['messages']
        assistant_messages = [
            msg['content'] for msg in messages 
            if msg['role'] == 'assistant'
        ]
        
        if not assistant_messages:
            return {'error': 'No assistant messages found'}
        
        # Calculate custom metrics
        metrics = {
            'empathy_score': self._calculate_empathy_score(assistant_messages),
            'therapeutic_language': self._assess_therapeutic_language(assistant_messages),
            'response_appropriateness': self._assess_response_appropriateness(messages),
            'conversation_flow': self._assess_conversation_flow(messages),
            'safety_compliance': self._assess_safety_compliance(assistant_messages)
        }
        
        # Calculate overall score
        weights = {
            'empathy_score': 0.25,
            'therapeutic_language': 0.25,
            'response_appropriateness': 0.20,
            'conversation_flow': 0.15,
            'safety_compliance': 0.15
        }
        
        overall_score = sum(
            metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        metrics['overall_quality'] = overall_score
        return metrics
    
    def _calculate_empathy_score(self, messages: List[str]) -> float:
        """Calculate empathy score using custom model."""
        
        empathy_scores = []
        for message in messages:
            result = self.empathy_classifier(message)[0]
            if result['label'] == 'EMPATHETIC':
                empathy_scores.append(result['score'])
            else:
                empathy_scores.append(1 - result['score'])
        
        return sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0
    
    def _assess_therapeutic_language(self, messages: List[str]) -> float:
        """Assess use of therapeutic language patterns."""
        
        therapeutic_patterns = [
            r'\b(understand|feel|sounds like|that must be)\b',
            r'\b(can you tell me more|how does that make you feel)\b',
            r'\b(it\'s normal to|many people experience)\b',
            r'\b(what would help|what do you think)\b'
        ]
        
        total_score = 0
        for message in messages:
            doc = self.nlp(message.lower())
            pattern_matches = 0
            
            for pattern in therapeutic_patterns:
                if re.search(pattern, message.lower()):
                    pattern_matches += 1
            
            # Score based on pattern density
            message_score = min(pattern_matches / len(therapeutic_patterns), 1.0)
            total_score += message_score
        
        return total_score / len(messages) if messages else 0.0
```

### 2. Real-Time Processing with WebSockets

```python
# Real-time processing with WebSocket integration
import asyncio
import websockets
import json

class RealTimeProcessor:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.active_connections = set()
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time updates."""
        
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
        
        if message_type == 'validate_conversation':
            await self._handle_validation_request(websocket, data)
        elif message_type == 'process_batch':
            await self._handle_batch_request(websocket, data)
        elif message_type == 'subscribe_to_job':
            await self._handle_job_subscription(websocket, data)
    
    async def _handle_validation_request(self, websocket, data):
        """Handle real-time conversation validation."""
        
        conversation = data.get('conversation')
        request_id = data.get('request_id')
        
        try:
            # Validate conversation
            validation = self.api.validate_conversation_quality(conversation)
            
            # Send result back
            response = {
                'type': 'validation_result',
                'request_id': request_id,
                'result': validation
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'request_id': request_id,
                'error': str(e)
            }
            await websocket.send(json.dumps(error_response))
    
    async def broadcast_job_update(self, job_id, status, progress):
        """Broadcast job updates to all subscribed clients."""
        
        message = {
            'type': 'job_update',
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all connected clients
        if self.active_connections:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.active_connections],
                return_exceptions=True
            )
```

### 3. Machine Learning Pipeline Integration

```python
# ML pipeline integration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

class MLPipelineIntegration:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
    
    def prepare_training_data(self, dataset_name, sample_size=1000):
        """Prepare training data from Pixelated Empathy AI dataset."""
        
        # Get high-quality conversations for training
        conversations = self.api.get_conversations(
            dataset=dataset_name,
            tier="professional",
            min_quality=0.8,
            limit=sample_size
        )
        
        training_data = []
        labels = []
        
        for conv_summary in conversations['conversations']:
            # Get full conversation
            conversation = self.api.get_conversation(conv_summary['id'])
            
            # Extract features
            text_features = self._extract_text_features(conversation)
            quality_score = conversation['quality_metrics']['overall_quality']
            
            training_data.append(text_features)
            labels.append(1 if quality_score >= 0.8 else 0)  # Binary classification
        
        return training_data, labels
    
    def train_model(self, dataset_name):
        """Train ML model using Pixelated Empathy AI data."""
        
        print("Preparing training data...")
        texts, labels = self.prepare_training_data(dataset_name)
        
        print("Vectorizing text features...")
        X = self.vectorizer.fit_transform(texts)
        
        print("Training classifier...")
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        print("Training completed!")
        
        # Save model
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        joblib.dump(self.classifier, 'classifier.pkl')
    
    def predict_quality(self, conversation_text):
        """Predict conversation quality using trained model."""
        
        if not self.is_trained:
            # Load pre-trained model
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.classifier = joblib.load('classifier.pkl')
            self.is_trained = True
        
        # Vectorize input
        X = self.vectorizer.transform([conversation_text])
        
        # Predict
        prediction = self.classifier.predict_proba(X)[0]
        
        return {
            'high_quality_probability': prediction[1],
            'predicted_class': 'high_quality' if prediction[1] > 0.5 else 'low_quality'
        }
    
    def _extract_text_features(self, conversation):
        """Extract text features from conversation."""
        
        # Combine all messages into single text
        all_text = []
        for message in conversation['messages']:
            all_text.append(message['content'])
        
        return ' '.join(all_text)
```

---

## Development Tools

### 1. CLI Tools

```bash
# Install CLI tools
pip install pixelated-empathy-cli

# Initialize new project
pe-cli init my-project --template=chatbot

# Validate API key
pe-cli auth validate

# Download dataset
pe-cli data download --dataset=professional --format=jsonl --output=./data/

# Run quality validation
pe-cli quality validate --input=./conversations.jsonl --output=./results.json

# Generate SDK documentation
pe-cli docs generate --language=python --output=./docs/
```

### 2. Testing Framework

```python
# Testing utilities
import pytest
from pixelated_empathy_testing import MockAPI, ConversationFactory

class TestConversationProcessor:
    def setup_method(self):
        self.mock_api = MockAPI()
        self.processor = ConversationProcessor(self.mock_api)
    
    def test_quality_validation(self):
        # Create test conversation
        conversation = ConversationFactory.create_conversation(
            quality_score=0.85,
            tier="professional"
        )
        
        # Mock API response
        self.mock_api.set_response(
            'validate_conversation_quality',
            {
                'validation_results': {
                    'overall_quality': 0.85,
                    'therapeutic_accuracy': 0.80
                },
                'tier_classification': 'professional'
            }
        )
        
        # Test validation
        result = self.processor.validate_conversation(conversation)
        
        assert result['validation_results']['overall_quality'] == 0.85
        assert result['tier_classification'] == 'professional'
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        # Create test dataset
        conversations = ConversationFactory.create_batch(
            count=10,
            quality_range=(0.7, 0.9)
        )
        
        # Mock API responses
        for conv in conversations:
            self.mock_api.set_conversation(conv['id'], conv)
        
        # Test batch processing
        results = await self.processor.process_batch(
            [conv['id'] for conv in conversations]
        )
        
        assert len(results) == 10
        assert all(r['quality_score'] >= 0.7 for r in results)
```

### 3. Monitoring and Observability

```python
# Monitoring integration
from prometheus_client import Counter, Histogram, Gauge
import logging

class APIMonitoring:
    def __init__(self):
        self.request_count = Counter(
            'pixelated_empathy_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'pixelated_empathy_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_connections = Gauge(
            'pixelated_empathy_active_connections',
            'Number of active connections'
        )
        
        self.logger = logging.getLogger(__name__)
    
    def track_request(self, method, endpoint, status_code, duration):
        """Track API request metrics."""
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status_code
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log request details
        self.logger.info(
            f"API Request: {method} {endpoint} - {status_code} - {duration:.3f}s"
        )
    
    def track_connection(self, action='connect'):
        """Track connection metrics."""
        
        if action == 'connect':
            self.active_connections.inc()
        elif action == 'disconnect':
            self.active_connections.dec()
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/pixelated-empathy/api.git
cd api

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Code Standards

- **Python**: Follow PEP 8, use Black for formatting
- **JavaScript**: Follow ESLint configuration, use Prettier
- **Documentation**: Use docstrings for all public functions
- **Testing**: Maintain >90% test coverage
- **Type Hints**: Use type hints for all Python code

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run test suite: `pytest`
5. Run linting: `flake8` and `black`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open Pull Request

---

**For detailed API reference, see our [Complete API Documentation](../api/complete_api_documentation.md).**

**For getting started, see our [User Guides](../user_guides/getting_started.md).**

*This documentation is continuously updated. Last updated: 2025-08-17*
