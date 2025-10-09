# Developer Guide: Building Applications with Pixelated Empathy AI

**Target Audience**: Software developers, ML engineers, and technical teams building empathy-aware applications.

## Table of Contents

1. [Quick Start for Developers](#quick-start-for-developers)
2. [SDK Installation and Setup](#sdk-installation-and-setup)
3. [Building Your First Application](#building-your-first-application)
4. [Common Development Patterns](#common-development-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)

---

## Quick Start for Developers

### 30-Second Setup

```bash
# Install Python SDK
pip install pixelated-empathy-api

# Or install JavaScript SDK
npm install pixelated-empathy-api
```

```python
# Python - Get started in 3 lines
from pixelated_empathy_api import PixelatedEmpathyAPI

api = PixelatedEmpathyAPI("your_api_key")
conversations = api.get_conversations(tier="professional", limit=10)
print(f"Retrieved {len(conversations['conversations'])} conversations")
```

```javascript
// JavaScript - Get started in 3 lines
const { PixelatedEmpathyAPI } = require('pixelated-empathy-api');

const api = new PixelatedEmpathyAPI('your_api_key');
const conversations = await api.getConversations({ tier: 'professional', limit: 10 });
console.log(`Retrieved ${conversations.conversations.length} conversations`);
```

---

## SDK Installation and Setup

### Python SDK

```bash
# Install via pip
pip install pixelated-empathy-api

# Or install from source
git clone https://github.com/pixelated-empathy/python-sdk
cd python-sdk
pip install -e .
```

**Environment Setup**:
```python
import os
from pixelated_empathy_api import PixelatedEmpathyAPI

# Option 1: Direct API key
api = PixelatedEmpathyAPI("your_api_key_here")

# Option 2: Environment variable (recommended)
os.environ['PIXELATED_EMPATHY_API_KEY'] = 'your_api_key_here'
api = PixelatedEmpathyAPI()

# Option 3: Configuration file
api = PixelatedEmpathyAPI.from_config('config.json')
```

### JavaScript/Node.js SDK

```bash
# Install via npm
npm install pixelated-empathy-api

# Or install via yarn
yarn add pixelated-empathy-api
```

**Environment Setup**:
```javascript
const { PixelatedEmpathyAPI } = require('pixelated-empathy-api');

// Option 1: Direct API key
const api = new PixelatedEmpathyAPI('your_api_key_here');

// Option 2: Environment variable (recommended)
process.env.PIXELATED_EMPATHY_API_KEY = 'your_api_key_here';
const api = new PixelatedEmpathyAPI();

// Option 3: Configuration object
const api = new PixelatedEmpathyAPI('your_api_key', {
    baseUrl: 'https://api.pixelatedempathy.com/v1',
    timeout: 30000,
    maxRetries: 3
});
```

---

## Building Your First Application

### Example 1: Therapeutic Chatbot Training Data Collector

```python
#!/usr/bin/env python3
"""
Therapeutic Chatbot Training Data Collector
Collects high-quality conversations for training a therapeutic chatbot
"""

from pixelated_empathy_api import PixelatedEmpathyAPI
import json
import os
from datetime import datetime

class TherapeuticDataCollector:
    def __init__(self, api_key):
        self.api = PixelatedEmpathyAPI(api_key)
        self.collected_data = []
    
    def collect_training_data(self, conditions, min_quality=0.75, conversations_per_condition=100):
        """Collect training data for specific mental health conditions."""
        
        for condition in conditions:
            print(f"Collecting data for: {condition}")
            
            # Search for condition-specific conversations
            results = self.api.search_conversations(
                query=condition,
                filters={
                    "tier": "professional",
                    "min_quality": min_quality
                },
                limit=conversations_per_condition
            )
            
            # Process each conversation
            for result in results['results']:
                conversation = self.api.get_conversation(result['conversation_id'])
                
                # Format for training (OpenAI format)
                training_example = self._format_for_training(conversation, condition)
                self.collected_data.append(training_example)
            
            print(f"Collected {len(results['results'])} conversations for {condition}")
    
    def _format_for_training(self, conversation, condition):
        """Format conversation for OpenAI fine-tuning."""
        messages = []
        
        for msg in conversation['messages']:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        return {
            "messages": messages,
            "metadata": {
                "condition": condition,
                "quality_score": conversation['quality_metrics']['overall_quality'],
                "therapeutic_accuracy": conversation['quality_metrics']['therapeutic_accuracy']
            }
        }
    
    def save_training_data(self, filename=None):
        """Save collected data to JSONL file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"therapeutic_training_data_{timestamp}.jsonl"
        
        with open(filename, 'w') as f:
            for example in self.collected_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(self.collected_data)} training examples to {filename}")
        return filename

# Usage
if __name__ == "__main__":
    collector = TherapeuticDataCollector("your_api_key")
    
    # Collect data for common mental health conditions
    conditions = ["anxiety", "depression", "stress", "grief", "trauma"]
    collector.collect_training_data(conditions, min_quality=0.8, conversations_per_condition=50)
    
    # Save training data
    filename = collector.save_training_data()
    print(f"Training data ready: {filename}")
```

### Example 2: Real-Time Conversation Quality Validator

```javascript
/**
 * Real-Time Conversation Quality Validator
 * Validates conversation quality in real-time for a chat application
 */

const { PixelatedEmpathyAPI } = require('pixelated-empathy-api');
const express = require('express');

class ConversationQualityValidator {
    constructor(apiKey) {
        this.api = new PixelatedEmpathyAPI(apiKey);
        this.app = express();
        this.setupRoutes();
    }

    setupRoutes() {
        this.app.use(express.json());

        // Validate conversation quality endpoint
        this.app.post('/validate-conversation', async (req, res) => {
            try {
                const { conversation } = req.body;
                
                // Validate with Pixelated Empathy AI
                const validation = await this.api.validateConversationQuality(conversation);
                
                // Determine if conversation meets quality standards
                const qualityThreshold = 0.7;
                const isHighQuality = validation.validation_results.overall_quality >= qualityThreshold;
                
                res.json({
                    success: true,
                    quality_score: validation.validation_results.overall_quality,
                    tier_classification: validation.tier_classification,
                    meets_standards: isHighQuality,
                    recommendations: validation.recommendations,
                    detailed_metrics: validation.validation_results
                });
                
            } catch (error) {
                res.status(500).json({
                    success: false,
                    error: error.message
                });
            }
        });

        // Get quality suggestions endpoint
        this.app.post('/quality-suggestions', async (req, res) => {
            try {
                const { partial_conversation } = req.body;
                
                // Search for similar high-quality conversations
                const suggestions = await this.api.searchConversations(
                    partial_conversation.last_user_message,
                    {
                        filters: { tier: 'professional', min_quality: 0.8 },
                        limit: 5
                    }
                );
                
                // Extract response patterns from high-quality conversations
                const responsePatterns = suggestions.results.map(result => ({
                    conversation_id: result.conversation_id,
                    quality_score: result.quality_score,
                    snippet: result.snippet
                }));
                
                res.json({
                    success: true,
                    suggestions: responsePatterns,
                    total_found: suggestions.total_matches
                });
                
            } catch (error) {
                res.status(500).json({
                    success: false,
                    error: error.message
                });
            }
        });
    }

    start(port = 3000) {
        this.app.listen(port, () => {
            console.log(`Quality validator running on port ${port}`);
        });
    }
}

// Usage
const validator = new ConversationQualityValidator('your_api_key');
validator.start(3000);
```

---

## Common Development Patterns

### Pattern 1: Batch Processing with Progress Tracking

```python
def process_conversations_batch(api, conversation_ids, batch_size=10):
    """Process conversations in batches with progress tracking."""
    
    results = []
    total = len(conversation_ids)
    
    for i in range(0, total, batch_size):
        batch = conversation_ids[i:i + batch_size]
        batch_results = []
        
        for conv_id in batch:
            try:
                conversation = api.get_conversation(conv_id)
                batch_results.append(conversation)
            except Exception as e:
                print(f"Error processing {conv_id}: {e}")
                continue
        
        results.extend(batch_results)
        
        # Progress tracking
        processed = min(i + batch_size, total)
        progress = (processed / total) * 100
        print(f"Progress: {processed}/{total} ({progress:.1f}%)")
    
    return results
```

### Pattern 2: Async Processing with Rate Limiting

```javascript
class AsyncConversationProcessor {
    constructor(api, maxConcurrent = 5) {
        this.api = api;
        this.maxConcurrent = maxConcurrent;
        this.queue = [];
        this.processing = 0;
    }

    async processConversations(conversationIds) {
        const results = [];
        const promises = [];

        for (const id of conversationIds) {
            const promise = this.processWithRateLimit(id);
            promises.push(promise);
            
            if (promises.length >= this.maxConcurrent) {
                const batch = await Promise.allSettled(promises.splice(0, this.maxConcurrent));
                results.push(...batch.filter(r => r.status === 'fulfilled').map(r => r.value));
            }
        }

        // Process remaining
        if (promises.length > 0) {
            const batch = await Promise.allSettled(promises);
            results.push(...batch.filter(r => r.status === 'fulfilled').map(r => r.value));
        }

        return results;
    }

    async processWithRateLimit(conversationId) {
        return new Promise((resolve, reject) => {
            this.queue.push({ conversationId, resolve, reject });
            this.processQueue();
        });
    }

    async processQueue() {
        if (this.processing >= this.maxConcurrent || this.queue.length === 0) {
            return;
        }

        this.processing++;
        const { conversationId, resolve, reject } = this.queue.shift();

        try {
            const conversation = await this.api.getConversation(conversationId);
            resolve(conversation);
        } catch (error) {
            reject(error);
        } finally {
            this.processing--;
            setTimeout(() => this.processQueue(), 100); // Rate limiting delay
        }
    }
}
```

### Pattern 3: Caching and Offline Support

```python
import pickle
import os
from datetime import datetime, timedelta

class CachedPixelatedEmpathyAPI:
    def __init__(self, api_key, cache_dir="./cache", cache_ttl_hours=24):
        self.api = PixelatedEmpathyAPI(api_key)
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _is_cache_valid(self, cache_path):
        if not os.path.exists(cache_path):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < self.cache_ttl
    
    def get_conversation_cached(self, conversation_id):
        cache_key = f"conv_{conversation_id}"
        cache_path = self._get_cache_path(cache_key)
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Fetch from API
        try:
            conversation = self.api.get_conversation(conversation_id)
            
            # Cache the result
            with open(cache_path, 'wb') as f:
                pickle.dump(conversation, f)
            
            return conversation
        except Exception as e:
            # Return cached version if API fails
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            raise e
```

---

## Performance Optimization

### 1. Connection Pooling and Reuse

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedPixelatedEmpathyAPI(PixelatedEmpathyAPI):
    def __init__(self, api_key, **kwargs):
        super().__init__(api_key, **kwargs)
        
        # Configure session with connection pooling
        self.session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        })
```

### 2. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelConversationProcessor:
    def __init__(self, api_key, max_workers=5):
        self.api = PixelatedEmpathyAPI(api_key)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.processed_count = 0
    
    def process_conversations_parallel(self, conversation_ids):
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._process_single, conv_id): conv_id 
                for conv_id in conversation_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                conv_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self.lock:
                        self.processed_count += 1
                        if self.processed_count % 10 == 0:
                            print(f"Processed {self.processed_count}/{len(conversation_ids)}")
                            
                except Exception as e:
                    print(f"Error processing {conv_id}: {e}")
        
        return results
    
    def _process_single(self, conversation_id):
        return self.api.get_conversation(conversation_id)
```

---

## Production Deployment

### 1. Environment Configuration

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    api_key: str
    base_url: str = "https://api.pixelatedempathy.com/v1"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_hour: int = 1000
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv('PIXELATED_EMPATHY_API_KEY'),
            base_url=os.getenv('PIXELATED_EMPATHY_BASE_URL', cls.base_url),
            timeout=int(os.getenv('PIXELATED_EMPATHY_TIMEOUT', cls.timeout)),
            max_retries=int(os.getenv('PIXELATED_EMPATHY_MAX_RETRIES', cls.max_retries)),
            rate_limit_per_hour=int(os.getenv('PIXELATED_EMPATHY_RATE_LIMIT', cls.rate_limit_per_hour))
        )

# Usage
config = APIConfig.from_env()
api = PixelatedEmpathyAPI(config.api_key, base_url=config.base_url, timeout=config.timeout)
```

### 2. Error Handling and Logging

```python
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pixelated_empathy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator

class ProductionPixelatedEmpathyAPI:
    def __init__(self, config: APIConfig):
        self.api = PixelatedEmpathyAPI(config.api_key, base_url=config.base_url)
        self.config = config
        
    @retry_with_backoff(max_retries=3)
    def get_conversation_safe(self, conversation_id):
        try:
            logger.info(f"Fetching conversation: {conversation_id}")
            conversation = self.api.get_conversation(conversation_id)
            logger.info(f"Successfully fetched conversation: {conversation_id}")
            return conversation
        except Exception as e:
            logger.error(f"Error fetching conversation {conversation_id}: {e}")
            raise
```

### 3. Health Monitoring

```python
import time
from datetime import datetime, timedelta

class APIHealthMonitor:
    def __init__(self, api):
        self.api = api
        self.health_history = []
        self.error_count = 0
        self.last_check = None
    
    def check_health(self):
        """Check API health and log metrics."""
        start_time = time.time()
        
        try:
            # Simple health check
            response = self.api.health_check()
            response_time = time.time() - start_time
            
            health_record = {
                'timestamp': datetime.now(),
                'healthy': response,
                'response_time': response_time,
                'error': None
            }
            
            if response:
                self.error_count = 0  # Reset error count on success
            
        except Exception as e:
            response_time = time.time() - start_time
            self.error_count += 1
            
            health_record = {
                'timestamp': datetime.now(),
                'healthy': False,
                'response_time': response_time,
                'error': str(e)
            }
        
        self.health_history.append(health_record)
        self.last_check = datetime.now()
        
        # Keep only last 100 records
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_record
    
    def get_health_summary(self):
        """Get health summary for the last hour."""
        if not self.health_history:
            return None
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_checks = [
            check for check in self.health_history 
            if check['timestamp'] > one_hour_ago
        ]
        
        if not recent_checks:
            return None
        
        healthy_checks = [check for check in recent_checks if check['healthy']]
        avg_response_time = sum(check['response_time'] for check in recent_checks) / len(recent_checks)
        
        return {
            'total_checks': len(recent_checks),
            'healthy_checks': len(healthy_checks),
            'health_percentage': (len(healthy_checks) / len(recent_checks)) * 100,
            'average_response_time': avg_response_time,
            'current_error_count': self.error_count,
            'last_check': self.last_check
        }
```

---

**Ready to build your application?** Check out our [API Documentation](../api/complete_api_documentation.md) for detailed endpoint information and our [GitHub Examples](https://github.com/pixelated-empathy/examples) for more code samples.

*For technical support, contact dev-support@pixelatedempathy.com*
