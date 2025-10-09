# SDK Reference: Pixelated Empathy AI

**Complete reference documentation for Python and JavaScript SDKs.**

## Table of Contents

1. [Python SDK Reference](#python-sdk-reference)
2. [JavaScript SDK Reference](#javascript-sdk-reference)
3. [Error Handling](#error-handling)
4. [Configuration Options](#configuration-options)
5. [Advanced Usage](#advanced-usage)

---

## Python SDK Reference

### Installation

```bash
pip install pixelated-empathy-api
```

### Basic Usage

```python
from pixelated_empathy_api import PixelatedEmpathyAPI

# Initialize client
api = PixelatedEmpathyAPI("your_api_key")

# Basic operations
datasets = api.list_datasets()
conversations = api.get_conversations(tier="professional", limit=10)
```

### Class: PixelatedEmpathyAPI

#### Constructor

```python
PixelatedEmpathyAPI(
    api_key: str,
    base_url: str = "https://api.pixelatedempathy.com/v1",
    timeout: int = 30,
    max_retries: int = 3
)
```

**Parameters:**
- `api_key` (str): Your API key from the dashboard
- `base_url` (str, optional): API base URL
- `timeout` (int, optional): Request timeout in seconds
- `max_retries` (int, optional): Maximum retry attempts

#### Dataset Methods

##### `list_datasets() -> List[Dict[str, Any]]`

List all available datasets.

```python
datasets = api.list_datasets()
for dataset in datasets:
    print(f"{dataset['name']}: {dataset['conversations']} conversations")
```

**Returns:** List of dataset information dictionaries

**Example Response:**
```python
[
    {
        "name": "priority_complete_fixed",
        "description": "Priority conversations with complete processing",
        "conversations": 297917,
        "quality_score": 0.624,
        "tiers": ["basic", "standard", "professional"]
    }
]
```

##### `get_dataset_info(dataset_name: str) -> Dict[str, Any]`

Get detailed information about a specific dataset.

```python
info = api.get_dataset_info("priority_complete_fixed")
print(f"Total conversations: {info['statistics']['total_conversations']}")
```

**Parameters:**
- `dataset_name` (str): Name of the dataset

**Returns:** Dataset information dictionary

#### Conversation Methods

##### `get_conversations(**kwargs) -> Dict[str, Any]`

Get conversations with optional filtering.

```python
conversations = api.get_conversations(
    dataset="priority_complete_fixed",
    tier="professional",
    min_quality=0.75,
    limit=100,
    offset=0
)
```

**Parameters:**
- `dataset` (str, optional): Filter by dataset name
- `tier` (str, optional): Filter by quality tier
- `min_quality` (float, optional): Minimum quality score (0.0-1.0)
- `limit` (int, optional): Maximum results (default: 100, max: 1000)
- `offset` (int, optional): Pagination offset (default: 0)

**Returns:** Dictionary with conversations and pagination info

##### `get_conversation(conversation_id: str) -> Dict[str, Any]`

Get a specific conversation by ID.

```python
conversation = api.get_conversation("conv_000001")
print(f"Messages: {len(conversation['messages'])}")
print(f"Quality: {conversation['quality_metrics']['overall_quality']}")
```

**Parameters:**
- `conversation_id` (str): Unique conversation identifier

**Returns:** Complete conversation data

##### `iter_conversations(**kwargs) -> Iterator[Dict[str, Any]]`

Iterate through all conversations with automatic pagination.

```python
for conversation in api.iter_conversations(tier="professional", batch_size=50):
    print(f"Processing conversation: {conversation['id']}")
    # Process conversation...
```

**Parameters:**
- Same as `get_conversations()` plus:
- `batch_size` (int, optional): Conversations per request (default: 100)

**Yields:** Individual conversation dictionaries

#### Quality Methods

##### `get_quality_metrics(**kwargs) -> Dict[str, Any]`

Get quality metrics for datasets or tiers.

```python
metrics = api.get_quality_metrics(dataset="professional_datasets_final")
print(f"Average quality: {metrics['overall_statistics']['average_quality']}")
```

**Parameters:**
- `dataset` (str, optional): Filter by dataset name
- `tier` (str, optional): Filter by quality tier

**Returns:** Quality metrics dictionary

##### `validate_conversation_quality(conversation: Dict[str, Any]) -> Dict[str, Any]`

Validate conversation quality using NLP-based assessment.

```python
conversation = {
    "id": "test_conv",
    "messages": [
        {"role": "user", "content": "I'm feeling anxious."},
        {"role": "assistant", "content": "I understand. Can you tell me more?"}
    ]
}

validation = api.validate_conversation_quality(conversation)
print(f"Quality score: {validation['validation_results']['overall_quality']}")
print(f"Tier: {validation['tier_classification']}")
```

**Parameters:**
- `conversation` (dict): Conversation data to validate

**Returns:** Validation results with quality metrics and recommendations

#### Processing Methods

##### `submit_processing_job(dataset_name: str, processing_type: str, parameters: Dict = None) -> Dict[str, Any]`

Submit a processing job for dataset analysis or export.

```python
job = api.submit_processing_job(
    dataset_name="priority_complete_fixed",
    processing_type="quality_validation",
    parameters={
        "tier_filter": "professional",
        "min_quality": 0.8
    }
)
print(f"Job ID: {job['job_id']}")
```

**Parameters:**
- `dataset_name` (str): Dataset to process
- `processing_type` (str): Type of processing (quality_validation, export, analysis)
- `parameters` (dict, optional): Processing parameters

**Returns:** Job information dictionary

##### `get_job_status(job_id: str) -> Dict[str, Any]`

Get the status of a processing job.

```python
status = api.get_job_status("job_20250817_010000")
print(f"Status: {status['status']} ({status['progress']}%)")
```

**Parameters:**
- `job_id` (str): Unique job identifier

**Returns:** Job status dictionary

##### `wait_for_job(job_id: str, poll_interval: int = 30, timeout: int = 3600) -> Dict[str, Any]`

Wait for a processing job to complete.

```python
final_status = api.wait_for_job("job_20250817_010000", poll_interval=10)
print(f"Job completed with status: {final_status['status']}")
```

**Parameters:**
- `job_id` (str): Job identifier to wait for
- `poll_interval` (int, optional): Seconds between status checks
- `timeout` (int, optional): Maximum wait time in seconds

**Returns:** Final job status

#### Search Methods

##### `search_conversations(query: str, **kwargs) -> Dict[str, Any]`

Search conversations using advanced filters.

```python
results = api.search_conversations(
    "anxiety therapy techniques",
    filters={"tier": "professional", "min_quality": 0.8},
    limit=50
)
print(f"Found {results['total_matches']} matching conversations")
```

**Parameters:**
- `query` (str): Search query string
- `filters` (dict, optional): Search filters
- `limit` (int, optional): Maximum results
- `offset` (int, optional): Pagination offset

**Returns:** Search results dictionary

#### Statistics Methods

##### `get_statistics_overview() -> Dict[str, Any]`

Get comprehensive statistics about the API and datasets.

```python
stats = api.get_statistics_overview()
print(f"Total conversations: {stats['total_conversations']}")
print(f"API requests today: {stats['api_usage']['requests_today']}")
```

**Returns:** Statistics overview dictionary

#### Export Methods

##### `export_data(dataset: str, **kwargs) -> Dict[str, Any]`

Export data in specified format.

```python
export = api.export_data(
    dataset="professional_datasets_final",
    format="jsonl",
    tier="professional",
    min_quality=0.8
)
print(f"Export ID: {export['export_id']}")
print(f"Download URL: {export['download_url']}")
```

**Parameters:**
- `dataset` (str): Dataset name to export
- `format` (str, optional): Export format (jsonl, csv, parquet, huggingface, openai)
- `tier` (str, optional): Quality tier filter
- `min_quality` (float, optional): Minimum quality score

**Returns:** Export information dictionary

#### Utility Methods

##### `health_check() -> bool`

Check if the API is healthy.

```python
if api.health_check():
    print("API is healthy")
else:
    print("API is not responding")
```

**Returns:** True if healthy, False otherwise

---

## JavaScript SDK Reference

### Installation

```bash
npm install pixelated-empathy-api
```

### Basic Usage

```javascript
const { PixelatedEmpathyAPI } = require('pixelated-empathy-api');

// Initialize client
const api = new PixelatedEmpathyAPI('your_api_key');

// Basic operations
const datasets = await api.listDatasets();
const conversations = await api.getConversations({ tier: 'professional', limit: 10 });
```

### Class: PixelatedEmpathyAPI

#### Constructor

```javascript
new PixelatedEmpathyAPI(apiKey, options = {})
```

**Parameters:**
- `apiKey` (string): Your API key from the dashboard
- `options` (object, optional): Configuration options
  - `baseUrl` (string): API base URL
  - `timeout` (number): Request timeout in milliseconds
  - `maxRetries` (number): Maximum retry attempts

#### Dataset Methods

##### `listDatasets() -> Promise<Array>`

List all available datasets.

```javascript
const datasets = await api.listDatasets();
datasets.forEach(dataset => {
    console.log(`${dataset.name}: ${dataset.conversations} conversations`);
});
```

**Returns:** Promise resolving to array of dataset objects

##### `getDatasetInfo(datasetName) -> Promise<Object>`

Get detailed information about a specific dataset.

```javascript
const info = await api.getDatasetInfo('priority_complete_fixed');
console.log(`Total conversations: ${info.statistics.total_conversations}`);
```

**Parameters:**
- `datasetName` (string): Name of the dataset

**Returns:** Promise resolving to dataset information object

#### Conversation Methods

##### `getConversations(options = {}) -> Promise<Object>`

Get conversations with optional filtering.

```javascript
const conversations = await api.getConversations({
    dataset: 'priority_complete_fixed',
    tier: 'professional',
    minQuality: 0.75,
    limit: 100,
    offset: 0
});
```

**Parameters:**
- `options` (object, optional): Filtering options
  - `dataset` (string): Filter by dataset name
  - `tier` (string): Filter by quality tier
  - `minQuality` (number): Minimum quality score (0.0-1.0)
  - `limit` (number): Maximum results (default: 100, max: 1000)
  - `offset` (number): Pagination offset (default: 0)

**Returns:** Promise resolving to conversations and pagination info

##### `getConversation(conversationId) -> Promise<Object>`

Get a specific conversation by ID.

```javascript
const conversation = await api.getConversation('conv_000001');
console.log(`Messages: ${conversation.messages.length}`);
console.log(`Quality: ${conversation.quality_metrics.overall_quality}`);
```

**Parameters:**
- `conversationId` (string): Unique conversation identifier

**Returns:** Promise resolving to complete conversation data

##### `iterConversations(options = {}) -> AsyncGenerator<Object>`

Iterate through all conversations with automatic pagination.

```javascript
for await (const conversation of api.iterConversations({ tier: 'professional' })) {
    console.log(`Processing conversation: ${conversation.id}`);
    // Process conversation...
}
```

**Parameters:**
- Same as `getConversations()` plus:
- `batchSize` (number): Conversations per request (default: 100)

**Yields:** Individual conversation objects

#### Quality Methods

##### `getQualityMetrics(options = {}) -> Promise<Object>`

Get quality metrics for datasets or tiers.

```javascript
const metrics = await api.getQualityMetrics({ dataset: 'professional_datasets_final' });
console.log(`Average quality: ${metrics.overall_statistics.average_quality}`);
```

**Parameters:**
- `options` (object, optional): Filtering options
  - `dataset` (string): Filter by dataset name
  - `tier` (string): Filter by quality tier

**Returns:** Promise resolving to quality metrics object

##### `validateConversationQuality(conversation) -> Promise<Object>`

Validate conversation quality using NLP-based assessment.

```javascript
const conversation = {
    id: 'test_conv',
    messages: [
        { role: 'user', content: "I'm feeling anxious." },
        { role: 'assistant', content: 'I understand. Can you tell me more?' }
    ]
};

const validation = await api.validateConversationQuality(conversation);
console.log(`Quality score: ${validation.validation_results.overall_quality}`);
console.log(`Tier: ${validation.tier_classification}`);
```

**Parameters:**
- `conversation` (object): Conversation data to validate

**Returns:** Promise resolving to validation results

#### Processing Methods

##### `submitProcessingJob(datasetName, processingType, parameters = {}) -> Promise<Object>`

Submit a processing job for dataset analysis or export.

```javascript
const job = await api.submitProcessingJob(
    'priority_complete_fixed',
    'quality_validation',
    {
        tier_filter: 'professional',
        min_quality: 0.8
    }
);
console.log(`Job ID: ${job.job_id}`);
```

**Parameters:**
- `datasetName` (string): Dataset to process
- `processingType` (string): Type of processing
- `parameters` (object): Processing parameters

**Returns:** Promise resolving to job information

##### `getJobStatus(jobId) -> Promise<Object>`

Get the status of a processing job.

```javascript
const status = await api.getJobStatus('job_20250817_010000');
console.log(`Status: ${status.status} (${status.progress}%)`);
```

**Parameters:**
- `jobId` (string): Unique job identifier

**Returns:** Promise resolving to job status

##### `waitForJob(jobId, options = {}) -> Promise<Object>`

Wait for a processing job to complete.

```javascript
const finalStatus = await api.waitForJob('job_20250817_010000', {
    pollInterval: 10,
    timeout: 3600
});
console.log(`Job completed with status: ${finalStatus.status}`);
```

**Parameters:**
- `jobId` (string): Job identifier to wait for
- `options` (object): Wait options
  - `pollInterval` (number): Seconds between status checks
  - `timeout` (number): Maximum wait time in seconds

**Returns:** Promise resolving to final job status

#### Search Methods

##### `searchConversations(query, options = {}) -> Promise<Object>`

Search conversations using advanced filters.

```javascript
const results = await api.searchConversations(
    'anxiety therapy techniques',
    {
        filters: { tier: 'professional', min_quality: 0.8 },
        limit: 50
    }
);
console.log(`Found ${results.total_matches} matching conversations`);
```

**Parameters:**
- `query` (string): Search query string
- `options` (object): Search options
  - `filters` (object): Search filters
  - `limit` (number): Maximum results
  - `offset` (number): Pagination offset

**Returns:** Promise resolving to search results

#### Statistics Methods

##### `getStatisticsOverview() -> Promise<Object>`

Get comprehensive statistics about the API and datasets.

```javascript
const stats = await api.getStatisticsOverview();
console.log(`Total conversations: ${stats.total_conversations}`);
console.log(`API requests today: ${stats.api_usage.requests_today}`);
```

**Returns:** Promise resolving to statistics overview

#### Export Methods

##### `exportData(dataset, options = {}) -> Promise<Object>`

Export data in specified format.

```javascript
const exportInfo = await api.exportData('professional_datasets_final', {
    format: 'jsonl',
    tier: 'professional',
    minQuality: 0.8
});
console.log(`Export ID: ${exportInfo.export_id}`);
console.log(`Download URL: ${exportInfo.download_url}`);
```

**Parameters:**
- `dataset` (string): Dataset name to export
- `options` (object): Export options
  - `format` (string): Export format
  - `tier` (string): Quality tier filter
  - `minQuality` (number): Minimum quality score

**Returns:** Promise resolving to export information

#### Utility Methods

##### `healthCheck() -> Promise<boolean>`

Check if the API is healthy.

```javascript
const isHealthy = await api.healthCheck();
if (isHealthy) {
    console.log('API is healthy');
} else {
    console.log('API is not responding');
}
```

**Returns:** Promise resolving to boolean health status

---

## Error Handling

### Python Error Classes

```python
from pixelated_empathy_api import (
    PixelatedEmpathyAPIError,
    RateLimitError,
    AuthenticationError,
    ValidationError
)

try:
    conversations = api.get_conversations(tier="invalid_tier")
except ValidationError as e:
    print(f"Validation error: {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
except PixelatedEmpathyAPIError as e:
    print(f"API error: {e.message} (Code: {e.error_code})")
```

### JavaScript Error Handling

```javascript
const { 
    PixelatedEmpathyAPIError, 
    RateLimitError 
} = require('pixelated-empathy-api');

try {
    const conversations = await api.getConversations({ tier: 'invalid_tier' });
} catch (error) {
    if (error instanceof RateLimitError) {
        console.log(`Rate limited. Retry after ${error.retryAfter} seconds`);
    } else if (error instanceof PixelatedEmpathyAPIError) {
        console.log(`API error: ${error.message} (Code: ${error.errorCode})`);
    } else {
        console.log(`Unexpected error: ${error.message}`);
    }
}
```

---

## Configuration Options

### Environment Variables

```bash
# API Configuration
export PIXELATED_EMPATHY_API_KEY="your_api_key"
export PIXELATED_EMPATHY_BASE_URL="https://api.pixelatedempathy.com/v1"
export PIXELATED_EMPATHY_TIMEOUT="60"
export PIXELATED_EMPATHY_MAX_RETRIES="5"

# Logging Configuration
export PIXELATED_EMPATHY_LOG_LEVEL="INFO"
export PIXELATED_EMPATHY_LOG_FILE="./pixelated_empathy.log"
```

### Configuration Files

#### Python Configuration

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    api_key: str
    base_url: str = "https://api.pixelatedempathy.com/v1"
    timeout: int = 30
    max_retries: int = 3
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str):
        import json
        with open(config_path) as f:
            config_data = json.load(f)
        return cls(**config_data)

# Usage
config = APIConfig.from_file('config.json')
api = PixelatedEmpathyAPI(config=config)
```

#### JavaScript Configuration

```javascript
// config.js
const config = {
    apiKey: process.env.PIXELATED_EMPATHY_API_KEY,
    baseUrl: process.env.PIXELATED_EMPATHY_BASE_URL || 'https://api.pixelatedempathy.com/v1',
    timeout: parseInt(process.env.PIXELATED_EMPATHY_TIMEOUT) || 30000,
    maxRetries: parseInt(process.env.PIXELATED_EMPATHY_MAX_RETRIES) || 3,
    logLevel: process.env.PIXELATED_EMPATHY_LOG_LEVEL || 'INFO'
};

// Usage
const api = new PixelatedEmpathyAPI(config.apiKey, {
    baseUrl: config.baseUrl,
    timeout: config.timeout,
    maxRetries: config.maxRetries
});
```

---

## Advanced Usage

### Custom HTTP Client (Python)

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class CustomPixelatedEmpathyAPI(PixelatedEmpathyAPI):
    def __init__(self, api_key, **kwargs):
        super().__init__(api_key, **kwargs)
        
        # Configure custom session
        self.session = requests.Session()
        
        # Custom retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Custom adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=50
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
```

### Middleware Integration (JavaScript)

```javascript
class MiddlewareAPI extends PixelatedEmpathyAPI {
    constructor(apiKey, options = {}) {
        super(apiKey, options);
        this.middleware = [];
    }
    
    use(middleware) {
        this.middleware.push(middleware);
    }
    
    async _makeRequest(method, endpoint, options = {}) {
        // Apply middleware
        for (const middleware of this.middleware) {
            options = await middleware(method, endpoint, options);
        }
        
        return super._makeRequest(method, endpoint, options);
    }
}

// Usage with logging middleware
const api = new MiddlewareAPI('your_api_key');

api.use(async (method, endpoint, options) => {
    console.log(`Making ${method} request to ${endpoint}`);
    return options;
});
```

---

**For more examples and advanced usage patterns, see our [Developer Guide](README.md) and [User Guides](../user_guides/).**

*This SDK reference is automatically generated from source code. Last updated: 2025-08-17*
