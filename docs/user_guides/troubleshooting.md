# Troubleshooting Guide: Pixelated Empathy AI

**Quick solutions to common issues when using the Pixelated Empathy AI API.**

## Table of Contents

1. [Authentication Issues](#authentication-issues)
2. [Rate Limiting Problems](#rate-limiting-problems)
3. [Data Quality Issues](#data-quality-issues)
4. [Performance Problems](#performance-problems)
5. [API Response Issues](#api-response-issues)
6. [SDK and Integration Issues](#sdk-and-integration-issues)
7. [Getting Additional Help](#getting-additional-help)

---

## Authentication Issues

### Problem: 401 Unauthorized Error

**Symptoms**:
```json
{
    "success": false,
    "error": {
        "code": "INVALID_API_KEY",
        "message": "Invalid API key"
    }
}
```

**Solutions**:

1. **Verify API Key Format**:
```python
def check_api_key_format(api_key):
    if not api_key:
        print("❌ API key is empty")
        return False
    
    if len(api_key) < 20:
        print("❌ API key is too short")
        return False
    
    if not api_key.startswith(('pk_', 'sk_')):
        print("❌ API key doesn't have expected prefix")
        return False
    
    print("✅ API key format looks correct")
    return True
```

2. **Test API Key**:
```python
from pixelated_empathy_api import PixelatedEmpathyAPI

try:
    api = PixelatedEmpathyAPI("your_api_key")
    result = api.health_check()
    print("✅ API key is working")
except Exception as e:
    print(f"❌ API key test failed: {e}")
```

3. **Check Environment Variables**:
```bash
# Verify environment variable is set
echo $PIXELATED_EMPATHY_API_KEY

# Set environment variable if missing
export PIXELATED_EMPATHY_API_KEY="your_api_key_here"
```

### Problem: 403 Forbidden Error

**Symptoms**: API key works but access to specific resources is denied.

**Solutions**:

1. **Check Tier Permissions**:
```python
def check_tier_access(api, tier):
    try:
        conversations = api.get_conversations(tier=tier, limit=1)
        print(f"✅ Access to {tier} tier confirmed")
        return True
    except Exception as e:
        print(f"❌ No access to {tier} tier: {e}")
        return False

# Test access to different tiers
tiers = ['basic', 'standard', 'professional', 'clinical', 'research']
for tier in tiers:
    check_tier_access(api, tier)
```

2. **Verify Account Status**:
   - Log into your dashboard at https://api.pixelatedempathy.com/dashboard
   - Check account status and tier permissions
   - Ensure billing is up to date

---

## Rate Limiting Problems

### Problem: 429 Too Many Requests

**Symptoms**:
```json
{
    "success": false,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded. Try again in 3600 seconds."
    }
}
```

**Solutions**:

1. **Implement Exponential Backoff**:
```python
import time
import random

def api_call_with_backoff(api_func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = min(300, (2 ** attempt) + random.uniform(0, 1))
            print(f"Rate limited. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
```

2. **Monitor Rate Limit Headers**:
```python
def check_rate_limit_status(response):
    headers = response.headers
    
    limit = headers.get('X-RateLimit-Limit')
    remaining = headers.get('X-RateLimit-Remaining')
    reset = headers.get('X-RateLimit-Reset')
    
    print(f"Rate Limit: {remaining}/{limit}")
    if reset:
        reset_time = datetime.fromtimestamp(int(reset))
        print(f"Resets at: {reset_time}")
```

3. **Optimize Request Patterns**:
```python
# Bad: Too many individual requests
for conv_id in conversation_ids:
    conversation = api.get_conversation(conv_id)  # 100 API calls

# Good: Use batch operations
conversations = api.get_conversations(limit=100)  # 1 API call
```

### Problem: Hitting Rate Limits Too Quickly

**Solutions**:

1. **Implement Request Throttling**:
```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests_per_hour=1000):
        self.max_requests = max_requests_per_hour
        self.requests = []
    
    def wait_if_needed(self):
        now = datetime.now()
        
        # Remove requests older than 1 hour
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(hours=1)]
        
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_until = oldest_request + timedelta(hours=1)
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                print(f"Rate limit reached. Waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds)
        
        # Record this request
        self.requests.append(now)

# Usage
rate_limiter = RateLimiter(max_requests_per_hour=900)  # Leave some buffer

for conv_id in conversation_ids:
    rate_limiter.wait_if_needed()
    conversation = api.get_conversation(conv_id)
```

---

## Data Quality Issues

### Problem: Lower Quality Scores Than Expected

**Symptoms**: Conversations have quality scores below your expectations.

**Solutions**:

1. **Analyze Quality Distribution**:
```python
def analyze_quality_distribution(conversations):
    scores = [conv['quality_score'] for conv in conversations]
    
    print(f"Quality Score Statistics:")
    print(f"  Mean: {sum(scores) / len(scores):.3f}")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    
    # Distribution by ranges
    ranges = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in ranges:
        count = sum(1 for score in scores if low <= score < high)
        percentage = (count / len(scores)) * 100
        print(f"  {low:.1f}-{high:.1f}: {count} ({percentage:.1f}%)")
```

2. **Validate Quality Metrics**:
```python
def validate_quality_metrics(conversation):
    metrics = conversation['quality_metrics']
    
    issues = []
    
    if metrics['therapeutic_accuracy'] < 0.7:
        issues.append("Low therapeutic accuracy")
    
    if metrics['conversation_coherence'] < 0.7:
        issues.append("Poor conversation coherence")
    
    if metrics['safety_score'] < 0.9:
        issues.append("Safety concerns")
    
    if len(conversation['messages']) < 4:
        issues.append("Conversation too short")
    
    return issues

# Check a sample of conversations
for conv in sample_conversations:
    issues = validate_quality_metrics(conv)
    if issues:
        print(f"Conversation {conv['id']}: {', '.join(issues)}")
```

3. **Use Appropriate Quality Thresholds**:
```python
# Adjust thresholds based on your use case
quality_thresholds = {
    'research': 0.85,      # Highest quality for research
    'clinical': 0.80,      # High quality for clinical use
    'commercial': 0.75,    # Good quality for commercial apps
    'development': 0.70,   # Acceptable for development
    'training': 0.65       # Basic quality for training data
}

use_case = 'commercial'
min_quality = quality_thresholds[use_case]

conversations = api.get_conversations(
    tier='professional',
    min_quality=min_quality
)
```

### Problem: Inconsistent Quality Across Conversations

**Solutions**:

1. **Filter by Multiple Quality Dimensions**:
```python
def filter_by_quality_dimensions(conversations, thresholds):
    filtered = []
    
    for conv in conversations:
        full_conv = api.get_conversation(conv['id'])
        metrics = full_conv['quality_metrics']
        
        passes_all_checks = True
        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                passes_all_checks = False
                break
        
        if passes_all_checks:
            filtered.append(full_conv)
    
    return filtered

# Define quality thresholds for each dimension
quality_thresholds = {
    'therapeutic_accuracy': 0.75,
    'conversation_coherence': 0.80,
    'emotional_authenticity': 0.70,
    'clinical_compliance': 0.75,
    'safety_score': 0.90
}

high_quality_conversations = filter_by_quality_dimensions(
    conversations['conversations'], 
    quality_thresholds
)
```

---

## Performance Problems

### Problem: Slow API Response Times

**Symptoms**: API calls taking longer than expected (>2 seconds).

**Solutions**:

1. **Implement Connection Pooling**:
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedAPI:
    def __init__(self, api_key):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Connection': 'keep-alive'
        })
```

2. **Use Async Processing**:
```python
import asyncio
import aiohttp

async def fetch_conversations_async(conversation_ids):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for conv_id in conversation_ids:
            task = fetch_single_conversation(session, conv_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        return successful_results
```

3. **Optimize Request Size**:
```python
# Bad: Requesting too much data at once
conversations = api.get_conversations(limit=1000)  # May timeout

# Good: Use reasonable batch sizes
batch_size = 100
all_conversations = []

for offset in range(0, total_count, batch_size):
    batch = api.get_conversations(limit=batch_size, offset=offset)
    all_conversations.extend(batch['conversations'])
```

### Problem: Memory Issues with Large Datasets

**Solutions**:

1. **Use Streaming Processing**:
```python
def process_conversations_streaming(api, dataset_name, processor_func):
    """Process conversations one at a time to avoid memory issues."""
    
    batch_size = 50
    offset = 0
    processed_count = 0
    
    while True:
        # Get batch
        batch = api.get_conversations(
            dataset=dataset_name,
            limit=batch_size,
            offset=offset
        )
        
        if not batch['conversations']:
            break
        
        # Process each conversation individually
        for conv_summary in batch['conversations']:
            try:
                # Get full conversation
                full_conv = api.get_conversation(conv_summary['id'])
                
                # Process it
                processor_func(full_conv)
                processed_count += 1
                
                # Clear from memory
                del full_conv
                
            except Exception as e:
                print(f"Error processing {conv_summary['id']}: {e}")
        
        offset += len(batch['conversations'])
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} conversations...")
        
        # Break if we got fewer results than requested
        if len(batch['conversations']) < batch_size:
            break
    
    print(f"Total processed: {processed_count}")
```

---

## API Response Issues

### Problem: Unexpected Response Format

**Symptoms**: API returns data in unexpected format or structure.

**Solutions**:

1. **Validate Response Structure**:
```python
def validate_api_response(response_data):
    """Validate API response has expected structure."""
    
    required_fields = ['success', 'timestamp']
    
    for field in required_fields:
        if field not in response_data:
            raise ValueError(f"Missing required field: {field}")
    
    if not response_data['success']:
        if 'error' not in response_data:
            raise ValueError("Unsuccessful response missing error details")
    else:
        if 'data' not in response_data:
            raise ValueError("Successful response missing data field")
    
    return True

# Use with API calls
response = api.get_conversations()
validate_api_response(response)
```

2. **Handle API Version Differences**:
```python
def handle_api_version_differences(response_data):
    """Handle differences between API versions."""
    
    # Check API version from response headers or data
    api_version = response_data.get('api_version', '1.0')
    
    if api_version.startswith('1.0'):
        # Handle v1.0 format
        return response_data['data']
    elif api_version.startswith('2.0'):
        # Handle v2.0 format (hypothetical)
        return response_data['result']
    else:
        raise ValueError(f"Unsupported API version: {api_version}")
```

### Problem: Missing or Null Data Fields

**Solutions**:

1. **Implement Safe Data Access**:
```python
def safe_get_nested(data, *keys, default=None):
    """Safely get nested dictionary values."""
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

# Usage
conversation = api.get_conversation("conv_001")

# Safe access to nested data
quality_score = safe_get_nested(
    conversation, 'quality_metrics', 'overall_quality', 
    default=0.0
)

therapeutic_accuracy = safe_get_nested(
    conversation, 'quality_metrics', 'therapeutic_accuracy',
    default=0.0
)
```

---

## SDK and Integration Issues

### Problem: SDK Installation Issues

**Solutions**:

1. **Python SDK Issues**:
```bash
# Clear pip cache
pip cache purge

# Install with verbose output
pip install -v pixelated-empathy-api

# Install from source if package issues
pip install git+https://github.com/pixelated-empathy/python-sdk.git
```

2. **JavaScript SDK Issues**:
```bash
# Clear npm cache
npm cache clean --force

# Install with verbose output
npm install --verbose pixelated-empathy-api

# Try yarn if npm fails
yarn add pixelated-empathy-api
```

### Problem: Import/Require Errors

**Solutions**:

1. **Python Import Issues**:
```python
# Try different import methods
try:
    from pixelated_empathy_api import PixelatedEmpathyAPI
except ImportError:
    try:
        import pixelated_empathy_api
        PixelatedEmpathyAPI = pixelated_empathy_api.PixelatedEmpathyAPI
    except ImportError:
        print("SDK not installed. Run: pip install pixelated-empathy-api")
```

2. **JavaScript Require Issues**:
```javascript
// Try different require methods
let PixelatedEmpathyAPI;

try {
    const sdk = require('pixelated-empathy-api');
    PixelatedEmpathyAPI = sdk.PixelatedEmpathyAPI;
} catch (error) {
    try {
        PixelatedEmpathyAPI = require('pixelated-empathy-api').default;
    } catch (error2) {
        console.error('SDK not installed. Run: npm install pixelated-empathy-api');
    }
}
```

---

## Getting Additional Help

### Self-Diagnosis Tools

1. **API Health Check**:
```python
def comprehensive_health_check(api_key):
    """Perform comprehensive health check."""
    
    print("🔍 Running comprehensive health check...")
    
    try:
        api = PixelatedEmpathyAPI(api_key)
        
        # Test basic connectivity
        health = api.health_check()
        print(f"✅ API Health: {'Healthy' if health else 'Unhealthy'}")
        
        # Test authentication
        datasets = api.list_datasets()
        print(f"✅ Authentication: Working ({len(datasets)} datasets accessible)")
        
        # Test data access
        conversations = api.get_conversations(limit=1)
        print(f"✅ Data Access: Working ({len(conversations['conversations'])} conversations retrieved)")
        
        # Test quality validation
        sample_conv = {
            "id": "test_conv",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ]
        }
        validation = api.validate_conversation_quality(sample_conv)
        print(f"✅ Quality Validation: Working (score: {validation['validation_results']['overall_quality']:.3f})")
        
        print("\n🎉 All systems operational!")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
```

### Contact Support

**Before contacting support, please gather**:

1. **Error Details**:
   - Full error message and stack trace
   - API endpoint being called
   - Request parameters used
   - Timestamp of the error

2. **Environment Information**:
   - SDK version (Python/JavaScript)
   - Operating system
   - Programming language version
   - Network configuration (if relevant)

3. **Account Information**:
   - Account email
   - API key (first 8 characters only)
   - Subscription tier

**Support Channels**:

- **Technical Issues**: api-support@pixelatedempathy.com
- **Billing Questions**: billing@pixelatedempathy.com  
- **Research Inquiries**: research@pixelatedempathy.com
- **Status Updates**: https://status.pixelatedempathy.com
- **Community Forum**: https://community.pixelatedempathy.com

**Response Times**:
- Critical issues: 2-4 hours
- Standard issues: 24 hours
- General questions: 48 hours

### Emergency Contacts

For critical production issues affecting live systems:

- **Emergency Email**: emergency@pixelatedempathy.com
- **Phone Support**: Available for Enterprise customers
- **Slack Channel**: #emergency-support (Enterprise customers)

---

**Most issues can be resolved quickly with the solutions above. If you're still experiencing problems, don't hesitate to contact our support team!**

*This troubleshooting guide is updated regularly based on common user issues.*
