# Pixelated Empathy AI - API Examples and Tutorials

## Quick Start Tutorial

### 1. Authentication
```bash
export API_KEY="your_api_key_here"
```

### 2. Basic Conversation
```bash
curl -X POST "https://api.pixelatedempathy.com/v1/conversation" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "I am feeling anxious", "user_id": "user123"}'
```

### 3. Python Example
```python
from pixelated_empathy import PixelatedEmpathyClient

client = PixelatedEmpathyClient("your_api_key")
result = client.create_conversation("I need help", "user123")
print(result["response"])
```

### 4. JavaScript Example
```javascript
const client = new PixelatedEmpathyClient('your_api_key');
const result = await client.createConversation('Hello', 'user123');
console.log(result.response);
```
