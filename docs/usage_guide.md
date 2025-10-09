# Usage Guide and Best Practices

## Getting Started

### Loading the Dataset

#### JSONL Format (Recommended for large datasets)

```python
import json

conversations = []
with open('pixelated_empathy_dataset.jsonl', 'r') as f:
    for line in f:
        conversations.append(json.loads(line))

print(f"Loaded {len(conversations)} conversations")
```

#### CSV Format (Good for analysis)

```python
import pandas as pd

df = pd.read_csv('pixelated_empathy_dataset.csv')
print(f"Dataset shape: {df.shape}")
```

### Basic Analysis

```python
# Dataset overview
print(f"Total conversations: {len(conversations)}")

# Quality distribution
quality_scores = [c['quality_score'] for c in conversations]
print(f"Average quality: {sum(quality_scores) / len(quality_scores):.3f}")

# Message statistics
total_messages = sum(len(c['messages']) for c in conversations)
print(f"Total messages: {total_messages}")
print(f"Average messages per conversation: {total_messages / len(conversations):.1f}")
```

## Use Cases

### 1. Training Conversational AI

```python
def prepare_training_data(conversations, min_quality=0.7):
    training_pairs = []
    
    for conv in conversations:
        if conv['quality_score'] >= min_quality:
            messages = conv['messages']
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]['content']
                    assistant_msg = messages[i + 1]['content']
                    training_pairs.append({
                        'input': user_msg,
                        'output': assistant_msg,
                        'quality': conv['quality_score']
                    })
    
    return training_pairs

training_data = prepare_training_data(conversations)
print(f"Generated {len(training_data)} training pairs")
```

### 2. Quality-based Filtering

```python
# Create quality tiers
premium = [c for c in conversations if c['quality_score'] >= 0.9]
standard = [c for c in conversations if 0.7 <= c['quality_score'] < 0.9]
basic = [c for c in conversations if 0.5 <= c['quality_score'] < 0.7]

print(f"Premium: {len(premium)} conversations")
print(f"Standard: {len(standard)} conversations")
print(f"Basic: {len(basic)} conversations")
```

### 3. Content Analysis

```python
from collections import Counter

# Analyze message lengths
message_lengths = []
for conv in conversations:
    for msg in conv['messages']:
        message_lengths.append(len(msg['content'].split()))

print(f"Average message length: {sum(message_lengths) / len(message_lengths):.1f} words")
```

## Best Practices

### Data Quality
1. Always check quality scores before using conversations
2. Filter by appropriate quality thresholds for your use case
3. Validate data integrity after loading

### Performance
1. Use JSONL format for large-scale processing
2. Use CSV format for analysis and visualization
3. Consider streaming for very large datasets

### Ethical Considerations
1. Respect privacy - all data is anonymized
2. Use appropriate quality filters
3. Follow therapeutic guidelines for healthcare applications

---

*Generated on 2025-08-03 19:09:15 UTC*
