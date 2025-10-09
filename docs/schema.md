# Dataset Schema Documentation

## Overview

This document describes the schema for all export formats.

## JSONL Format Schema

### `conversation_id`

- **Type**: `string`
- **Required**: Yes
- **Description**: Unique identifier for the conversation
- **Example**: `"550e8400-e29b-41d4-a716-446655440000"`

### `messages`

- **Type**: `array`
- **Required**: Yes
- **Description**: Array of conversation messages
- **Example**: `[{"role": "user", "content": "I'm feeling anxious."}, {"role": "assistant", "content": "I understand..."}]`

### `quality_score`

- **Type**: `number`
- **Required**: Yes
- **Description**: Overall quality score
- **Example**: `0.85`

### Example Record

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "role": "user",
      "content": "I'm feeling anxious."
    },
    {
      "role": "assistant",
      "content": "I understand..."
    }
  ],
  "quality_score": 0.85
}
```

## CSV Format Schema

### `conversation_id`

- **Type**: `string`
- **Required**: Yes
- **Description**: Unique identifier
- **Example**: `"550e8400-e29b-41d4-a716-446655440000"`

### `messages_json`

- **Type**: `string`
- **Required**: Yes
- **Description**: JSON-encoded messages
- **Example**: `"[{\"role\": \"user\", \"content\": \"Hello\"}]"`

### `quality_score`

- **Type**: `number`
- **Required**: Yes
- **Description**: Quality score
- **Example**: `0.85`

### Example Record

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages_json": "[{\"role\": \"user\", \"content\": \"Hello\"}]",
  "quality_score": 0.85
}
```

---

*Generated on 2025-08-03 19:09:15 UTC*
