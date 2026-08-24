# Dream-Reflection Integration Guide

This guide documents the integration between reflection tasks and dream cycles, enabling insights from reflections to feed back into long-term memory.

## Overview

The dream-reflection integration implements a sleep-inspired memory consolidation model:

```
Memories → Dream Cycle → Consolidated Memories → Reflection → New Insight Memories
    ↑                                                                      |
    └──────────────────────────────────────────────────────────────────────┘
```

### Phases

1. **NREM Phase**: Memory reactivation and strengthening
2. **REM Phase**: Pattern extraction and theme identification
3. **Consolidation**: Integration of related memories
4. **Reflection**: Post-dream analysis and insight generation
5. **Storage**: New insights stored with dream lineage

## Files

```
ai/memory/
├── dream_manager.py                     # Dream cycle orchestration
├── dream_reflection_integration.py      # Reflection integration
└── DREAM_REFLECTION_GUIDE.md            # This documentation
```

## Quick Start

```python
from ai.memory.dream_manager import DreamManager, DreamManagerConfig
from ai.memory.local_foresight_manager import LocalForesightMemoryManager

# Initialize
memory_manager = LocalForesightMemoryManager()
config = DreamManagerConfig(
    nrem_duration=60,  # seconds
    rem_duration=90,
    enable_reflection_integration=True,
)

dream_manager = DreamManager(memory_manager, config)

# Start dream cycle
result = await dream_manager.start_dream_cycle(
    user_id="user_123",
    memories=[...],  # Recent memories to process
)

print(f"Themes: {result.themes}")
print(f"Patterns: {result.patterns}")
print(f"Reflection triggered: {result.reflection_triggered}")
```

## Architecture

### DreamManager

Orchestrates the complete dream cycle:

```python
from ai.memory.dream_manager import DreamManager

async with DreamManager() as dream_manager:
    # Start full dream cycle
    result = await dream_manager.start_dream_cycle(
        user_id="user_123"
    )
    
    # Check status
    status = await dream_manager.get_dream_status(result.dream_id)
```

### DreamReflectionIntegration

Handles post-dream reflection triggering:

```python
from ai.memory.dream_reflection_integration import (
    DreamReflectionIntegration,
    DreamReflectionConfig,
    create_dream_output,
)

config = DreamReflectionConfig(
    post_dream_delay_minutes=5,
    enable_post_dream_reflection=True,
)

integration = DreamReflectionIntegration(config=config)

# Trigger reflection after dream
dream_output = create_dream_output(
    user_id="user_123",
    themes=["theme1", "theme2"],
    patterns=["pattern1"],
    emotional_tone="anxious",
)

await integration.trigger_post_dream_reflection(
    user_id="user_123",
    dream_output=dream_output,
)
```

## Configuration

### DreamManagerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nrem_duration` | 60 | NREM phase duration (seconds) |
| `rem_duration` | 90 | REM phase duration (seconds) |
| `post_dream_delay` | 5 | Delay before reflection (seconds) |
| `min_memories_for_dream` | 5 | Minimum memories to trigger dream |
| `max_dream_themes` | 5 | Maximum themes to extract |
| `max_dream_patterns` | 3 | Maximum patterns to extract |
| `enable_reflection_integration` | True | Enable post-dream reflection |
| `store_dream_lineage` | True | Track dream lineage in memories |

### DreamReflectionConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `post_dream_delay_minutes` | 5 | Delay before triggering reflection |
| `max_reflection_timeout_minutes` | 30 | Max wait time for reflection |
| `enable_post_dream_reflection` | True | Enable feature |
| `enable_dream_lineage_tracking` | True | Track dream lineage |
| `min_dream_confidence` | 0.5 | Minimum confidence threshold |
| `max_reflection_topics` | 5 | Max reflection topics |
| `store_dream_lineage` | True | Store lineage metadata |
| `lineage_depth` | 3 | Generations to track |

## Data Flow

### 1. Dream Cycle Input

```python
memories = [
    {"content": "...", "category": "therapeutic_insight"},
    {"content": "...", "category": "emotional_state"},
    # ... recent memories
]
```

### 2. NREM Phase Output

```python
nrem_result = {
    "reactivated_memories": [...],
    "reactivation_count": 10,
}
```

### 3. REM Phase Output

```python
rem_result = {
    "themes": ["anxiety", "work_stress"],
    "patterns": ["recurring_therapeutic_insight"],
    "emotional_tone": "anxious",
}
```

### 4. Consolidation Output

```python
consolidated = [
    {
        "content": "...",
        "consolidated": True,
        "consolidation_time": "2026-05-11T04:00:00Z",
        "dream_consolidated": True,
    }
]
```

### 5. Reflection Output

```python
insights = [
    {
        "insight_id": "insight_abc123",
        "content": "Dream theme 'anxiety' suggests processing...",
        "category": "therapeutic_insight",
        "dream_lineage": "dream_xyz789",
    }
]
```

## Dream Lineage Tracking

When `store_dream_lineage` is enabled, memories are tagged with their dream origin:

```python
memory_metadata = {
    "tags": [
        "dream:abc123def456",
        "dream_lineage:abc123def456",
        "theme:anxiety",
        "theme:work_stress",
    ]
}
```

This enables querying memories by their dream cycle:

```python
# Find all memories from a specific dream cycle
memories = await memory_manager.search_memories(
    query="dream_lineage:abc123def456",
    user_id="user_123",
)
```

## Timing Coordination

### Default Timing

```\
T+0s:  Dream cycle starts
T+60s: NREM phase completes
T+150s: REM phase completes
T+155s: Consolidation completes
T+160s: Reflection triggered (5 min delay)
T+190s: Reflection completes
```

### Custom Timing

```python
config = DreamManagerConfig(
    nrem_duration=300,    # 5 minutes
    rem_duration=600,     # 10 minutes
    post_dream_delay=300, # 5 minutes
)
```

## Reflection Insight Categories

Insights are categorized by `MemoryCategory`:

| Category | Description |
|----------|-------------|
| `therapeutic_insight` | Clinical insights from dream content |
| `emotional_state` | Emotional processing results |
| `preference` | User preferences discovered |
| `session_summary` | Summaries of dream themes |
| `general` | General insights |

## Monitoring

### Dream Cycle Status

```python
status = await dream_manager.get_dream_status(dream_id)
# {
#     "status": "completed",
#     "phases": {
#         "nrem_completed": true,
#         "rem_completed": true,
#         "consolidation_completed": true,
#         "reflection_triggered": true
#     }
# }
```

### Reflection Status

```python
status = await reflection_integration.get_reflection_status(dream_id)
# {
#     "status": "completed",
#     "insights": [...]
# }
```

## Troubleshooting

### Dream cycle not triggering

**Cause**: Insufficient memories

**Solution**: Ensure at least `min_memories_for_dream` memories exist

### Reflection not triggering after dream

**Cause**: Feature disabled or low confidence

**Solution**: Check `enable_post_dream_reflection` and `min_dream_confidence` config

### Lineage tracking not working

**Cause**: `store_dream_lineage` disabled

**Solution**: Enable in `DreamReflectionConfig`

## Best Practices

1. **Run dream cycles during low-activity periods** - Similar to sleep, dreams work best when the system is idle
2. **Allow sufficient delay before reflection** - Give time for consolidation to complete
3. **Monitor dream lineage** - Track how insights propagate through dream cycles
4. **Adjust timing based on load** - Longer durations for larger memory sets
5. **Enable reflection integration** - Critical for closing the insight loop

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-05-11 | Initial implementation |
