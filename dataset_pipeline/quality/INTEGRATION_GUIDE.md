# Quality Scoring v1 Integration Guide

**Status**: ✅ Integrated  
**Issue**: KAN-12  
**Date**: 2025-12-18

---

## Overview

Quality Scoring v1 has been integrated into the dataset pipeline at multiple points to provide comprehensive quality assessment and filtering capabilities.

---

## Integration Points

### 1. Pipeline Orchestrator ✅

**File**: `ai/dataset_pipeline/orchestration/pipeline_orchestrator.py`

**Integration**: Quality scoring is integrated into the `_validate_quality` method (Stage 5: Quality Validation).

**How it works**:
- Quality Scoring v1 is initialized in `__init__` (if enabled)
- During quality validation, each conversation is scored using Quality Scoring v1
- Filtering uses three-tier decision system: accept/curate/reject
- Falls back to acquisition monitor if Quality Scoring v1 unavailable

**Configuration**:
```python
config = PipelineConfig(
    enable_quality_scoring_v1=True,  # Enable Quality Scoring v1
    quality_scoring_config="scripts/quality_scoring/config.example.json",  # Optional config
    quality_threshold=0.7,  # Minimum composite score threshold
)
```

### 2. Unified Preprocessing Pipeline ✅

**File**: `ai/dataset_pipeline/unified_preprocessing_pipeline.py`

**Integration**: Quality scoring is integrated into the `enhance_record` method.

**How it works**:
- When enhancing records, Quality Scoring v1 scores are computed
- Scores stored in `metadata.quality_score` and `metadata.quality_scoring_v1`
- Falls back to `estimate_quality_score` if Quality Scoring v1 unavailable

**Metadata Structure**:
```json
{
  "metadata": {
    "quality_score": 0.75,
    "quality_scoring_v1": {
      "signals": {
        "empathy": 0.8,
        "fidelity": 0.9,
        "domain": 0.7,
        "harm": 0.0
      },
      "decision": "accept"
    }
  }
}
```

### 3. Standalone Quality Filter ✅

**File**: `ai/dataset_pipeline/quality/quality_filter_v1.py`

**Component**: `QualityFilterV1`

**Usage**:
```python
from ai.dataset_pipeline.quality.quality_filter_v1 import QualityFilterV1

# Create filter
filter = QualityFilterV1(
    min_decision="curate",
    min_composite=0.6,
    enabled=True
)

# Filter single conversation
if filter.filter(conversation):
    # Process conversation
    pass

# Filter batch
filtered, results = filter.filter_batch(conversations)
```

---

## Usage Examples

### In Pipeline Orchestrator

```python
from ai.dataset_pipeline.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
)

# Configure with Quality Scoring v1
config = PipelineConfig(
    enable_quality_scoring_v1=True,
    quality_threshold=0.65,  # Minimum composite score
)

orchestrator = PipelineOrchestrator(config)

# Execute pipeline - Quality Scoring v1 will be used automatically
result = await orchestrator.execute_pipeline()
```

### Direct Usage

```python
from ai.dataset_pipeline.quality.quality_scoring_v1 import QualityScoringV1

# Initialize
scoring = QualityScoringV1(
    config_path="scripts/quality_scoring/config.example.json",
    enabled=True
)

# Score a conversation
result = scoring.score_conversation(conversation)
print(f"Decision: {result['decision']}, Composite: {result['composite']:.2f}")

# Filter conversations
filtered, results = scoring.filter_conversations(
    conversations,
    min_decision="accept",
    min_composite=0.7
)
```

### In Preprocessing

The unified preprocessing pipeline automatically uses Quality Scoring v1 when enhancing records. No additional code needed - it's integrated transparently.

---

## Configuration

### Default Weights and Thresholds

```json
{
  "weights": {
    "empathy": 0.25,
    "fidelity": 0.25,
    "domain": 0.25,
    "harm": 0.25
  },
  "thresholds": {
    "harm_max": 0.05,
    "accept_min": 0.60,
    "curate_min": 0.45
  }
}
```

### Custom Configuration

Create a config file at `scripts/quality_scoring/config.json`:

```json
{
  "weights": {
    "empathy": 0.30,
    "fidelity": 0.30,
    "domain": 0.20,
    "harm": 0.20
  },
  "thresholds": {
    "harm_max": 0.03,
    "accept_min": 0.65,
    "curate_min": 0.50
  }
}
```

Then pass the path to the orchestrator:

```python
config = PipelineConfig(
    quality_scoring_config="scripts/quality_scoring/config.json"
)
```

---

## Decision Logic

The quality scoring system uses a three-tier decision system:

1. **Reject**: `harm > harm_max` OR `composite < curate_min`
2. **Curate**: `composite >= curate_min` AND `composite < accept_min` AND `harm <= harm_max`
3. **Accept**: `composite >= accept_min` AND `harm <= harm_max`

**Filtering Behavior**:
- `min_decision="accept"`: Only accepts conversations with decision="accept"
- `min_decision="curate"`: Accepts both "accept" and "curate" decisions
- `min_decision="reject"`: Accepts all (effectively no filtering)

---

## Fallback Behavior

If Quality Scoring v1 components are not available:
- Pipeline orchestrator falls back to `acquisition_monitor.process_conversation`
- Unified preprocessing falls back to `estimate_quality_score`
- Quality filter returns all conversations (pass-through)

This ensures the pipeline continues to function even if Quality Scoring v1 dependencies are missing.

---

## Performance Considerations

- Quality scoring is performed per-conversation during quality validation stage
- Scoring is synchronous (consider async batch processing for large datasets)
- Pattern matching uses compiled regex for efficiency
- Production validators are lazy-loaded (only initialized when needed)

---

## Monitoring

Quality scores are logged and can be tracked via:
- Pipeline metrics (average quality scores)
- Quality validation reports
- Metadata stored with each conversation

---

## Related Documentation

- **Quality Scoring Implementation**: `scripts/quality_scoring/README.md`
- **Quality Scoring Summary**: `scripts/quality_scoring/IMPLEMENTATION_SUMMARY.md`
- **Jira Issue**: KAN-12
- **Confluence Spec**: Ingestion & Quality Scoring child page

---

**Integration Complete** ✅  
Quality Scoring v1 is now integrated into the dataset pipeline and ready for use.
