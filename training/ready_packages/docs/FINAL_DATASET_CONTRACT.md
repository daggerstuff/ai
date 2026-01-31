# Final Training Dataset Contract

**Last Updated**: 2025-12-12  
**Purpose**: Canonical specification for the final training dataset artifact  
**Status**: Contract definition for final dataset compilation

## Overview

The final training dataset is a **dual artifact**:
1. **Manifest**: JSON file pointing to sharded datasets in S3 with hashes and provenance
2. **Compiled Export**: Single ChatML JSONL file for portability

Both artifacts are stored in S3 as the canonical source of truth.

---

## Dataset Schema

### ChatML Format (Compiled Export)

Each line in the compiled JSONL export follows this schema:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a therapeutic AI assistant..."
    },
    {
      "role": "user",
      "content": "I've been struggling with..."
    },
    {
      "role": "assistant",
      "content": "I understand that you're going through..."
    }
  ],
  "metadata": {
    "source_family": "long_running_therapy",
    "source_key": "s3://pixel-data/gdrive/processed/professional_therapeutic/therapist_sft/...",
    "content_hash": "sha256:abc123...",
    "pii_status": "scrubbed",
    "license_tag": "therapeutic_license",
    "split": "train",
    "phase": "stage1_foundation",
    "conversation_length": 15,
    "total_tokens": 2847,
    "provenance": {
      "original_s3_key": "datasets/gdrive/raw/therapist-sft-format/...",
      "processing_pipeline": "integrated_training_pipeline_v1.0",
      "processed_at": "2025-12-12T14:00:00Z",
      "dedup_status": "unique"
    }
  }
}
```

### Required Fields

- **messages**: Array of ChatML message objects (system/user/assistant)
- **metadata.source_family**: One of the required dataset families (see coverage report)
- **metadata.source_key**: S3 path to original source data
- **metadata.content_hash**: SHA256 hash of normalized conversation text (for dedup)
- **metadata.pii_status**: `"scrubbed"`, `"none_detected"`, or `"requires_review"`
- **metadata.license_tag**: License identifier for the source data
- **metadata.split**: `"train"`, `"val"`, or `"test"`
- **metadata.phase**: Training phase this data belongs to (`stage1_foundation`, `stage2_therapeutic_expertise`, etc.)
- **metadata.provenance**: Immutable tracking of data origin and processing

### Optional Fields

- **metadata.conversation_length**: Number of turns in conversation
- **metadata.total_tokens**: Approximate token count
- **metadata.quality_score**: Quality assessment score (0-1)
- **metadata.bias_flags**: Array of detected bias categories (if any)

---

## Manifest Schema

The manifest is a JSON file that provides an index to all sharded datasets:

```json
{
  "manifest_version": "1.0",
  "generated_at": "2025-12-12T14:00:00Z",
  "total_conversations": 608497,
  "total_tokens_approx": 1250000000,
  "splits": {
    "train": {
      "conversations": 547647,
      "shards": [
        {
          "shard_id": "train_000",
          "s3_path": "s3://pixel-data/final_dataset/train/train_000.jsonl",
          "size_bytes": 1048576000,
          "sha256": "abc123...",
          "conversation_count": 10000,
          "source_families": ["mental_health_datasets", "long_running_therapy"]
        }
      ]
    },
    "val": {
      "conversations": 30425,
      "shards": [...]
    },
    "test": {
      "conversations": 30425,
      "shards": [...]
    }
  },
  "source_families": {
    "edge_case_generator": {
      "conversations": 5000,
      "splits": {"train": 4000, "val": 500, "test": 500}
    },
    "long_running_therapy": {
      "conversations": 25000,
      "splits": {"train": 20000, "val": 2500, "test": 2500}
    }
  },
  "provenance_map": {
    "sha256:abc123...": {
      "source_key": "s3://pixel-data/gdrive/processed/...",
      "source_family": "mental_health_datasets",
      "original_format": "json",
      "processing_steps": ["encoding_fix", "dedup", "chatml_convert"]
    }
  },
  "holdout_families": {
    "long_running_therapy": {
      "test_split_only": true,
      "rationale": "Hard holdout for evaluation"
    },
    "edge_case_crisis": {
      "test_split_only": true,
      "rationale": "Hard holdout for crisis handling evaluation"
    },
    "sarcasm": {
      "test_split_only": true,
      "rationale": "Hard holdout for sarcasm detection evaluation"
    },
    "voice_persona": {
      "test_split_only": true,
      "rationale": "Hard holdout for voice consistency evaluation"
    }
  }
}
```

---

## Split Rules

### Standard Split (90/5/5)

- **Train**: 90% of conversations
- **Val**: 5% of conversations
- **Test**: 5% of conversations

### Hard Holdout Families

These families are **excluded from train/val** and only appear in test:

1. **long_running_therapy**: Long-running therapy sessions (actual therapy)
2. **edge_case_crisis**: Crisis-level edge cases
3. **sarcasm**: Sarcasm detection examples
4. **voice_persona**: Voice/persona transcripts

**Rationale**: These are evaluation-critical and must not leak into training.

### Split Leakage Prevention

- **Exact duplicates**: Same content_hash cannot appear in multiple splits
- **Near-duplicates**: Semantic similarity > 0.95 cannot cross split boundaries
- **Source family isolation**: Hard holdout families are completely isolated to test

---

## Provenance Tracking

Every conversation in the final dataset must have **immutable provenance**:

1. **Original S3 key**: Where the data came from
2. **Source family**: Which dataset family it belongs to
3. **Processing pipeline**: Which pipeline processed it
4. **Processing timestamp**: When it was processed
5. **Processing steps**: List of transformations applied (encoding_fix, dedup, chatml_convert, etc.)

This enables:
- **Reproducibility**: Rebuild dataset from source
- **Auditability**: Trace any conversation back to origin
- **Quality tracking**: Identify problematic sources

---

## Content Hash Algorithm

For deduplication and provenance:

```python
import hashlib
import json

def compute_content_hash(messages: list[dict]) -> str:
    """Compute SHA256 hash of normalized conversation content"""
    # Extract all content from messages
    content_parts = []
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content_parts.append(msg['content'].strip().lower())
    
    # Normalize: lowercase, strip whitespace, sort
    normalized = ' '.join(sorted(content_parts))
    
    # Compute hash
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
```

---

## PII Handling

All conversations must be tagged with PII status:

- **scrubbed**: PII detected and removed
- **none_detected**: No PII found
- **requires_review**: PII detection uncertain, needs human review

**Hard constraint**: Conversations with `requires_review` status must be excluded from final dataset until reviewed.

---

## License Tags

Each conversation must have a license tag indicating source license:

- `therapeutic_license`: Professional therapeutic datasets
- `reddit_cc`: Reddit data (Creative Commons)
- `synthetic`: Synthetically generated
- `public_domain`: Public domain content
- `custom`: Custom license (specify in metadata)

---

## Canonical S3 Locations

### Manifest
```
s3://pixel-data/final_dataset/manifest.json
```

### Compiled Export
```
s3://pixel-data/final_dataset/compiled/final_training_dataset.jsonl
```

### Sharded Datasets
```
s3://pixel-data/final_dataset/train/train_000.jsonl
s3://pixel-data/final_dataset/train/train_001.jsonl
...
s3://pixel-data/final_dataset/val/val_000.jsonl
...
s3://pixel-data/final_dataset/test/test_000.jsonl
...
```

---

## Validation Rules

Before final dataset is considered complete:

1. **Coverage gate**: All required families present (or explicitly waived)
2. **Leakage gate**: No near-duplicates across splits for holdout families
3. **PII gate**: No `requires_review` conversations in final dataset
4. **Provenance gate**: All conversations have complete provenance
5. **Hash gate**: All conversations have valid content_hash
6. **Split gate**: Hard holdout families only in test split
7. **Stats gate**: Distribution report generated (counts/tokens by family, phase, split)

---

## Related Documentation

- **Coverage Report**: `ai/training_ready/data/dataset_coverage_report.json`
- **S3 Structure**: `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md`
- **Deduplication**: `ai/training_ready/data/FULL_DEDUPLICATION_SUMMARY.md`
