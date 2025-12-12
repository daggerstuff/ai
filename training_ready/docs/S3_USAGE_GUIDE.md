# S3 Training Data Usage Guide

**Purpose**: How to use S3 (the training mecca) in training scripts  
**Status**: S3 is canonical - all training data should be loaded from S3

## Quick Start

### Basic Usage

```python
from ai.training_ready.utils.s3_dataset_loader import load_dataset_from_s3

# Load dataset from S3 (automatic path resolution)
data = load_dataset_from_s3(
    dataset_name="clinical_diagnosis_mental_health.json",
    category="cot_reasoning"
)

# Use in training
conversations = data['conversations']
```

### With Local Caching

```python
from pathlib import Path
from ai.training_ready.utils.s3_dataset_loader import load_dataset_from_s3

# Load with local cache (downloads once, reuses cache)
data = load_dataset_from_s3(
    dataset_name="priority_1_FINAL.jsonl",
    category="priority",
    cache_local=Path("./cache/priority_1.jsonl")
)
```

### Direct S3 Path

```python
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader()
data = loader.load_json("s3://pixelated-training-data/gdrive/processed/cot_reasoning/clinical_diagnosis_mental_health.json")
```

---

## Environment Setup

### Required Environment Variables

The loader automatically loads from `.env` file (project root or `ai/.env`).

**OVH S3 credentials** (preferred):
```bash
# In .env file or environment
OVH_S3_ACCESS_KEY="your_access_key"
OVH_S3_SECRET_KEY="your_secret_key"
OVH_S3_ENDPOINT="https://s3.us-east-va.cloud.ovh.us"  # Optional, has default
OVH_S3_REGION="us-east-va"  # Optional, has default
```

**Fallback options** (for compatibility):
- `OVH_ACCESS_KEY` / `OVH_SECRET_KEY`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`

### Install Dependencies

```bash
uv pip install boto3
# or
pip install boto3
```

---

## S3 Path Resolution

The loader automatically resolves paths in this order:

1. **Canonical processed structure**: `s3://pixelated-training-data/gdrive/processed/{category}/{dataset_name}`
2. **Raw structure**: `s3://pixelated-training-data/gdrive/raw/{dataset_name}`
3. **Acquired datasets**: `s3://pixelated-training-data/acquired/{dataset_name}`

### Category Mapping

| Training Stage | Category | S3 Path |
|---------------|----------|---------|
| Stage 1: Foundation | `professional_therapeutic` | `gdrive/processed/professional_therapeutic/` |
| Stage 2: Expertise | `cot_reasoning` | `gdrive/processed/cot_reasoning/` |
| Stage 3: Edge Cases | `edge_cases` | `gdrive/processed/edge_cases/` |
| Stage 4: Voice | `voice` | `voice/` |
| Priority | `priority` | `gdrive/processed/priority/` |

---

## Updating Training Scripts

### Before (Local File)

```python
# ❌ Old pattern - local file
with open('training_dataset.json', 'r') as f:
    data = json.load(f)
```

### After (S3-First)

```python
# ✅ New pattern - S3 canonical
from ai.training_ready.utils.s3_dataset_loader import load_dataset_from_s3

data = load_dataset_from_s3(
    dataset_name="training_dataset.json",
    category="professional_therapeutic"
)
```

### With Fallback (Backward Compatible)

```python
# ✅ S3-first with local fallback
from ai.training_ready.utils.s3_dataset_loader import load_dataset_from_s3
from pathlib import Path

def load_training_data(dataset_name: str, category: str = None):
    try:
        # Try S3 first (canonical)
        return load_dataset_from_s3(dataset_name, category=category)
    except FileNotFoundError:
        # Fallback to local (for development/testing)
        local_path = Path(f"./data/{dataset_name}")
        if local_path.exists():
            with open(local_path, 'r') as f:
                return json.load(f)
        raise
```

---

## Streaming Large Datasets

For large JSONL files, use streaming to avoid loading everything into memory:

```python
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader()

# Stream JSONL line by line
for conversation in loader.stream_jsonl("s3://pixelated-training-data/gdrive/processed/cot_reasoning/clinical_diagnosis_mental_health.jsonl"):
    # Process one conversation at a time
    process_conversation(conversation)
```

---

## Integration with Training Scripts

### Updated `train_optimized.py` Pattern

```python
def analyze_dataset(s3_path: str = None, dataset_path: str = None):
    """Analyze dataset - S3 is preferred"""
    if s3_path:
        from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
        loader = S3DatasetLoader()
        data = loader.load_json(s3_path)
    elif dataset_path:
        # Fallback to local
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    else:
        # Auto-detect from S3
        data = load_dataset_from_s3('training_dataset.json', category='professional_therapeutic')
    
    # ... rest of analysis
```

### Updated `train_moe_h100.py` Pattern

```python
def load_training_data(s3_path: str = None, dataset_path: str = None):
    """Load training data - S3 is canonical"""
    if s3_path:
        from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
        loader = S3DatasetLoader()
        data = loader.load_json(s3_path)
    elif dataset_path:
        # Fallback to local
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    else:
        # Auto-detect from S3
        data = load_dataset_from_s3('training_dataset.json', category='professional_therapeutic')
    
    # ... rest of loading
```

---

## Listing Available Datasets

```python
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader()

# List all datasets in processed structure
datasets = loader.list_datasets(prefix="gdrive/processed/")
for dataset_path in datasets:
    print(dataset_path)
```

---

## Error Handling

```python
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader, FileNotFoundError

try:
    loader = S3DatasetLoader()
    data = loader.load_json("s3://pixelated-training-data/gdrive/processed/cot_reasoning/dataset.json")
except FileNotFoundError:
    print("Dataset not found in S3")
except Exception as e:
    print(f"Error loading from S3: {e}")
```

---

## Performance Tips

1. **Use local caching** for frequently accessed datasets
2. **Stream JSONL** for large files instead of loading all at once
3. **Check object existence** before loading if unsure
4. **Use canonical paths** (`gdrive/processed/`) for best performance

---

## Verification

### Quick Test

```bash
cd ai/training_ready
python scripts/verify_s3_access.py
```

This will:
- Test S3 connection
- Verify credentials are loaded from `.env`
- List available datasets
- Show connection status

### Manual Test

```python
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader()
datasets = loader.list_datasets(prefix="gdrive/processed/")
print(f"Found {len(datasets)} datasets")
```

---

## Related Documentation

- **S3 Structure**: `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md` - Complete S3 organization
- **S3 Execution Order**: `ai/training_ready/docs/S3_EXECUTION_ORDER.md` - S3-first workflow
- **Dataset Registry**: `ai/data/dataset_registry.json` - Dataset catalog with S3 paths
- **Verification Script**: `ai/training_ready/scripts/verify_s3_access.py` - Test S3 access
