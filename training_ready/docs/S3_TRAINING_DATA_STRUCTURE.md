# S3 Training Data Structure - The Training Mecca

**Last Updated**: 2025-12-11  
**Purpose**: Canonical reference for S3 training dataset organization  
**Status**: S3 is the single source of truth for all training data

## Architecture Overview

```
Google Drive (Source/Staging)
    ↓ [rclone sync]
S3 Buckets (Training Mecca - Canonical)
    ↓ [Training Scripts Read From]
Model Training
```

**Key Principle**: All training data flows through S3. Google Drive is a staging area that syncs to S3. Training scripts read from S3, not Google Drive or local files.

---

## S3 Bucket Structure

### Primary Training Data Bucket: `pixelated-training-data` (OVH)

```
s3://pixelated-training-data/
├── acquired/                          # Locally acquired datasets
│   ├── cot_reasoning.json
│   └── mental_health_counseling.json
│
├── gdrive/                            # Google Drive datasets (synced)
│   ├── raw/                           # Raw Google Drive structure (legacy)
│   │   ├── CoT_Reasoning_Clinical_Diagnosis_Mental_Health/
│   │   ├── CoT_Heartbreak_and_Breakups_downloaded.json
│   │   ├── mental_health_counseling_conversations/
│   │   ├── therapist-sft-format/
│   │   ├── Psych8k/
│   │   └── datasets-wendy/
│   │       ├── priority_1_FINAL.jsonl
│   │       ├── priority_2_FINAL.jsonl
│   │       └── priority_3_FINAL.jsonl
│   │
│   └── processed/                     # Organized structure (canonical)
│       ├── cot_reasoning/
│       │   ├── clinical_diagnosis_mental_health.json
│       │   ├── heartbreak_and_breakups.json
│       │   ├── neurodivergent_vs_neurotypical.json
│       │   ├── mens_mental_health.json
│       │   ├── cultural_nuances.json
│       │   ├── philosophical_understanding.json
│       │   └── temporal_reasoning.json
│       │
│       ├── professional_therapeutic/
│       │   ├── mental_health_counseling/
│       │   ├── soulchat2.0/
│       │   ├── counsel_chat/
│       │   ├── llama3_mental_counseling/
│       │   ├── therapist_sft/
│       │   ├── neuro_qa_sft/
│       │   └── psych8k/
│       │
│       ├── priority/
│       │   ├── priority_1_FINAL.jsonl
│       │   ├── priority_2_FINAL.jsonl
│       │   └── priority_3_FINAL.jsonl
│       │
│       └── edge_cases/
│           └── raw/
│               └── reddit/
│
├── lightning/                         # Lightning.ai training data
│   ├── expert_therapeutic.json
│   ├── expert_empathetic.json
│   ├── expert_practical.json
│   ├── expert_educational.json
│   └── train.json
│
├── voice/                             # Voice/persona training data
│   ├── tim_fletcher_voice_profile.json
│   └── synthetic_conversations.json
│
├── pixel_voice/                       # Pixel Voice pipeline output
│
├── edge_cases/                        # Edge case scenarios
│
├── dual_persona/                      # Dual persona training
│
└── config/                            # Training configurations
    ├── dataset_registry.json
    └── moe_training_config.json
```

### Checkpoint Bucket: `pixelated-checkpoints` (OVH)

```
s3://pixelated-checkpoints/
├── foundation/
│   └── final/
├── reasoning/
│   └── final/
├── voice/
│   └── final/
└── final_model/
```

---

## Data Flow: Google Drive → S3

### Current Sync Process

**Active Uploads** (per `.notes/markdown/one.md`):
- `rclone copy gdrive:datasets ovh:pixel-data/datasets/gdrive/raw ...`
- Status: `processed` tier is DONE. `raw` tier is IN PROGRESS.
- Log: `upload_raw_final.log`

### Sync Scripts

1. **Background rclone sync** (stable, low-priority):
   ```bash
   # Running in tmux session
   rclone copy gdrive:datasets ovh:pixel-data/datasets/gdrive/raw \
     --transfers=2 \
     --progress
   ```

2. **OVH sync script** (`ai/training_ready/platforms/ovh/sync-datasets.sh`):
   ```bash
   ./ai/training_ready/platforms/ovh/sync-datasets.sh upload
   # Syncs local + Google Drive datasets to S3
   ```

### Sync Strategy

**Two-Tier Approach**:
1. **Raw Tier** (`gdrive/raw/`): Direct mirror of Google Drive structure (for backup/reference)
2. **Processed Tier** (`gdrive/processed/`): Organized canonical structure (for training)

**Migration Path**:
- Current: Google Drive → `s3://pixelated-training-data/gdrive/raw/` (in progress)
- Future: Process and organize → `s3://pixelated-training-data/gdrive/processed/` (canonical)

---

## Training Script Access Pattern

### S3-First Access (Recommended)

```python
import boto3
from pathlib import Path

def get_s3_dataset_path(dataset_name: str, category: str = None) -> str:
    """
    Get S3 path for dataset - S3 is the canonical source.
    
    Args:
        dataset_name: Name of the dataset
        category: Optional category (cot_reasoning, professional_therapeutic, etc.)
    
    Returns:
        S3 path (s3://bucket/path)
    """
    bucket = "pixelated-training-data"
    
    # Try processed/canonical structure first
    if category:
        return f"s3://{bucket}/gdrive/processed/{category}/{dataset_name}"
    
    # Fallback to raw structure
    return f"s3://{bucket}/gdrive/raw/{dataset_name}"


def load_dataset_from_s3(s3_path: str, local_cache: Path = None):
    """
    Load dataset from S3 with optional local caching.
    
    Args:
        s3_path: S3 path (s3://bucket/key)
        local_cache: Optional local cache path
    
    Returns:
        Dataset data
    """
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    
    # Check local cache first
    if local_cache and local_cache.exists():
        return json.load(local_cache)
    
    # Download from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj['Body'].read())
    
    # Cache locally if requested
    if local_cache:
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(local_cache, 'w') as f:
            json.dump(data, f)
    
    return data
```

### Example Usage in Training Scripts

```python
# Load CoT reasoning dataset from S3
cot_path = get_s3_dataset_path(
    "clinical_diagnosis_mental_health.json",
    category="cot_reasoning"
)
dataset = load_dataset_from_s3(cot_path)

# Load priority dataset from S3
priority_path = get_s3_dataset_path(
    "priority_1_FINAL.jsonl",
    category="priority"
)
priority_data = load_dataset_from_s3(priority_path)
```

---

## S3 Dataset Organization by Training Stage

### Stage 1: Foundation (`stage1_foundation`)
**S3 Paths**:
- `s3://pixelated-training-data/gdrive/processed/professional_therapeutic/`
- `s3://pixelated-training-data/gdrive/processed/priority/`
- `s3://pixelated-training-data/acquired/mental_health_counseling.json`

**Purpose**: Natural therapeutic dialogue patterns

### Stage 2: Therapeutic Expertise (`stage2_therapeutic_expertise`)
**S3 Paths**:
- `s3://pixelated-training-data/gdrive/processed/cot_reasoning/`
- `s3://pixelated-training-data/acquired/cot_reasoning.json`

**Purpose**: Clinical reasoning patterns (Chain of Thought)

### Stage 3: Edge Stress Test (`stage3_edge_stress_test`)
**S3 Paths**:
- `s3://pixelated-training-data/gdrive/processed/edge_cases/raw/reddit/`
- `s3://pixelated-training-data/edge_cases/`

**Purpose**: Crisis scenarios and edge cases

### Stage 4: Voice Persona (`stage4_voice_persona`)
**S3 Paths**:
- `s3://pixelated-training-data/voice/tim_fletcher_voice_profile.json`
- `s3://pixelated-training-data/voice/synthetic_conversations.json`
- `s3://pixelated-training-data/pixel_voice/`

**Purpose**: Tim Fletcher teaching style/personality

---

## Consolidation Strategy for S3

### Current State
- Google Drive datasets syncing to `gdrive/raw/` (in progress)
- Some organized data in `gdrive/processed/` (target structure)
- Local acquired datasets in `acquired/`
- Platform-specific data in `lightning/`, `voice/`, etc.

### Target Canonical Structure

All training data should be organized under:
```
s3://pixelated-training-data/
├── gdrive/processed/          # Canonical organized structure (primary)
│   ├── cot_reasoning/
│   ├── professional_therapeutic/
│   ├── priority/
│   └── edge_cases/
│
├── gdrive/raw/                # Raw backup (reference only)
│
├── acquired/                  # Locally acquired (small datasets)
│
└── [platform-specific]/      # Lightning, voice, etc.
```

### Migration Plan

1. **Complete raw sync** (in progress):
   - Finish syncing all Google Drive datasets to `gdrive/raw/`

2. **Process and organize**:
   - Process raw datasets into canonical structure
   - Upload to `gdrive/processed/` with organized paths

3. **Update training scripts**:
   - Point all training scripts to S3 paths
   - Use `gdrive/processed/` as primary source
   - Fallback to `gdrive/raw/` if needed

4. **Update dataset registry**:
   - Update `dataset_registry.json` with S3 paths as canonical
   - Mark Google Drive paths as "source" only

---

## Access Methods

### Using OVH CLI (ovhai)

```bash
# List S3 contents
ovhai bucket list pixelated-training-data

# Download dataset
ovhai data pull pixelated-training-data gdrive/processed/cot_reasoning/ ./local/

# Upload processed data
ovhai data push pixelated-training-data ./processed/ gdrive/processed/
```

### Using boto3 (Python)

```python
import boto3

s3 = boto3.client('s3', 
    endpoint_url='https://s3.us-east-va.cloud.ovh.us',
    aws_access_key_id=os.getenv('OVH_S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('OVH_S3_SECRET_KEY')
)

# List datasets
response = s3.list_objects_v2(
    Bucket='pixelated-training-data',
    Prefix='gdrive/processed/cot_reasoning/'
)

# Download dataset
s3.download_file(
    'pixelated-training-data',
    'gdrive/processed/cot_reasoning/clinical_diagnosis_mental_health.json',
    './local/dataset.json'
)
```

### Using rclone

```bash
# Configure OVH S3 remote
rclone config
# Name: ovh
# Type: s3
# Provider: Other
# Endpoint: s3.us-east-va.cloud.ovh.us
# Access Key ID: [your key]
# Secret Access Key: [your secret]

# List S3 contents
rclone lsd ovh:pixelated-training-data/gdrive/processed/

# Download dataset
rclone copy ovh:pixelated-training-data/gdrive/processed/cot_reasoning/ ./local/
```

---

## Integration with Training Scripts

### Updated Pattern (S3-First)

```python
def get_training_dataset_path(dataset_name: str, stage: str) -> str:
    """
    Get dataset path - S3 is canonical, with fallbacks.
    
    Priority:
    1. S3 processed/canonical structure
    2. S3 raw structure
    3. Local cache (if exists)
    4. Download from S3 to cache
    """
    bucket = "pixelated-training-data"
    
    # Map stage to S3 category
    stage_to_category = {
        "stage1_foundation": "professional_therapeutic",
        "stage2_therapeutic_expertise": "cot_reasoning",
        "stage3_edge_stress_test": "edge_cases",
        "stage4_voice_persona": "voice"
    }
    
    category = stage_to_category.get(stage)
    
    # Try canonical processed structure
    s3_path = f"s3://{bucket}/gdrive/processed/{category}/{dataset_name}"
    if s3_object_exists(s3_path):
        return s3_path
    
    # Fallback to raw structure
    s3_path = f"s3://{bucket}/gdrive/raw/{dataset_name}"
    if s3_object_exists(s3_path):
        return s3_path
    
    # Fallback to acquired
    s3_path = f"s3://{bucket}/acquired/{dataset_name}"
    if s3_object_exists(s3_path):
        return s3_path
    
    raise FileNotFoundError(f"Dataset {dataset_name} not found in S3")
```

---

## Related Documentation

- **S3 Execution Order**: `ai/training_ready/docs/S3_EXECUTION_ORDER.md`
- **Google Drive Structure**: `ai/training_ready/docs/GDRIVE_STRUCTURE.md` (source/staging)
- **Dataset Registry**: `ai/data/dataset_registry.json` (should reference S3 paths)
- **Sync Scripts**: `ai/training_ready/platforms/ovh/sync-datasets.sh`

---

## Notes

- **S3 is canonical** - All training scripts should read from S3
- **Google Drive is source** - Syncs to S3 via rclone, not used directly for training
- **Local is cache only** - Local files are temporary caches, not source of truth
- **Two-tier S3 structure** - `raw/` for backup, `processed/` for training
