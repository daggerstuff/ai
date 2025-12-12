# Google Drive Training Dataset Structure (Source/Staging)

**Last Updated**: 2025-12-11  
**Purpose**: Reference for Google Drive dataset organization (source/staging area)  
**Note**: S3 is the canonical training data location. Google Drive syncs to S3 via rclone. See `S3_TRAINING_DATA_STRUCTURE.md` for the training mecca structure.

## Access Methods

### Primary Access (Recommended)
```bash
# Using rclone (most reliable)
rclone lsd gdrive:datasets
rclone copy gdrive:datasets/cot_reasoning/ ./local/cot_reasoning/
```

### Direct Mount (if available)
```bash
# Mount point
/mnt/gdrive/datasets

# List contents
ls /mnt/gdrive/datasets/
```

### Download Scripts
```bash
# Use OVH sync script
./ai/training_ready/platforms/ovh/gdrive-download.sh download-all

# Or interactive mode
./ai/training_ready/platforms/ovh/gdrive-download.sh download-interactive
```

---

## Proposed Canonical Structure

> **Note**: This is the target structure. Current structure may differ. See migration notes below.

```
gdrive:datasets/
├── cot_reasoning/                    # Chain of Thought reasoning datasets
│   ├── clinical_diagnosis_mental_health.json
│   ├── heartbreak_and_breakups.json
│   ├── neurodivergent_vs_neurotypical.json
│   ├── mens_mental_health.json
│   ├── cultural_nuances.json
│   ├── philosophical_understanding.json
│   └── temporal_reasoning.json
│
├── professional_therapeutic/         # Licensed therapist responses
│   ├── mental_health_counseling/
│   ├── soulchat2.0/
│   ├── counsel_chat/
│   ├── llama3_mental_counseling/
│   ├── therapist_sft/                # 406 MB - largest dataset
│   ├── neuro_qa_sft/
│   └── psych8k/
│
├── priority/                         # Top-tier curated conversations
│   ├── priority_1_FINAL.jsonl
│   ├── priority_2_FINAL.jsonl
│   └── priority_3_FINAL.jsonl
│
├── edge_cases/                       # Crisis scenarios and edge cases
│   ├── raw/
│   │   └── reddit/                   # Reddit archives
│   └── processed/                    # Processed edge cases
│
└── training_packages/                # Complete training packages (if any)
    ├── lightning_package/
    └── therapeutic_package/
```

---

## Current vs Proposed Path Mapping

### CoT Reasoning Datasets

| Current Path | Proposed Path | Status |
|-------------|---------------|--------|
| `/mnt/gdrive/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health/CoT_Reasoning_Clinical_Diagnosis_Mental_Health.json` | `gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT_Heartbreak_and_Breakups_downloaded.json` | `gdrive:datasets/cot_reasoning/heartbreak_and_breakups.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json` | `gdrive:datasets/cot_reasoning/neurodivergent_vs_neurotypical.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT_Reasoning_Mens_Mental_Health_downloaded.json` | `gdrive:datasets/cot_reasoning/mens_mental_health.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT-Reasoning_Cultural_Nuances/CoT-Reasoning_Cultural_Nuances_Dataset.json` | `gdrive:datasets/cot_reasoning/cultural_nuances.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT_Philosophical_Understanding/CoT_Philosophical_Understanding.json` | `gdrive:datasets/cot_reasoning/philosophical_understanding.json` | Migration needed |
| `/mnt/gdrive/datasets/CoT_Temporal_Reasoning_Dataset/CoT_Temporal_Reasoning_Dataset.json` | `gdrive:datasets/cot_reasoning/temporal_reasoning.json` | Migration needed |

### Professional Therapeutic Datasets

| Current Path | Proposed Path | Status |
|-------------|---------------|--------|
| `/mnt/gdrive/datasets/mental_health_counseling_conversations` | `gdrive:datasets/professional_therapeutic/mental_health_counseling/` | Migration needed |
| `/mnt/gdrive/datasets/SoulChat2.0` | `gdrive:datasets/professional_therapeutic/soulchat2.0/` | Migration needed |
| `/mnt/gdrive/datasets/counsel-chat` | `gdrive:datasets/professional_therapeutic/counsel_chat/` | Migration needed |
| `/mnt/gdrive/datasets/LLAMA3_Mental_Counseling_Data` | `gdrive:datasets/professional_therapeutic/llama3_mental_counseling/` | Migration needed |
| `/mnt/gdrive/datasets/therapist-sft-format` | `gdrive:datasets/professional_therapeutic/therapist_sft/` | Migration needed |
| `/mnt/gdrive/datasets/neuro_qa_SFT_Trainer` | `gdrive:datasets/professional_therapeutic/neuro_qa_sft/` | Migration needed |
| `/mnt/gdrive/datasets/Psych8k` | `gdrive:datasets/professional_therapeutic/psych8k/` | Migration needed |

### Priority Datasets

| Current Path | Proposed Path | Status |
|-------------|---------------|--------|
| `/mnt/gdrive/datasets/datasets-wendy/priority_1_FINAL.jsonl` | `gdrive:datasets/priority/priority_1_FINAL.jsonl` | Migration needed (rename folder) |
| `/mnt/gdrive/datasets/datasets-wendy/priority_2_FINAL.jsonl` | `gdrive:datasets/priority/priority_2_FINAL.jsonl` | Migration needed (rename folder) |
| `/mnt/gdrive/datasets/datasets-wendy/priority_3_FINAL.jsonl` | `gdrive:datasets/priority/priority_3_FINAL.jsonl` | Migration needed (rename folder) |

### Edge Cases

| Current Path | Proposed Path | Status |
|-------------|---------------|--------|
| `/mnt/gdrive/datasets/reddit` | `gdrive:datasets/edge_cases/raw/reddit/` | Migration needed |

---

## Usage in Training Scripts

### ⚠️ Important: S3 is Canonical

**Training scripts should read from S3, not Google Drive.** Google Drive is a source/staging area that syncs to S3.

See `S3_TRAINING_DATA_STRUCTURE.md` for the recommended S3-first access pattern.

### Google Drive Access (For Sync/Upload Only)

Google Drive access is primarily for:
1. **Syncing to S3** (via rclone)
2. **Uploading new datasets** before they reach S3
3. **Backup/reference** purposes

```python
# For syncing/uploading only - not for training
def sync_gdrive_to_s3():
    """Sync Google Drive datasets to S3"""
    # Use rclone or sync scripts
    # See: ai/training_ready/platforms/ovh/sync-datasets.sh
    pass
```

---

## Migration Guide

### For Scripts Using Old Paths

1. **Update direct path references**:
   ```python
   # Old
   path = "/mnt/gdrive/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health/..."
   
   # New
   path = get_dataset_path("clinical_diagnosis_mental_health", stage="cot_reasoning")
   ```

2. **Update dataset_registry.json**:
   - Update paths to use canonical structure
   - Add migration notes for old paths

3. **Update sync scripts**:
   - `gdrive-download.sh` - Update folder paths
   - `sync-datasets.sh` - Update upload/download paths

### For Google Drive Reorganization

If you have access to reorganize Google Drive:

1. **Create new folder structure**:
   ```bash
   rclone mkdir gdrive:datasets/cot_reasoning
   rclone mkdir gdrive:datasets/professional_therapeutic
   rclone mkdir gdrive:datasets/priority
   rclone mkdir gdrive:datasets/edge_cases/raw
   ```

2. **Move files** (example for CoT):
   ```bash
   # Move CoT datasets
   rclone move gdrive:datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health \
              gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health \
              --transfers=1
   ```

3. **Verify and update references**:
   - Test access to new paths
   - Update `dataset_registry.json`
   - Update training scripts

---

## Dataset Registry Integration

The `ai/data/dataset_registry.json` should reference canonical paths:

```json
{
  "cot_reasoning": {
    "clinical_diagnosis_mental_health": {
      "path": "gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health.json",
      "legacy_paths": [
        "/mnt/gdrive/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health/..."
      ],
      "size_mb": 20,
      "stage": "stage2_therapeutic_expertise"
    }
  }
}
```

---

## Backward Compatibility

Until migration is complete, scripts should support both old and new paths:

```python
def find_dataset(name: str) -> Path:
    """Find dataset with backward compatibility"""
    # Try new canonical path first
    new_path = Path(f"/mnt/gdrive/datasets/cot_reasoning/{name}.json")
    if new_path.exists():
        return new_path
    
    # Fallback to old paths
    old_paths = [
        Path(f"/mnt/gdrive/datasets/CoT_Reasoning_{name}/..."),
        Path(f"/mnt/gdrive/datasets/{name}_downloaded.json"),
        # ... other old patterns
    ]
    
    for path in old_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Dataset {name} not found")
```

---

## Related Documentation

- **S3 Training Mecca**: `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md` - **Canonical S3 structure (read this for training)**
- **Full Audit**: `.notes/markdown/three.md` - Complete Google Drive → S3 consolidation audit
- **Dataset Registry**: `ai/data/dataset_registry.json` - Complete dataset catalog (should reference S3 paths)
- **Download Scripts**: `ai/training_ready/platforms/ovh/gdrive-download.sh` - For staging/syncing
- **Sync Scripts**: `ai/training_ready/platforms/ovh/sync-datasets.sh` - Google Drive → S3 sync
