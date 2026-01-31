# Google Drive Dataset Migration Guide

**Purpose**: Step-by-step guide for reorganizing Google Drive datasets into canonical structure  
**Status**: Planning document - execute when ready

## Prerequisites

1. **rclone configured** with Google Drive access:
   ```bash
   rclone listremotes
   # Should show: gdrive:
   ```

2. **Backup created** (recommended):
   ```bash
   # Create backup before reorganization
   rclone copy gdrive:datasets gdrive:datasets_backup_$(date +%Y%m%d) --progress
   ```

3. **Access verified**:
   ```bash
   rclone lsd gdrive:datasets
   ```

---

## Migration Steps

### Step 1: Create New Folder Structure

```bash
# Create main category folders
rclone mkdir gdrive:datasets/cot_reasoning
rclone mkdir gdrive:datasets/professional_therapeutic
rclone mkdir gdrive:datasets/priority
rclone mkdir gdrive:datasets/edge_cases
rclone mkdir gdrive:datasets/edge_cases/raw
```

### Step 2: Migrate CoT Reasoning Datasets

```bash
# Move each CoT dataset to canonical location
# Use --transfers=1 to avoid rate limits

# Clinical Diagnosis
rclone move gdrive:datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health \
           gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health \
           --transfers=1 --progress

# Heartbreak and Breakups
rclone move gdrive:datasets/CoT_Heartbreak_and_Breakups_downloaded.json \
           gdrive:datasets/cot_reasoning/heartbreak_and_breakups.json \
           --transfers=1 --progress

# Neurodivergent
rclone move gdrive:datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json \
           gdrive:datasets/cot_reasoning/neurodivergent_vs_neurotypical.json \
           --transfers=1 --progress

# Men's Mental Health
rclone move gdrive:datasets/CoT_Reasoning_Mens_Mental_Health_downloaded.json \
           gdrive:datasets/cot_reasoning/mens_mental_health.json \
           --transfers=1 --progress

# Cultural Nuances
rclone move gdrive:datasets/CoT-Reasoning_Cultural_Nuances \
           gdrive:datasets/cot_reasoning/cultural_nuances \
           --transfers=1 --progress

# Philosophical Understanding
rclone move gdrive:datasets/CoT_Philosophical_Understanding \
           gdrive:datasets/cot_reasoning/philosophical_understanding \
           --transfers=1 --progress

# Temporal Reasoning
rclone move gdrive:datasets/CoT_Temporal_Reasoning_Dataset \
           gdrive:datasets/cot_reasoning/temporal_reasoning \
           --transfers=1 --progress
```

### Step 3: Migrate Professional Therapeutic Datasets

```bash
# Create professional_therapeutic folder structure
rclone mkdir gdrive:datasets/professional_therapeutic

# Move each professional dataset
rclone move gdrive:datasets/mental_health_counseling_conversations \
           gdrive:datasets/professional_therapeutic/mental_health_counseling \
           --transfers=1 --progress

rclone move gdrive:datasets/SoulChat2.0 \
           gdrive:datasets/professional_therapeutic/soulchat2.0 \
           --transfers=1 --progress

rclone move gdrive:datasets/counsel-chat \
           gdrive:datasets/professional_therapeutic/counsel_chat \
           --transfers=1 --progress

rclone move gdrive:datasets/LLAMA3_Mental_Counseling_Data \
           gdrive:datasets/professional_therapeutic/llama3_mental_counseling \
           --transfers=1 --progress

rclone move gdrive:datasets/therapist-sft-format \
           gdrive:datasets/professional_therapeutic/therapist_sft \
           --transfers=1 --progress

rclone move gdrive:datasets/neuro_qa_SFT_Trainer \
           gdrive:datasets/professional_therapeutic/neuro_qa_sft \
           --transfers=1 --progress

rclone move gdrive:datasets/Psych8k \
           gdrive:datasets/professional_therapeutic/psych8k \
           --transfers=1 --progress
```

### Step 4: Migrate Priority Datasets

```bash
# Rename datasets-wendy to priority
rclone move gdrive:datasets/datasets-wendy \
           gdrive:datasets/priority \
           --transfers=1 --progress
```

### Step 5: Migrate Edge Cases

```bash
# Move Reddit archives
rclone move gdrive:datasets/reddit \
           gdrive:datasets/edge_cases/raw/reddit \
           --transfers=1 --progress
```

### Step 6: Verify Migration

```bash
# List new structure
rclone tree gdrive:datasets --depth=2

# Verify key datasets exist
rclone ls gdrive:datasets/cot_reasoning/
rclone ls gdrive:datasets/professional_therapeutic/
rclone ls gdrive:datasets/priority/
```

---

## Post-Migration Updates

### 1. Update dataset_registry.json

Update all paths in `ai/data/dataset_registry.json` to use canonical structure:

```json
{
  "cot_reasoning": {
    "clinical_diagnosis_mental_health": {
      "path": "gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health.json",
      "legacy_paths": [
        "gdrive:datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health/..."
      ]
    }
  }
}
```

### 2. Update Training Scripts

Update any scripts that reference old paths to use new canonical paths or the `get_dataset_path()` helper function.

### 3. Update Sync Scripts

The sync scripts (`gdrive-download.sh`, `sync-datasets.sh`) have been updated to support both canonical and legacy structures with automatic fallback.

### 4. Test Access

```bash
# Test rclone access
rclone lsd gdrive:datasets/cot_reasoning
rclone lsd gdrive:datasets/professional_therapeutic
rclone lsd gdrive:datasets/priority

# Test download
rclone copy gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health.json ./test/
```

---

## Rollback Plan

If migration needs to be reversed:

```bash
# Restore from backup
rclone copy gdrive:datasets_backup_YYYYMMDD gdrive:datasets --progress

# Or manually move back (reverse of migration steps)
rclone move gdrive:datasets/cot_reasoning/clinical_diagnosis_mental_health \
           gdrive:datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health \
           --transfers=1
```

---

## Notes

- **Use `--transfers=1`** to avoid Google Drive rate limits
- **Monitor progress** with `--progress` flag
- **Test with one dataset first** before migrating all
- **Keep backup** until migration is verified
- **Update documentation** as you go

---

## Related Documentation

- **Structure Reference**: `ai/training_ready/docs/GDRIVE_STRUCTURE.md`
- **Full Audit**: `.notes/markdown/three.md`
- **Dataset Registry**: `ai/data/dataset_registry.json`
