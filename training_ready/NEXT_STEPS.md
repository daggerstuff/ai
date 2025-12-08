# Next Steps: S3 Integration & Pipeline Updates

## Overview

While local dataset uploads continue in the background, we can work on integrating S3 storage into the data processing pipeline and preparing for VPS deployment.

## Priority Tasks

### 1. ✅ S3 Dataset Loaders (High Priority)
**Goal**: Create loaders that can read datasets directly from S3

**Tasks**:
- [ ] Create `S3DatasetLoader` class that reads JSONL/JSON from S3
- [ ] Support streaming for large datasets
- [ ] Integrate with existing pipeline scripts
- [ ] Test with uploaded HuggingFace datasets

**Files to create**:
- `ai/training_ready/tools/data_preparation/s3_dataset_loader.py`

### 2. ✅ Update Pipeline Scripts for S3 (High Priority)
**Goal**: Modify pipeline to work with S3 paths

**Tasks**:
- [ ] Update `source_datasets.py` to check S3 first, then local
- [ ] Update `process_all_datasets.py` to read from S3
- [ ] Update `assemble_final_dataset.py` to read from S3
- [ ] Add S3 path resolution logic

**Files to update**:
- `ai/training_ready/tools/data_preparation/source_datasets.py`
- `ai/training_ready/pipelines/integrated/process_all_datasets.py`
- `ai/training_ready/pipelines/integrated/assemble_final_dataset.py`

### 3. ✅ Manifest S3 Path Mapping (Medium Priority)
**Goal**: Update manifest with S3 paths for uploaded datasets

**Tasks**:
- [ ] Create script to map local paths to S3 paths
- [ ] Update manifest with S3 URLs
- [ ] Create mapping file for reference

**Files to create**:
- `ai/training_ready/scripts/update_manifest_s3_paths.py`

### 4. ✅ VPS Environment Setup (Medium Priority)
**Goal**: Complete VPS setup documentation and scripts

**Tasks**:
- [ ] Verify `uv` installation on VPS
- [ ] Create VPS environment setup script
- [ ] Test pipeline on VPS with S3
- [ ] Document VPS-specific considerations

**Files to create/update**:
- `ai/training_ready/VPS_SETUP_COMPLETE.md`
- `ai/training_ready/scripts/vps_environment_setup.sh`

### 5. ✅ Pipeline Testing with S3 (High Priority)
**Goal**: Test end-to-end pipeline with S3 datasets

**Tasks**:
- [ ] Test S3 loader with sample dataset
- [ ] Test full pipeline with S3 data
- [ ] Verify data integrity
- [ ] Performance benchmarking

### 6. ✅ Documentation Updates (Low Priority)
**Goal**: Update all docs to reflect S3 usage

**Tasks**:
- [ ] Update README.md with S3 instructions
- [ ] Update TRAINING_PLAN.md with S3 dataset references
- [ ] Update GET_UP_TO_SPEED.md
- [ ] Create S3 usage guide

## Implementation Order

1. **S3 Dataset Loader** (Foundation - needed for everything else)
2. **Pipeline Updates** (Core functionality)
3. **Manifest Updates** (Data organization)
4. **Testing** (Validation)
5. **VPS Setup** (Deployment)
6. **Documentation** (Knowledge transfer)

## Quick Wins

While uploads continue, we can:
- ✅ Create S3 loader (can test with existing 6 HF datasets)
- ✅ Update manifest mapping script (doesn't need uploads to complete)
- ✅ Create VPS setup documentation (planning)

## Estimated Time

- S3 Loader: 30-45 min
- Pipeline Updates: 45-60 min
- Manifest Updates: 15-30 min
- Testing: 30-45 min
- VPS Setup: 30-45 min
- Documentation: 30 min

**Total**: ~3-4 hours of work

