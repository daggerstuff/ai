# Dataset Migration to S3 - Status Report

## Overview

This document tracks the progress of migrating all training datasets to OVH S3 object storage (`pixel-data` bucket).

## Migration Strategy

1. **HuggingFace Datasets**: Download directly from HuggingFace Hub to S3 (bypasses local/VPS)
2. **Local-Only Datasets**: Upload from local machine to S3
3. **Kaggle Datasets**: Download directly from Kaggle to S3 (if any)
4. **URL Datasets**: Download directly from URLs to S3 (if any)

## Current Status

### ✅ Completed Setup

- [x] OVH S3 credentials configured (local and VPS)
- [x] S3 connection tested and verified
- [x] Dataset cataloging completed (5,208 datasets analyzed)
- [x] HuggingFace dataset ID extraction (6 unique IDs from loader scripts)
- [x] Download scripts created and tested
- [x] Upload scripts created and tested

### 📊 Dataset Breakdown

| Category | Count | Status |
|----------|-------|--------|
| **Total Datasets** | 5,208 | Cataloged |
| **Local-Only** | 5,200 | Ready for upload |
| **HuggingFace** | 6 unique IDs | Ready for download |
| **Kaggle** | 0 | N/A |
| **URL** | 0 | N/A |
| **Unknown** | 5 | Needs review |

### ✅ Completed

#### HuggingFace Dataset Downloads

**Extracted Dataset IDs:**
1. `mlx-community/Human-Like-DPO` ✅
2. `flammenai/character-roleplay-DPO` ✅
3. `PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT` ✅
4. `iqrakiran/customized-mental-health-snli2` ✅
5. `typosonlr/MentalHealthPreProcessed` ✅
6. `ShreyaR/DepressionDetection` ✅

**Status**: ✅ **COMPLETE** - All 6 datasets downloaded
- Script: `ai/training_ready/scripts/download_hf_datasets_to_s3.py`
- Results: `ai/training_ready/scripts/output/hf_download_results.json`
- Success rate: 100% (6/6)

#### Local Dataset Uploads

**Status**: 🔄 **IN PROGRESS**
- Script: `ai/training_ready/scripts/upload_local_datasets_to_s3.py`
- Total files: 5,200
- Processed: 100+ (continuing in background)
- Success rate: 100% (no failures so far)
- Resume script: `ai/training_ready/scripts/continue_uploads.sh`

## Scripts and Tools

### Catalog Script
```bash
uv run python3 ai/training_ready/scripts/catalog_local_only_datasets.py
```
- Output: `scripts/output/dataset_accessibility_catalog.json`
- Extracts HuggingFace dataset IDs from loader scripts
- Classifies datasets by accessibility type

### HuggingFace Download Script
```bash
set -a && source ai/.env && set +a
uv run python3 ai/training_ready/scripts/download_hf_datasets_to_s3.py
```
- Downloads all HuggingFace datasets directly to S3
- Progress tracking and error reporting
- Results saved to `output/hf_download_results.json`

### Local Upload Script
```bash
set -a && source ai/.env && set +a

# Test with small sample
uv run python3 ai/training_ready/scripts/upload_local_datasets_to_s3.py --max-files 10

# Full upload (can be resumed)
uv run python3 ai/training_ready/scripts/upload_local_datasets_to_s3.py --skip-existing

# Resume from previous run
uv run python3 ai/training_ready/scripts/upload_local_datasets_to_s3.py \
  --resume ai/training_ready/scripts/output/local_upload_results.json
```

**Features:**
- Resume interrupted uploads
- Skip existing files
- Progress tracking
- Error reporting
- Large file support (multipart upload)

## S3 Structure

```
s3://pixel-data/
├── datasets/
│   ├── huggingface/          # Direct HF downloads
│   │   ├── mlx-community_Human-Like-DPO.jsonl
│   │   ├── flammenai_character-roleplay-DPO.jsonl
│   │   └── ...
│   └── local/                 # Local file uploads
│       ├── ai/
│       │   ├── data_designer/
│       │   ├── dataset_pipeline/
│       │   └── ...
```

## Environment Variables

Required in `ai/.env`:
```bash
OVH_S3_BUCKET=pixel-data
OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
OVH_S3_REGION=us-east-va
OVH_S3_ACCESS_KEY=<your-access-key>
OVH_S3_SECRET_KEY=<your-secret-key>
```

## Next Steps

1. **Monitor HuggingFace Downloads**
   - Check `output/hf_download_results.json` for completion status
   - Re-run failed downloads if needed

2. **Start Local Uploads**
   - Begin with small batches to test
   - Use `--skip-existing` to avoid duplicates
   - Monitor progress and resume as needed

3. **Process Unknown Datasets**
   - Review the 5 "unknown" datasets
   - Classify and add to appropriate category

4. **Verify S3 Contents**
   - List all uploaded files
   - Verify file integrity
   - Update manifest with S3 paths

5. **Update Pipeline**
   - Modify data processing pipeline to read from S3
   - Test end-to-end workflow
   - Update documentation

## Notes

- **Local Connection**: Too slow for large uploads, so direct HF→S3 downloads are preferred
- **Resume Capability**: Both scripts support resuming interrupted operations
- **Error Handling**: Failed uploads/downloads are logged and can be retried
- **Testing**: Always test with `--max-files` flag before full runs

## Files

- Catalog: `ai/training_ready/scripts/output/dataset_accessibility_catalog.json`
- HF Download Results: `ai/training_ready/scripts/output/hf_download_results.json`
- Local Upload Results: `ai/training_ready/scripts/output/local_upload_results.json`
- Upload List: `ai/training_ready/scripts/output/local_only_upload_list.txt`

