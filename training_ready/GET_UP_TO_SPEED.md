# Get Up to Speed: Training Consolidation Project

## Project Context

This project consolidates all training assets from 20+ AI directories into a unified `ai/training_ready/` structure for immediate training deployment. The system includes:

- **Asset Discovery**: Systematic exploration and cataloging
- **Manifest Generation**: Complete inventory of 9,608 assets
- **Data Processing Pipeline**: Source → Process → Filter → Format → Assemble
- **S3 Integration**: Direct downloads from HuggingFace/Kaggle to S3

## Current Status

### ✅ Completed
- All 18 spec tasks implemented
- Directory exploration (9,608 files cataloged)
- Training manifest generated (4.2MB, all assets)
- Data processing pipeline scripts created
- CPU-only torch installed for local execution
- Dataset accessibility cataloging

### 🔄 In Progress
- VPS migration with S3 storage
- Direct HF/Kaggle → S3 downloads
- Local-only dataset identification

## Key Files

### Core Scripts
- `scripts/explore_directories.py` - Catalog all AI directories
- `scripts/generate_manifest.py` - Generate TRAINING_MANIFEST.json
- `scripts/catalog_local_only_datasets.py` - **NEW** Identify local vs remote datasets
- `scripts/download_to_s3.py` - **NEW** Download remote datasets directly to S3
- `scripts/prepare_training_data.py` - End-to-end data processing orchestration

### Data Processing Pipeline
- `tools/data_preparation/source_datasets.py` - Source/download datasets
- `tools/data_preparation/filter_and_clean.py` - Filter, clean, remove PII
- `tools/data_preparation/format_for_training.py` - Convert to standard format
- `pipelines/integrated/process_all_datasets.py` - Process through unified pipeline
- `pipelines/integrated/assemble_final_dataset.py` - Assemble final datasets

### Documentation
- `README.md` - Quick start and overview
- `TRAINING_PLAN.md` - Comprehensive training strategy
- `TRAINING_MANIFEST.json` - Complete asset inventory
- `VPS_MIGRATION_GUIDE.md` - **NEW** VPS migration instructions
- `GET_UP_TO_SPEED.md` - This file

## Architecture

### 4-Stage Training Architecture
1. **Stage 1 (40%)**: Foundation & Rapport
2. **Stage 2 (25%)**: Therapeutic Expertise & Reasoning
3. **Stage 3 (20%)**: Edge Stress Test
4. **Stage 4 (15%)**: Voice & Persona

### Data Flow
```
Local Machine → Catalog → Identify Local vs Remote
                                    ↓
Remote (HF/Kaggle/URL) → Direct Download → S3
                                    ↓
Local-Only → Upload from Local → S3
                                    ↓
S3 → Process → Filter → Format → Assemble → Final Datasets
```

## Current Task: VPS Migration

### What We're Doing
1. **Cataloging**: Identify which datasets are local-only vs remote-accessible
2. **S3 Downloads**: Download remote datasets (HF/Kaggle/URL) directly to S3
3. **Local Uploads**: Upload local-only datasets from local machine to S3
4. **Migration**: Move all scripts and configs to VPS

### Why
- Local connection is too slow for large uploads
- Most datasets can be downloaded directly from source to S3
- Only truly local-only datasets need manual upload

## Environment Setup

### Required Dependencies
```bash
# Core ML
torch torchvision torchaudio (CPU-only for local)

# Data Processing
datasets  # HuggingFace
kaggle    # Kaggle API
boto3     # S3 access
requests  # URL downloads

# Project dependencies
# (from main pyproject.toml)
```

### Environment Variables
```bash
# AWS/S3
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
S3_BUCKET

# Optional
KAGGLE_USERNAME
KAGGLE_KEY
HF_TOKEN
```

## Quick Commands

### Catalog Datasets
```bash
python3 scripts/catalog_local_only_datasets.py
# Output: scripts/output/dataset_accessibility_catalog.json
#         scripts/output/local_only_upload_list.txt
```

### Download to S3
```bash
python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json
```

### Process Data
```bash
uv run python3 scripts/prepare_training_data.py --all
```

## Next Steps

1. ✅ Run catalog script to identify local-only datasets
2. ✅ Test S3 download script with sample dataset
3. ✅ Create tarball with all necessary files
4. ⏳ Upload tarball to VPS
5. ⏳ Extract and set up on VPS
6. ⏳ Download remote datasets to S3
7. ⏳ Upload local-only datasets from local machine
8. ⏳ Run full pipeline on VPS

## Important Notes

- **Local Connection**: Too slow for large uploads, so we're using direct HF/Kaggle → S3
- **CPU Torch**: Installed for local testing, VPS can use GPU version
- **Manifest**: Contains all 9,608 assets with paths - will need S3 path updates
- **Pipeline**: Designed to work with S3 paths once datasets are uploaded

## Questions to Resolve

1. S3 bucket name and region?
2. AWS credentials setup method?
3. Kaggle API credentials location?
4. VPS location and access method?
5. Preferred S3 path structure?


