# Training Ready - Consolidated Training System

**Status**: Single canonical home for all training packages, scripts, configs, and documentation  
**Last Updated**: 2025-12-13

## Architecture

### Data Flow
```
Google Drive (Source/Staging)
    ↓ [rclone sync]
S3: s3://pixelated-training-data/ (Training Mecca - Canonical)
    ↓ [Training Scripts Read From]
Model Training
```

**Key Principle**: S3 is the training mecca - all training data flows through S3. Training scripts read from S3, not Google Drive or local files.

---

## Directory Structure

```
ai/training_ready/
├── docs/                          # All training documentation
│   ├── S3_TRAINING_DATA_STRUCTURE.md  ⭐ S3 canonical structure
│   ├── S3_USAGE_GUIDE.md          # How to use S3 in training scripts
│   ├── S3_EXECUTION_ORDER.md     # S3-first workflow
│   ├── GDRIVE_STRUCTURE.md       # Google Drive (source/staging)
│   ├── QUICK_START_GUIDE.md      # Quick start instructions
│   └── [other docs...]
│
├── scripts/                       # Data processing and utility scripts
│   ├── compile_final_dataset.py  # Compile final training dataset
│   ├── verify_final_dataset.py   # Verify dataset quality
│   ├── [other data processing scripts...]
│
├── packages/                      # Training packages
│   ├── apex/                     # KAN-28 enhanced training
│   │   └── scripts/train_enhanced.py
│   └── velocity/                 # MoE optimized training
│       └── training_scripts/
│           ├── train_optimized.py
│           ├── train_moe_h100.py
│           └── inference_service.py
│
├── configs/                       # Shared configuration files
│   └── training_curriculum_2025.json
│
├── packages/                      # Package-specific configs
│   ├── apex/config/              # Apex configurations
│   └── velocity/configs/          # Velocity configurations
│
├── pipelines/                     # Data pipeline scripts
│   └── integrated_training_pipeline.py  # Unified data pipeline (S3-aware)
│
├── models/                        # Shared model architecture files
│   ├── moe_architecture.py        # (shared by both packages)
│   └── therapeutic_progress_tracker.py  # (shared by both packages)
│
├── utils/                         # Utility modules
│   ├── s3_dataset_loader.py      # S3 dataset loader (streaming)
│   └── logger.py
│
├── data/                          # Local data (cache/temporary)
│   └── training_data_consolidated/
│
├── platforms/                     # Platform-specific integrations
│   └── ovh/                       # OVH platform integration
│
└── experimental/                  # Experimental features
    └── h100_moe/
```

---

## Quick Start

### 1. Environment Setup

```bash
cd ai/training_ready

# Install dependencies
uv pip install -r configs/requirements_moe.txt

# Optional: Install NGC CLI for NeMo resource downloads
# Note: NGC CLI must be downloaded from https://catalog.ngc.nvidia.com
# (It's not a PyPI package - download and extract to ~/ngc-cli/)
export NGC_API_KEY="your-ngc-api-key"  # Get from https://catalog.ngc.nvidia.com

# S3 credentials are loaded from .env automatically
# Ensure .env has:
# OVH_S3_ACCESS_KEY=...
# OVH_S3_SECRET_KEY=...
```

### 2. Verify S3 Access

```bash
# Test that S3 credentials are loaded and connection works
python scripts/verify_s3_access.py
```

### 3. Compile and Upload Final Dataset

```bash
# Compile final dataset (automatically uploads to S3 and removes local copies)
python scripts/compile_final_dataset.py

# Or upload existing local datasets to S3
python scripts/upload_local_datasets_to_s3.py
```

**Note**: The compile script automatically:
- Uploads shards to `s3://{bucket}/final_dataset/{split}/{shard_id}.jsonl`
- Uploads compiled export to `s3://{bucket}/final_dataset/compiled/final_training_dataset.jsonl`
- Uploads manifest to `s3://{bucket}/final_dataset/manifest.json`
- Removes local copies after successful upload (keeps manifest for reference)

### 4. Load Dataset from S3

```python
from ai.training_ready.utils.s3_dataset_loader import load_dataset_from_s3

# S3 is canonical - load from S3
data = load_dataset_from_s3(
    dataset_name="clinical_diagnosis_mental_health.json",
    category="cot_reasoning"
)
```

### 5. Run Training

```bash
# Training script automatically uses S3 if available
cd ai/training_ready
python scripts/train_optimized.py
```

---

## Key Features

### S3-First Architecture
- **S3 is canonical** - All training data in `s3://pixelated-training-data/`
- **Google Drive syncs to S3** - Source/staging area, not used directly
- **Local is cache only** - Temporary caches, not source of truth

### Consolidated Packages
- **Apex** - KAN-28 enhanced training (comprehensive, all 6 components)
- **Velocity** - MoE optimized training (fast iteration, flexible profiles)
- **OVH Platform Integration** - OVHcloud AI Training
- All merged into unified `training_ready/` structure

### S3 Dataset Loader
- **Streaming support** - Memory-efficient for large datasets
- **Automatic path resolution** - Finds datasets in canonical structure
- **Local caching** - Optional local cache for faster access
- **Backward compatible** - Falls back to local files if needed

---

## Documentation

### Essential Reading
1. **`docs/S3_TRAINING_DATA_STRUCTURE.md`** ⭐ - S3 canonical structure (start here)
2. **`docs/S3_USAGE_GUIDE.md`** - How to use S3 in training scripts
3. **`docs/QUICK_START_GUIDE.md`** - Complete setup and execution guide

### Reference
- **`docs/S3_EXECUTION_ORDER.md`** - S3-first workflow steps
- **`docs/GDRIVE_STRUCTURE.md`** - Google Drive organization (source/staging)
- **`docs/IMPLEMENTATION_COMPLETE.md`** - System implementation details

### Audit Notes
- **`.notes/markdown/two.md`** - Local package consolidation audit
- **`.notes/markdown/three.md`** - Google Drive → S3 consolidation audit
- **`.notes/markdown/four.md`** - S3 training mecca summary

---

## Training Scripts

### Dataset Compilation & Upload

- **`scripts/compile_final_dataset.py`** - Compile final training dataset
  - Reads from S3, creates shards and compiled export
  - **Automatically uploads to S3** and removes local copies
  - Creates manifest with S3 paths
  
- **`scripts/upload_local_datasets_to_s3.py`** - Upload existing local datasets
  - Uploads large dataset files to S3
  - Removes local copies after successful upload
  - Useful for migrating existing local datasets

### Main Training Scripts

**Apex Package** (KAN-28 Enhanced):
- **`packages/apex/scripts/train_enhanced.py`** - KAN-28 enhanced training

**Velocity Package** (MoE Optimized):
- **`packages/velocity/training_scripts/train_optimized.py`** - Automatic optimization, S3-aware
- **`packages/velocity/training_scripts/train_moe_h100.py`** - MoE training on H100, S3-aware
- **`packages/velocity/training_scripts/inference_service.py`** - FastAPI inference service
- **`packages/velocity/training_scripts/inference_optimizer.py`** - Inference optimization
- **`packages/velocity/training_scripts/progress_tracking_api.py`** - Progress tracking API

### Data Pipeline

- **`packages/velocity/data_pipeline/integrated_training_pipeline.py`** - Unified data pipeline (S3-aware)

---

## Configuration

### Training Configs

**Apex Package**:
- **`packages/apex/config/enhanced_training_config.json`** - KAN-28 enhanced config
- **`packages/apex/config/lightning_deployment_config.json`** - Lightning.ai deployment
- **`packages/apex/requirements.txt`** - Apex dependencies

**Velocity Package**:
- **`packages/velocity/configs/moe_training_config.json`** - MoE training configuration
- **`packages/velocity/configs/requirements_moe.txt`** - Velocity dependencies

---

## S3 Access

### Using S3DatasetLoader

```python
from ai.training_ready.utils.s3_dataset_loader import (
    S3DatasetLoader,
    get_s3_dataset_path,
    load_dataset_from_s3
)

# Load dataset
data = load_dataset_from_s3("dataset.json", category="cot_reasoning")

# Or use loader directly
loader = S3DatasetLoader()
data = loader.load_json("s3://pixelated-training-data/gdrive/processed/cot_reasoning/dataset.json")
```

### Environment Variables

```bash
# Required for S3 access
export OVH_ACCESS_KEY="your_access_key"
export OVH_SECRET_KEY="your_secret_key"

# Or AWS-compatible
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

---

## NGC CLI Integration

NGC CLI is integrated for downloading NeMo resources and training assets:

- **`utils/ngc_resources.py`** - NGC resource downloader
- **`docs/NGC_CLI_INTEGRATION.md`** - Complete NGC CLI setup and usage guide

See [NGC CLI Integration Guide](docs/NGC_CLI_INTEGRATION.md) for details.

## Related Systems

- **`ai/dataset_pipeline/`** - Dataset processing pipeline (still exists, used by integrated pipeline)
- **`ai/data/dataset_registry.json`** - Complete dataset catalog (should reference S3 paths)
- **`ai/training_ready/platforms/ovh/`** - OVH platform integration and sync scripts

---

## Migration Status

### ✅ Completed
- [x] Local training packages consolidated into `training_ready/`
- [x] S3 structure documented as canonical
- [x] S3DatasetLoader implemented
- [x] Training scripts updated to support S3
- [x] Documentation created

### 🔄 In Progress
- [ ] Complete raw sync: Google Drive → S3 `gdrive/raw/`
- [ ] Process and organize: `gdrive/raw/` → `gdrive/processed/`
- [ ] Update all training scripts to use S3 by default
- [ ] Update dataset_registry.json with S3 paths as primary

### 📋 Planned
- [ ] Implement S3DatasetLoader streaming for large files
- [ ] Update integrated pipeline to read from S3
- [ ] Create S3 path resolution helpers
- [ ] Full S3-first workflow implementation

---

## Support

For questions or issues:
1. Check `docs/S3_TRAINING_DATA_STRUCTURE.md` for S3 organization
2. Review `docs/S3_USAGE_GUIDE.md` for usage patterns
3. See `docs/QUICK_START_GUIDE.md` for setup instructions
