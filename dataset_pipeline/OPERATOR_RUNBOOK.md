# Dataset Pipeline Operator Runbook

Complete guide for operating the dataset pipeline, generating exports, and executing training runs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Storage Configuration](#storage-configuration)
3. [Dataset Export](#dataset-export)
4. [Quality Assurance](#quality-assurance)
5. [Training Execution](#training-execution)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Environment Setup

```bash
# Activate uv environment
cd /home/vivi/pixelated
uv sync

# Install training dependencies
uv pip install -r ai/config/requirements_training.txt
```

### Verify Installation

```bash
# Run verification script
uv run python ai/dataset_pipeline/verify_pipeline.py
```

Expected output:
- ✅ All imports successful
- ✅ Data sources accessible (some may be optional)
- ✅ Pipeline execution successful

## Storage Configuration

### Local Storage (Default)

No configuration needed. By default, dataset pipeline runtime artifacts are stored under
`tmp/dataset_pipeline/` (outside the package tree).

- **Data**: `tmp/dataset_pipeline/data/`
- Override with `DATASET_PIPELINE_OUTPUT_DIR` (see `ai/dataset_pipeline/storage.env.template`)

### S3 Storage

Set environment variables:

```bash
export DATASET_STORAGE_BACKEND=s3
export DATASET_S3_BUCKET=your-bucket-name
export DATASET_S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

### GCS Storage

Set environment variables:

```bash
export DATASET_STORAGE_BACKEND=gcs
export DATASET_GCS_BUCKET=your-bucket-name
export DATASET_GCS_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Verify Storage Configuration

```python
from ai.dataset_pipeline.storage_config import get_storage_config

config = get_storage_config()
is_valid, error = config.validate()
if not is_valid:
    print(f"Configuration error: {error}")
else:
    print("✅ Storage configuration valid")
```

## Dataset Export

### Basic Export

Generate a dataset export with default settings:

```bash
uv run python -m ai.dataset_pipeline.export_dataset \
    --version 1.0.0 \
    --target-samples 1000 \
    --seed 42
```

### Export Options

```bash
uv run python -m ai.dataset_pipeline.export_dataset \
    --version 1.0.0 \              # Dataset version
    --target-samples 1000 \        # Target number of samples
    --seed 42 \                    # Random seed for reproducibility
    --output-dir ./exports \       # Custom output directory
    --no-upload \                  # Skip storage upload
    --no-quality                   # Skip quality validation
```

### Export Outputs

After export, you'll find:

```
production_exports/v1.0.0/
├── dataset_v1.0.0.jsonl          # JSONL format
├── dataset_v1.0.0.parquet         # Parquet format
├── manifest_v1.0.0.json          # Export manifest
└── config_lock.json               # Locked configuration
```

### Manifest Contents

The manifest includes:
- File checksums (SHA256)
- File sizes and row counts
- Source distribution
- Configuration lock (git commit, seed, etc.)
- Quality summary
- Storage URLs (if uploaded)

### Verify Export Integrity

```python
from ai.dataset_pipeline.export_manifest import DatasetManifest
from pathlib import Path

manifest = DatasetManifest.load(Path("production_exports/v1.0.0/manifest_v1.0.0.json"))
is_valid, errors = manifest.verify_files(Path("production_exports/v1.0.0"))

if is_valid:
    print("✅ All files verified")
else:
    print(f"❌ Verification errors: {errors}")
```

## Quality Assurance

### Generate QA Report

```bash
uv run python -m ai.dataset_pipeline.qa_report_generator \
    production_exports/v1.0.0/dataset_v1.0.0.jsonl \
    --version 1.0.0 \
    --output production_exports/v1.0.0/qa_report_v1.0.0.json
```

### QA Report Contents

The QA report includes:
- **Quality Metrics**: Semantic coherence, therapeutic appropriateness, bias scores
- **Safety Metrics**: Crisis flags (detected, resolved, unresolved)
- **Privacy Metrics**: PII detection (detected, resolved, unresolved)
- **Threshold Validation**: Pass/fail against quality thresholds
- **Detailed Findings**: Crisis, PII, and bias findings

### Quality Thresholds

Default thresholds (configurable):

- Semantic coherence: ≥ 0.8
- Therapeutic appropriateness: ≥ 0.7
- Crisis flags: ≤ 0.5% unresolved
- PII detected: 0% (must be resolved)
- Bias score: ≥ 0.6
- Overall quality: ≥ 0.75

### Interpret QA Report

```bash
# View report summary
cat production_exports/v1.0.0/qa_report_v1.0.0.json | jq '.'

# Check if passes
cat production_exports/v1.0.0/qa_report_v1.0.0.json | jq '.passes_thresholds'

# View failures
cat production_exports/v1.0.0/qa_report_v1.0.0.json | jq '.failures'
```

## Training Execution

### Prerequisites for H100 Training

1. **Lightning.ai Account**: Set up H100 access
2. **Environment Variables**:
   ```bash
   export LIGHTNING_PROJECT_ID=your-project-id
   export WANDB_API_KEY=your-wandb-key
   export HF_TOKEN=your-huggingface-token
   ```

3. **Dataset Ready**: Ensure dataset export is complete and QA report passes

### Training Configuration

Training configuration is managed in:
- `ai/lightning/moe_training_config.json` - Model and training config
- `ai/lightning/train_optimized.py` - Training script

### Execute Training

```bash
cd ai/lightning
uv run python train_optimized.py \
    --dataset-path ../dataset_pipeline/production_exports/v1.0.0/dataset_v1.0.0.jsonl \
    --output-dir ./checkpoints/v1.0.0 \
    --config moe_training_config.json
```

### Training Outputs

After training:
- Model checkpoints in `ai/lightning/checkpoints/v1.0.0/`
- Training logs
- W&B experiment tracking
- Evaluation metrics

### Upload Training Artifacts

```python
from ai.dataset_pipeline.storage_manager import StorageManager
from pathlib import Path

storage = StorageManager()
checkpoint_path = Path("ai/lightning/checkpoints/v1.0.0/best_model.pt")

upload_info = storage.upload_with_checksum(
    checkpoint_path,
    f"checkpoints/v1.0.0/{checkpoint_path.name}",
    metadata={'version': '1.0.0', 'type': 'checkpoint'}
)

print(f"Uploaded to: {upload_info['storage_url']}")
```

## Troubleshooting

### Pipeline Import Errors

**Problem**: `ModuleNotFoundError` for transformers or other packages

**Solution**:
```bash
uv pip install "transformers>=4.35.0"
uv pip install -r ai/config/requirements_training.txt
```

### Psychology Knowledge Loader Errors

**Problem**: `'str' object has no attribute 'get'` errors when loading psychology knowledge

**Solution**: These are non-fatal warnings. The pipeline will continue with available data sources. To fix:
1. Check `ai/pixel/knowledge/psychology_knowledge_base_optimized.json` format
2. Ensure it's a list of objects, not strings

### Storage Upload Failures

**Problem**: S3/GCS upload fails

**Solution**:
1. Verify credentials: `aws s3 ls` or `gsutil ls`
2. Check bucket permissions
3. Verify environment variables are set correctly
4. Use `--no-upload` flag to skip upload during testing

### Quality Validation Failures

**Problem**: QA report shows failures

**Solution**:
1. Review failures in QA report
2. Check if thresholds are too strict
3. Re-run pipeline with quality validation enabled
4. Manually review flagged samples

### Training Failures

**Problem**: Training fails on H100

**Solution**:
1. Check Lightning.ai quota and access
2. Verify dataset format is correct
3. Check training config parameters
4. Review training logs for specific errors

## Quick Reference

### Common Commands

```bash
# Verify pipeline
uv run python ai/dataset_pipeline/verify_pipeline.py

# Export dataset
uv run python -m ai.dataset_pipeline.export_dataset \
    --version 1.0.0 --target-samples 1000 --seed 42

# Generate QA report
uv run python -m ai.dataset_pipeline.qa_report_generator \
    tmp/dataset_pipeline/production_exports/v1.0.0/dataset_v1.0.0.jsonl --version 1.0.0

# Check manifest
python -c "from ai.dataset_pipeline.export_manifest import DatasetManifest; \
    from pathlib import Path; \
    m = DatasetManifest.load(Path('tmp/dataset_pipeline/production_exports/v1.0.0/manifest_v1.0.0.json')); \
    print(m.to_dict())"
```

### File Locations

- **Exports**: `tmp/dataset_pipeline/production_exports/v{VERSION}/`
- **Config Locks**: `tmp/dataset_pipeline/production_exports/v{VERSION}/config_lock.json`
- **Manifests**: `tmp/dataset_pipeline/production_exports/v{VERSION}/manifest_v{VERSION}.json`
- **QA Reports**: `tmp/dataset_pipeline/production_exports/v{VERSION}/qa_report_v{VERSION}.json`
- **Checkpoints**: `ai/lightning/checkpoints/v{VERSION}/`

## Support

For issues or questions:
1. Check this runbook
2. Review error logs
3. Check `ai/dataset_pipeline/IMPLEMENTATION_SUMMARY.md` for architecture details
4. Review `.kiro/specs/foundation-model-training/` for requirements

