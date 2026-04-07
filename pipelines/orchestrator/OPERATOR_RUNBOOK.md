# Dataset Pipeline Operator Runbook

Complete guide for operating the dataset pipeline, generating exports, and
executing training runs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Storage Configuration](#storage-configuration)
3. [Quality Assurance](#quality-assurance)
4. [Training Execution](#training-execution)
5. [Troubleshooting](#troubleshooting)

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
# Run focused orchestrator validation
uv run pytest ai/pipelines/orchestrator/tests/test_intake_routing.py -q
```

Expected output:

- ✅ focused orchestrator tests pass

## Storage Configuration

### Local Storage (Default)

No configuration needed. By default, dataset pipeline runtime artifacts are
stored under `tmp/dataset_pipeline/` (outside the package tree).

- **Data**: `tmp/dataset_pipeline/data/`
- Override with `DATASET_PIPELINE_OUTPUT_DIR` (see
  `ai/pipelines/orchestrator/storage.env.template`)

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
from ai.pipelines.orchestrator.storage_config import get_storage_config

config = get_storage_config()
is_valid, error = config.validate()
if not is_valid:
    print(f"Configuration error: {error}")
else:
    print("✅ Storage configuration valid")
```

## Legacy Integrated Export

The legacy integrated export workflow has been removed. Do not use old
`verify_pipeline.py`, `export_dataset.py`, `run_integrated_probe.py`, or
`integrated_training_pipeline.py` references from previous notes or sessions.

## Quality Assurance

### Generate QA Report

```bash
uv run python -m ai.pipelines.orchestrator.qa_report_generator \
    production_exports/v1.0.0/dataset_v1.0.0.jsonl \
    --version 1.0.0 \
    --output production_exports/v1.0.0/qa_report_v1.0.0.json
```

### QA Report Contents

The QA report includes:

- **Quality Metrics**: Semantic coherence, therapeutic appropriateness, bias
  scores
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
    --dataset-path /path/to/current-dataset.jsonl \
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
from ai.pipelines.orchestrator.storage_manager import StorageManager
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

**Problem**: `'str' object has no attribute 'get'` errors when loading
psychology knowledge

**Solution**: These are non-fatal warnings. The pipeline will continue with
available data sources. To fix:

1. Check
   `ai/models/pixel_core/knowledge/psychology_knowledge_base_optimized.json`
   format
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
# Verify intake routing
uv run pytest ai/pipelines/orchestrator/tests/test_intake_routing.py -q

# Generate QA report
uv run python -m ai.pipelines.orchestrator.qa_report_generator \
    /path/to/current-dataset.jsonl --version 1.0.0

# Check manifest
python -c "from ai.pipelines.orchestrator.export_manifest import DatasetManifest; \
    from pathlib import Path; m = DatasetManifest.load(Path('/path/to/manifest.json')); \
    print(m.to_dict())"
```

### File Locations

- **Datasets**: project-specific output paths defined by the current workflow
- **Manifests**: release/export manifests generated by the active release path
- **Manifests**: `tmp/dataset_pipeline/production_exports/v{VERSION}/manifest_v{VERSION}.json`
- **QA Reports**: `tmp/dataset_pipeline/production_exports/v{VERSION}/qa_report_v{VERSION}.json`
- **Checkpoints**: `ai/lightning/checkpoints/v{VERSION}/`

## Support

For issues or questions:

1. Check this runbook
2. Review error logs
3. Check `ai/pipelines/orchestrator/IMPLEMENTATION_SUMMARY.md` for architecture details
4. Review `.kiro/specs/foundation-model-training/` for requirements
