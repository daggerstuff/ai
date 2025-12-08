# VPS Setup - Complete Guide

## Overview

This guide provides complete setup instructions for running the training consolidation system on your VPS with OVH S3 object storage.

## Prerequisites

- VPS access: `ssh -i ~/.ssh/planet vivi@146.71.78.184`
- Files extracted to: `/home/vivi/pixelated/ai/training_ready/`
- OVH S3 credentials (already configured in `ai/.env`)

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# SSH to VPS
ssh -i ~/.ssh/planet vivi@146.71.78.184

# Navigate to training_ready
cd /home/vivi/pixelated/ai/training_ready

# Run setup script
chmod +x scripts/vps_complete_setup.sh
zsh scripts/vps_complete_setup.sh
```

### Option 2: Manual Setup

Follow the steps below manually.

## Step-by-Step Setup

### 1. Install uv (Python Package Manager)

```bash
# On VPS
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Verify
uv --version
```

### 2. Install Python Dependencies

```bash
cd /home/vivi/pixelated

# Core dependencies
uv pip install boto3 datasets requests

# PyTorch (CPU version - can upgrade to GPU later)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Verify Environment Variables

```bash
cd /home/vivi/pixelated/ai/training_ready

# Load environment
set -a
source ../.env
set +a

# Verify OVH S3 credentials are set
echo "Bucket: $OVH_S3_BUCKET"
echo "Endpoint: $OVH_S3_ENDPOINT"
echo "Access Key: ${OVH_S3_ACCESS_KEY:0:10}..."
```

### 4. Test S3 Connection

```bash
cd /home/vivi/pixelated

# Test connection
uv run python3 ai/training_ready/scripts/test_ovh_s3.py
```

Expected output:
```
✅ S3 connection successful!
✅ Target bucket "pixel-data" found!
```

### 5. Test S3 Pipeline Integration

```bash
cd /home/vivi/pixelated

# Run S3 integration tests
uv run python3 ai/training_ready/scripts/test_s3_pipeline.py
```

Expected: All tests pass ✅

### 6. Verify Dataset Catalog

```bash
cd /home/vivi/pixelated/ai/training_ready

# Check catalog exists
python3 -c "
import json
with open('scripts/output/dataset_accessibility_catalog.json') as f:
    d = json.load(f)
    print(f'Total datasets: {d[\"summary\"][\"total\"]}')
    print(f'Local-only: {d[\"summary\"][\"local_only\"]}')
    print(f'HuggingFace IDs: {d[\"summary\"].get(\"huggingface_unique_ids\", 0)}')
"
```

## Running the Pipeline

### Monitor Upload Progress

```bash
cd /home/vivi/pixelated
uv run python3 ai/training_ready/scripts/monitor_upload_progress.py
```

### Continue Local Uploads (if needed)

```bash
cd /home/vivi/pixelated
./ai/training_ready/scripts/continue_uploads.sh
```

### Run Data Processing Pipeline

```bash
cd /home/vivi/pixelated

# Load environment
set -a
source ai/.env
set +a

# Run full pipeline
uv run python3 ai/training_ready/scripts/prepare_training_data.py --all
```

## S3 Dataset Access

### List Available Datasets

```bash
cd /home/vivi/pixelated
uv run python3 -c "
from ai.training_ready.tools.data_preparation.s3_dataset_loader import S3DatasetLoader
loader = S3DatasetLoader()
datasets = loader.list_datasets('datasets/huggingface/', max_keys=20)
for ds in datasets[:10]:
    print(ds)
"
```

### Load Dataset from S3

```bash
cd /home/vivi/pixelated
uv run python3 ai/training_ready/tools/data_preparation/s3_dataset_loader.py \
    s3://pixel-data/datasets/huggingface/huggingface/ShreyaR_DepressionDetection.jsonl 10
```

## Environment Configuration

### Required Environment Variables

Add to `ai/.env`:

```bash
# OVH Object Storage (S3-compatible)
OVH_S3_BUCKET=pixel-data
OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
OVH_S3_REGION=us-east-va
OVH_S3_ACCESS_KEY=<your-access-key>
OVH_S3_SECRET_KEY=<your-secret-key>

# Compatibility alias
S3_BUCKET=${OVH_S3_BUCKET}
```

### Loading Environment

Always load environment before running scripts:

```bash
set -a
source ai/.env
set +a
```

## Troubleshooting

### uv not found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Or add to ~/.zshrc
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### S3 Connection Failed

1. Verify credentials in `ai/.env`
2. Check endpoint URL is correct
3. Test with: `uv run python3 ai/training_ready/scripts/test_ovh_s3.py`

### Import Errors

```bash
# Make sure you're in project root
cd /home/vivi/pixelated

# Use uv run for scripts
uv run python3 ai/training_ready/scripts/...
```

### Module Not Found

```bash
# Install missing packages
uv pip install <package-name>

# Or reinstall all
uv pip install -r ai/training_ready/requirements.txt
```

## Quick Reference

### Essential Commands

```bash
# Test S3 connection
uv run python3 ai/training_ready/scripts/test_ovh_s3.py

# Test S3 pipeline
uv run python3 ai/training_ready/scripts/test_s3_pipeline.py

# Monitor uploads
uv run python3 ai/training_ready/scripts/monitor_upload_progress.py

# Run pipeline
uv run python3 ai/training_ready/scripts/prepare_training_data.py --all
```

### File Locations

- **Training Ready**: `/home/vivi/pixelated/ai/training_ready/`
- **Environment**: `/home/vivi/pixelated/ai/.env`
- **Catalog**: `ai/training_ready/scripts/output/dataset_accessibility_catalog.json`
- **Manifest**: `ai/training_ready/TRAINING_MANIFEST.json`

## Next Steps

1. ✅ Verify S3 connection works
2. ✅ Test S3 pipeline integration
3. ⏳ Monitor dataset uploads
4. ⏳ Run full data processing pipeline
5. ⏳ Begin training preparation

## Support

- Check logs: `ai/training_ready/scripts/output/`
- Test scripts: `ai/training_ready/scripts/test_*.py`
- Documentation: `ai/training_ready/*.md`

