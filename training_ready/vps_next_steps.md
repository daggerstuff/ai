# VPS Next Steps - Quick Reference

## ✅ Current Status

- **Tarball extracted**: `~/training_ready/training_ready/`
- **Python 3.12.3**: Available
- **Catalog files**: Present in `scripts/output/`
- **Documentation**: All guides available

## 📋 Immediate Next Steps

### 1. Install uv (if not installed)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc  # or source ~/.zshrc
```

### 2. Install Dependencies

```bash
cd ~/training_ready/training_ready
./install_dependencies.sh
```

Or manually:
```bash
# Using uv (recommended)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install boto3 datasets kaggle requests

# Or using pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install boto3 datasets kaggle requests
```

### 3. Configure AWS Credentials

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # or your region
export S3_BUCKET="your-bucket-name"
```

**Or use AWS CLI:**
```bash
aws configure
```

### 4. Verify Catalog

```bash
cd ~/training_ready/training_ready
python3 -c "
import json
with open('scripts/output/dataset_accessibility_catalog.json') as f:
    d = json.load(f)
    print(f\"Total datasets: {d['summary']['total']}\")
    print(f\"Local-only (need upload): {d['summary']['local_only']}\")
    print(f\"HuggingFace (direct to S3): {d['summary']['huggingface']}\")
"
```

**Expected output:**
- Total: 5,208
- Local-only: 5,134
- HuggingFace: 74

### 5. Download Remote Datasets to S3

```bash
cd ~/training_ready/training_ready

# Using uv (if installed)
uv run python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json

# Or using python3 directly
python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json
```

This will download **74 HuggingFace datasets** directly to S3.

### 6. Upload Local-Only Datasets (from your LOCAL machine)

**This must be done from your local machine where the files exist:**

```bash
# On your LOCAL machine
cd /home/vivi/pixelated

# Upload using the upload list
aws s3 sync \
  --exclude "*" \
  --include-from ai/training_ready/scripts/output/local_only_upload_list.txt \
  / \
  s3://$S3_BUCKET/datasets/local/
```

This will upload **5,134 local-only files** to S3.

### 7. Verify S3 Uploads

```bash
# Check S3 bucket contents
aws s3 ls s3://$S3_BUCKET/datasets/ --recursive | wc -l

# Should see:
# - datasets/huggingface/ (74 files)
# - datasets/local/ (5,134 files)
```

### 8. Run Data Processing Pipeline

```bash
cd ~/training_ready/training_ready

# Using uv
uv run python3 scripts/prepare_training_data.py --all

# Or using python3
python3 scripts/prepare_training_data.py --all
```

## 📚 Documentation Reference

- **Quick Start**: `cat GET_UP_TO_SPEED.md`
- **Migration Guide**: `cat VPS_MIGRATION_GUIDE.md`
- **Environment Setup**: `cat ENV_QUICKSTART.md`
- **Checklist**: `cat VPS_MIGRATION_CHECKLIST.md`

## 🔍 Troubleshooting

### uv not found
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### AWS credentials not working
```bash
# Test S3 access
aws s3 ls s3://$S3_BUCKET/

# Verify credentials
aws sts get-caller-identity
```

### Python import errors
```bash
# Use uv run to ensure correct environment
uv run python3 scripts/download_to_s3.py --help

# Or install dependencies globally
pip3 install boto3 datasets kaggle requests
```

### Slow downloads
- HuggingFace downloads are automatic and should be fast
- Local-only uploads (5,134 files) will be slow from your local machine
- Consider uploading in batches or during off-peak hours

## 📊 Summary

**What's on VPS:**
- ✅ All code and scripts
- ✅ Dataset catalog (5,208 datasets identified)
- ✅ Documentation

**What needs to happen:**
1. ✅ Install dependencies (uv, torch, boto3, etc.)
2. ✅ Configure AWS credentials
3. ⏳ Download 74 HuggingFace datasets → S3 (from VPS)
4. ⏳ Upload 5,134 local-only files → S3 (from local machine)
5. ⏳ Run data processing pipeline (on VPS)

**Time estimate:**
- Setup: ~10 minutes
- HF downloads: ~30-60 minutes (74 datasets)
- Local uploads: Several hours (5,134 files, depends on connection)
- Processing: Depends on dataset sizes

