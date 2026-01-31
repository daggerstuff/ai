# Environment Quickstart: Training Ready Pipeline

## Prerequisites

- Python 3.11+
- `uv` package manager (project standard)
- AWS account with S3 access
- (Optional) Kaggle account for Kaggle datasets
- (Optional) HuggingFace account for private datasets

## Quick Setup

### 1. Install Core Dependencies

```bash
cd ai/training_ready
./install_dependencies.sh
```

Or manually:
```bash
# CPU-only torch (for local)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Data processing
uv pip install datasets kaggle boto3 requests
```

### 2. Configure AWS/S3

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # or your preferred region
export S3_BUCKET="your-bucket-name"
```

Or use AWS CLI:
```bash
aws configure
```

### 3. (Optional) Configure Kaggle

```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

### 4. (Optional) Configure HuggingFace

```bash
# For private datasets
export HF_TOKEN="your-hf-token"

# Or login
huggingface-cli login
```

## Verification

### Test Torch
```bash
uv run python3 -c "import torch; print(f'Torch {torch.__version__} - CPU: {not torch.cuda.is_available()}')"
```

### Test S3 Access
```bash
python3 -c "import boto3; s3 = boto3.client('s3'); print('S3 buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])"
```

### Test HuggingFace
```bash
uv run python3 -c "from datasets import load_dataset; print('HF available')"
```

### Test Kaggle
```bash
python3 -c "import kaggle; print('Kaggle available')"
```

## Environment Variables Summary

```bash
# Required
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="your-bucket-name"

# Optional
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
export HF_TOKEN="..."
```

## Running Scripts

### With uv (Recommended)
```bash
uv run python3 scripts/prepare_training_data.py --all
```

### Direct Python
```bash
python3 scripts/prepare_training_data.py --all
```

## Common Issues

### Torch not found
- Use `uv run` to ensure correct environment
- Or install system-wide: `uv pip install --system torch ...`

### AWS credentials not found
- Set environment variables
- Or use `aws configure`
- Or use IAM role (on EC2)

### Kaggle API errors
- Verify `~/.kaggle/kaggle.json` exists
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Verify API key is valid

### S3 permission errors
- Verify IAM policy allows S3 read/write
- Check bucket name is correct
- Verify region matches bucket region

## Dependencies List

### Core
- `torch>=2.0.0` (CPU or GPU)
- `torchvision`
- `torchaudio`

### Data Sources
- `datasets>=2.14.0` (HuggingFace)
- `kaggle>=1.5.0` (Kaggle API)

### Cloud Storage
- `boto3>=1.26.0` (AWS S3)

### Utilities
- `requests>=2.28.0` (HTTP downloads)
- `numpy>=1.24.0`
- `pandas>=2.0.0`

## Project Dependencies

The main project `pyproject.toml` includes many of these. Check:
```bash
uv pip list | grep -E "torch|datasets|kaggle|boto3"
```


