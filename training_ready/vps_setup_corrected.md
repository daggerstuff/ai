# VPS Setup - Corrected for Proper Structure

## ✅ Corrected Location

The files have been moved to the correct location to match your repo structure:
- **Location**: `/home/vivi/pixelated/ai/training_ready/`
- **Matches local structure**: Same as your local machine
- **Uses existing environment**: Leverages your zshrc config and project setup

## 📋 Setup Steps (Using zsh)

### 1. Navigate to Correct Location

```bash
cd /home/vivi/pixelated/ai/training_ready
```

### 2. Source Your zshrc (for environment setup)

```bash
source ~/.zshrc
```

This will load:
- uv (if installed)
- Any project-specific PATH settings
- Other environment configurations

### 3. Check Environment

```bash
# Check Python
python3 --version

# Check uv (if available)
which uv
uv --version

# Check if we're in the right place
pwd  # Should be: /home/vivi/pixelated/ai/training_ready
```

### 4. Install Dependencies

Since you're in the project root structure, you can use the project's dependency management:

```bash
# If using uv (recommended)
cd /home/vivi/pixelated
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install boto3 datasets kaggle requests

# Or use the install script (it will detect uv)
cd /home/vivi/pixelated/ai/training_ready
./install_dependencies.sh
```

### 5. Configure OVH Object Storage (S3-compatible)

```bash
# Canonical OVH object storage config
export OVH_S3_BUCKET="pixel-data"
export OVH_S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
export OVH_S3_REGION="us-east-va"
export OVH_S3_ACCESS_KEY="your-ovh-access-key"
export OVH_S3_SECRET_KEY="your-ovh-secret-key"

# Convenience alias so existing scripts that expect S3_BUCKET keep working
export S3_BUCKET="$OVH_S3_BUCKET"
```

### 6. Verify Catalog

```bash
cd /home/vivi/pixelated/ai/training_ready
python3 -c "
import json
with open('scripts/output/dataset_accessibility_catalog.json') as f:
    d = json.load(f)
    print(f'Total: {d[\"summary\"][\"total\"]}')
    print(f'Local-only: {d[\"summary\"][\"local_only\"]}')
    print(f'HuggingFace: {d[\"summary\"][\"huggingface\"]}')
"
```

### 7. Download Remote Datasets to OVH S3

```bash
cd /home/vivi/pixelated/ai/training_ready

# Using uv (if available)
uv run python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json

# Or using python3 directly
python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json
```

### 8. Upload Local-Only Datasets (from LOCAL machine, to OVH S3)

```bash
# On your LOCAL machine
cd /home/vivi/pixelated
aws s3 sync \
  --endpoint-url "$OVH_S3_ENDPOINT" \
  --exclude "*" \
  --include-from ai/training_ready/scripts/output/local_only_upload_list.txt \
  / \
  s3://$S3_BUCKET/datasets/local/
```

## 🎯 Advantages of Correct Structure

1. **Environment consistency**: Same paths as local, same configs
2. **zshrc benefits**: Your zshrc configs are loaded automatically
3. **Project context**: Can access other project files if needed
4. **Dependency management**: Can use project's uv/pip setup
5. **Path resolution**: Scripts can find relative imports correctly

## 📚 Quick Commands

```bash
# Always use zsh for remote commands
ssh -i ~/.ssh/planet vivi@146.71.78.184 "zsh -c 'cd /home/vivi/pixelated/ai/training_ready && <command>'"

# Or SSH in and work interactively
ssh -i ~/.ssh/planet vivi@146.71.78.184
# Then:
cd /home/vivi/pixelated/ai/training_ready
source ~/.zshrc
```

## 🔍 Verify Setup

```bash
cd /home/vivi/pixelated/ai/training_ready
source ~/.zshrc

# Check structure
pwd  # Should be: /home/vivi/pixelated/ai/training_ready
ls -la | head -10

# Check catalog
test -f scripts/output/dataset_accessibility_catalog.json && echo "✅ Catalog found" || echo "❌ Catalog missing"

# Check Python
python3 --version

# Check dependencies
python3 -c "import torch; print('✅ torch available')" 2>/dev/null || echo "⚠️  torch not installed"
python3 -c "import boto3; print('✅ boto3 available')" 2>/dev/null || echo "⚠️  boto3 not installed"
```

