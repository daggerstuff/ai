# VPS Migration - Quick Start

## 🚀 What's Ready

Everything you need to migrate the training consolidation system to your VPS with S3 storage.

### Package Contents

1. **Tarball**: `training_ready_vps_*.tar.gz` (~1.2GB)
   - Complete `ai/training_ready/` system
   - Essential `ai/dataset_pipeline/` dependencies
   - All scripts and documentation

2. **Documentation**:
   - `GET_UP_TO_SPEED.md` - Quick context for new sessions
   - `VPS_MIGRATION_GUIDE.md` - Step-by-step migration instructions
   - `ENV_QUICKSTART.md` - Environment setup guide
   - `VPS_MIGRATION_CHECKLIST.md` - Complete checklist
   - `MIGRATION_SUMMARY.md` - What was done and why

3. **Scripts**:
   - `scripts/catalog_local_only_datasets.py` - Identify local vs remote datasets
   - `scripts/download_to_s3.py` - Download remote datasets to S3
   - `create_vps_tarball.sh` - Recreate tarball if needed

4. **Dataset Catalog**:
   - `scripts/output/dataset_accessibility_catalog.json` - Full catalog
   - `scripts/output/local_only_upload_list.txt` - Files to upload

## 📊 Dataset Breakdown

- **Total**: 5,208 datasets
- **Local-only** (need upload): 5,134 (98.6%)
- **HuggingFace** (direct to S3): 74 (1.4%)
- **Kaggle**: 0
- **URLs**: 0

## 🎯 Migration Strategy

### Why This Approach?

Your local connection is too slow for large uploads. Solution:
- **Remote datasets** (74 HF): Download directly from source → S3 (bypasses local/VPS)
- **Local-only datasets** (5,134): Upload from local machine → S3 (only these need slow upload)

### Workflow

```
Local Machine:
  1. Catalog datasets → Identify what's local vs remote
  2. Create tarball → Package code/configs (no datasets)
  3. Upload tarball to VPS

VPS:
  4. Extract tarball
  5. Set up environment
  6. Download 74 HuggingFace datasets → S3 directly

Local Machine:
  7. Upload 5,134 local-only files → S3

VPS:
  8. Run data processing pipeline
  9. Generate final training datasets
```

## 📦 Quick Start

### 1. Upload Tarball to VPS
```bash
scp training_ready_vps_*.tar.gz user@vps:/path/to/workspace/
```

### 2. On VPS: Extract and Set Up
```bash
cd /path/to/workspace
tar -xzf training_ready_vps_*.tar.gz
cd ai/training_ready

# Read the guides
cat GET_UP_TO_SPEED.md
cat VPS_MIGRATION_GUIDE.md

# Install dependencies
./install_dependencies.sh

# Configure AWS
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="your-bucket-name"
```

### 3. Download Remote Datasets to S3
```bash
python3 scripts/download_to_s3.py \
  --bucket $S3_BUCKET \
  --catalog scripts/output/dataset_accessibility_catalog.json
```

### 4. From Local: Upload Local-Only Datasets
```bash
aws s3 sync \
  --exclude "*" \
  --include-from ai/training_ready/scripts/output/local_only_upload_list.txt \
  / \
  s3://$S3_BUCKET/datasets/local/
```

### 5. On VPS: Run Pipeline
```bash
uv run python3 scripts/prepare_training_data.py --all
```

## 📚 Documentation Guide

- **New to this?** → Start with `GET_UP_TO_SPEED.md`
- **Ready to migrate?** → Follow `VPS_MIGRATION_GUIDE.md`
- **Setting up environment?** → See `ENV_QUICKSTART.md`
- **Need a checklist?** → Use `VPS_MIGRATION_CHECKLIST.md`
- **Want details?** → Read `MIGRATION_SUMMARY.md`

## ✅ Pre-Flight Checklist

Before uploading tarball:
- [ ] Tarball created and verified
- [ ] AWS credentials ready
- [ ] S3 bucket name known
- [ ] AWS region known
- [ ] VPS access confirmed

After extraction on VPS:
- [ ] Tarball extracted
- [ ] Dependencies installed
- [ ] AWS credentials configured
- [ ] S3 bucket accessible
- [ ] Catalog verified (74 HF, 5,134 local)

## 🔧 Troubleshooting

**Tarball too large?**
- Should be ~1.2GB (code + configs only)
- Datasets are NOT included (will be in S3)

**Import errors?**
- Use `uv run python3` to ensure correct environment
- Check that `ai/dataset_pipeline/` is in tarball

**S3 access issues?**
- Verify AWS credentials
- Check IAM permissions
- Verify bucket name and region

**Slow uploads?**
- Only 5,134 local-only files need upload
- 74 HuggingFace datasets download directly (fast)
- Use `aws s3 sync` with resume capability

## 📞 Next Steps

1. Upload tarball to VPS
2. Extract and read `GET_UP_TO_SPEED.md`
3. Follow `VPS_MIGRATION_GUIDE.md`
4. Download remote datasets to S3
5. Upload local-only datasets from local machine
6. Run pipeline on VPS

---

**Everything is ready. Good luck with the migration! 🚀**

