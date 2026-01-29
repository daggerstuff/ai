# S3 Audit & Local Storage Report

**Generated**: 2026-01-28
**Scope**: `pixelated/ai` vs `s3://pixel-data` (OVH)

## 🚨 Critical Findings

The local `ai` directory is consuming **33GB**, primarily driven by files that are **NOT** present on S3.

| Storage Category | Local Size | S3 Status | Notes |
| :--- | :--- | :--- | :--- |
| **Lightning Checkpoints** | ~11 GB | ❌ **MISSING** | `wayfarer-balanced` checkpoints (200MB+ each) are local only. |
| **Training Data** | ~5.6 GB | ❌ **MISSING** | `ULTIMATE_FINAL_DATASET.jsonl` and variants are local only. |
| **Raw Archives** | ~4 GB | ❌ **MISSING** | `RealLifeDeceptionDetection.2016.zip`, `glove.840B.300d.zip` are local only. |
| **Database** | ~800 MB | ❌ **MISSING** | `conversations.db` |
| **VPS Archaeology** | N/A | ✅ **ON S3** | `datasets/vps_archaeology` exists on S3 but not locally (good). |

## 📂 Detailed S3 Status

### 1. Missing Large Files (Candidates for Upload)
These files exist locally but were not found in the S3 bucket:

**Models & Checkpoints (`ai/lightning/`):**
- `pixelated/ai/lightning/pixelated-training/wayfarer-balanced/checkpoint-*/adapter_model.safetensors`
- `pixelated/ai/lightning/pixelated-training/wayfarer-balanced/checkpoint-*/optimizer.pt`
- `pixelated/ai/lightning/h100_deployment/therapeutic_ai_h100_deployment_*.zip`

**Datasets (`ai/training_ready/data/`):**
- `ULTIMATE_FINAL_DATASET.jsonl` (>100MB)
- `ULTIMATE_FINAL_DATASET_cleaned.jsonl`
- `final_corpus/merged_dataset_raw.jsonl` (Local is newer/larger than S3 equivalent)

**Archives (`ai/data/compress/`):**
- `RealLifeDeceptionDetection.2016.zip`
- `glove.840B.300d.zip`

### 2. Existing S3 Content
The S3 bucket `pixel-data` currently contains:
- `datasets/vps_archaeology/`: Extensive archive of older Reddit datasets.
- `processed/`:
  - `pixelated_tier1_priority_curated_dark_humor.jsonl` (1.2GB)
  - `pixelated_tier2_...` through `tier6`
- `gdrive/processed/long_running_therapy/`
- `releases/v2026-01-07/RELEASE_0_UNIFIED_MANIFEST.json`

## 📋 Manifest Analysis
The file `ai/training_ready/TRAINING_MANIFEST.json` currently references **Local Paths** (e.g., `/home/vivi/pixelated/ai/...`).
It does **not** map to S3 URIs, confirming that the system setup is currently detached from S3 for these artifacts.

## 🛠 Recommended Actions
1. **Upload Checkpoints**: Move `lightning/pixelated-training` checkpoints to `s3://pixel-data/models/checkpoints/`.
2. **Upload Datasets**: Move `ULTIMATE_FINAL_DATASET` artifacts to `s3://pixel-data/datasets/production/`.
3. **Upload Archives**: Move zip archives to `s3://pixel-data/archives/`.
4. **Cleanup Local**: Once verified on S3, delete these local files to reclaim ~20-25GB.
5. **Update Manifest**: Regenerate manifest to point to S3 URIs (using `s3_dataset_loader.py` logic).
