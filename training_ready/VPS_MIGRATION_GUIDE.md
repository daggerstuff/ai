# VPS Migration Guide (Consolidated)

One source of truth for moving `ai/training_ready` to a VPS with OVH S3.

## Quick Path (VPS)
```bash
cd /path/to/workspace
tar -xzf training_ready_vps*.tar.gz
cd ai/training_ready

# Install deps (uses uv if available)
./install_dependencies.sh

# Set OVH object storage creds
export OVH_S3_BUCKET="pixel-data"
export OVH_S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
export OVH_S3_REGION="us-east-va"
export OVH_S3_ACCESS_KEY="your-ovh-access-key"
export OVH_S3_SECRET_KEY="your-ovh-secret-key"
export S3_BUCKET="$OVH_S3_BUCKET"   # convenience for scripts
```

## Migration Steps
1. **Catalog** (already generated in `scripts/output/dataset_accessibility_catalog.json`; rerun if needed):
   ```bash
   python3 scripts/catalog_local_only_datasets.py
   ```
2. **Download remote (HF/Kaggle/URL) → S3**:
   ```bash
   python3 scripts/download_to_s3.py \
     --bucket $S3_BUCKET \
     --catalog scripts/output/dataset_accessibility_catalog.json
   ```
3. **Upload local-only datasets (run from LOCAL machine)**:
   ```bash
   aws s3 sync \
     --endpoint-url "$OVH_S3_ENDPOINT" \
     --exclude "*" \
     --include-from ai/training_ready/scripts/output/local_only_upload_list.txt \
     / \
     s3://$S3_BUCKET/datasets/local/
   ```
4. **Run pipeline (S3-aware)**:
   ```bash
   uv run python3 scripts/prepare_training_data.py --all
   ```

## VPS Checklist (merged)
- Tarball extracted under `/home/vivi/pixelated/ai/training_ready`
- Dependencies installed (`./install_dependencies.sh`)
- OVH S3 env vars set (`OVH_S3_*`, `S3_BUCKET`)
- Catalog present (`scripts/output/dataset_accessibility_catalog.json`)
- HF datasets downloaded to S3
- Local-only uploads started/completed
- Manifest updated with `s3_path`
- Pipeline run succeeds against S3

## Correct Location & Env Notes
- Work from `/home/vivi/pixelated/ai/training_ready` (matches local paths).
- Source shell profile if needed for uv/paths: `source ~/.zshrc`.
- Verify basics:
  ```bash
  python3 --version
  which uv
  ```

## S3 Structure (target)
```
s3://pixel-data/datasets/
├── huggingface/
├── kaggle/
├── urls/
└── local/
```

## Troubleshooting
- **Creds**: ensure `OVH_S3_ACCESS_KEY`, `OVH_S3_SECRET_KEY`, bucket/region correct.
- **Uploads slow**: rely on resumable `aws s3 sync`; run off-peak.
- **HF private**: set `HF_TOKEN`.
- **Imports**: always run via `uv run python3 ...`.

## Pointers
- Status: see `S3_MIGRATION_STATUS.md`.
- Pipeline S3 behavior: `PIPELINE_S3_INTEGRATION.md`.


