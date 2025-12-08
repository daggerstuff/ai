# S3-First Execution Order (Merged)

This sequence blends NEXT_STEPS, README, ENV_QUICKSTART, and IMPLEMENTATION_STATUS into one dependency-aware flow.

## Ordered Steps
1) Environment & Access
- Install deps via `./install_dependencies.sh` (or `uv pip install …`).
- Set OVH/AWS S3 env vars; confirm `uv` shell is active.
- Sanity checks: Torch import, S3 list, HF/Kaggle auth if needed.

2) Baseline Validation
- Inspect `TRAINING_MANIFEST.json` summary and read `TRAINING_PLAN.md`.

3) S3 Ingestion Foundation
- Implement `S3DatasetLoader` (streaming JSON/JSONL), add tests.
- Add S3 path resolution helpers.

4) Pipeline S3 Wiring
- Update `source_datasets.py` to prefer S3 then local.
- Update `process_all_datasets.py` and `assemble_final_dataset.py` to use S3 loader/paths.

5) Manifest S3 Mapping
- Write `scripts/update_manifest_s3_paths.py` to map local→S3 and update manifest + mapping file.

6) End-to-End S3 Verification
- Smoke test loader on sample HF dataset stored in S3.
- Run `scripts/prepare_training_data.py --report` using S3 paths; verify integrity and stage balances.

7) VPS Readiness
- Author `scripts/vps_environment_setup.sh`; document notes in `VPS_SETUP_COMPLETE.md`.
- On VPS, rerun deps and S3 checks; dry-run pipeline.

8) Documentation Refresh
- Update `README.md`, `TRAINING_PLAN.md`, `GET_UP_TO_SPEED.md` (if used), and add an S3 usage guide to reflect S3-first flow and env vars.

9) Final Validation
- Re-run quick verification commands (manifest, pipeline help flags).
- Note disk-space/manifest-reference caveats and PYTHONPATH guidance for `ai/dataset_pipeline` imports.

## Task Checklist
- [ ] Environment & access: deps installed, S3/HF/Kaggle creds set, sanity checks run
- [ ] Baseline validation: manifest summary reviewed; training plan skimmed
- [ ] S3DatasetLoader implemented with streaming + tests
- [ ] S3 path resolution helpers added
- [ ] `source_datasets.py` updated for S3-first resolution
- [ ] `process_all_datasets.py` updated to read via S3 loader
- [ ] `assemble_final_dataset.py` updated to read via S3 loader
- [ ] `scripts/update_manifest_s3_paths.py` created and manifest/mapping updated
- [ ] Loader smoke-tested on S3-hosted sample dataset
- [ ] Full pipeline `prepare_training_data.py --report` run against S3 sources
- [ ] `scripts/vps_environment_setup.sh` written; `VPS_SETUP_COMPLETE.md` documented
- [ ] Docs refreshed (README, TRAINING_PLAN, GET_UP_TO_SPEED, S3 usage guide)
- [ ] Final verification commands re-run; known issues noted
