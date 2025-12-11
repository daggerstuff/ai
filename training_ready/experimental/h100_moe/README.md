# H100 MoE Upgrade (Therapeutic AI Package v5.0)

This folder links the production-ready H100 MoE training bundle (`ai/therapeutic_ai_training_package_20251028_204104`) into the `training_ready` workflow so we can run the upgraded training, inference, and progress-tracking stack without hunting for files.

## What you get
- H100-optimized training scripts: `train_optimized.py`, `train_moe_h100.py`, `training_optimizer.py`
- MoE model + inference pieces: `moe_architecture.py`, `inference_optimizer.py`, `inference_service.py`
- Progress tracking: `therapeutic_progress_tracker.py`, `progress_tracking_api.py`
- Specialized loaders + orchestrator: `integrated_training_pipeline.py` and edge/dual-persona/voice/psychology loaders
- Config + deps: `configs/moe_training_config.json`, `configs/requirements_moe.txt`
- Docs: `docs/LIGHTNING_H100_QUICK_DEPLOY.md`, `docs/QUICK_START_GUIDE.md`

## Where the source lives
All upgrade assets remain in place to avoid duplication:
```
ai/therapeutic_ai_training_package_20251028_204104/
├── training_scripts/…              # training/inference/progress scripts
├── configs/moe_training_config.json
├── configs/requirements_moe.txt
├── data_pipeline/…                 # loaders + integrated orchestrator
├── docs/…                          # quick deploy guides
└── utils/logger.py
```

## Run the H100 MoE path with training_ready data
1) Prepare a dataset JSON the scripts expect (single `conversations` list of `{text}` entries). You can export from our unified pipeline, e.g.:
```
uv run python ai/training_ready/pipelines/integrated/assemble_final_dataset.py \
  --output ai/therapeutic_ai_training_package_20251028_204104/training_dataset.json
```
Adjust the export path/flags to match your actual integrated output.

2) Create minimal configs alongside the dataset (same directory):
- `training_config.json` (example):
```json
{
  "base_model": "LatitudeGames/Wayfarer-2-12B",
  "num_train_epochs": 3,
  "optimization_priority": "balanced",
  "max_training_hours": 12,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "learning_rate": 3e-4,
  "warmup_steps": 1000
}
```
- `safety_config.json`: include any guardrails you require; a minimal stub is:
```json
{
  "toxicity_threshold": 0.2,
  "enable_bias_checks": true,
  "enable_suicide_watch": true
}
```
- `wandb_config.json`:
```json
{
  "project": "pixelated-therapeutic",
  "entity": null,
  "name": "h100-moe",
  "tags": ["moe", "h100", "lora"],
  "notes": "H100 MoE training via training_ready",
  "config": {}
}
```

3) Install the H100-specific dependencies (from repo root):
```
uv pip install -r ai/therapeutic_ai_training_package_20251028_204104/configs/requirements_moe.txt
```

4) Launch training (from the package dir so relative imports work):
```
cd ai/therapeutic_ai_training_package_20251028_204104
uv run python training_scripts/train_optimized.py
# or: uv run python training_scripts/train_moe_h100.py
```

5) Serve the trained model:
```
uv run python training_scripts/inference_service.py
```

## Notes
- The scripts assume H100 class GPUs and will exit if CUDA is unavailable.
- Keep the dataset and configs in the package directory (or pass explicit paths if you modify the scripts).
- Progress tracking and checkpointing are enabled by default (30-minute checkpoints, WandB logging).
- The package defaults to `LatitudeGames/Wayfarer-2-12B` with 8k context and 4-expert MoE; adjust `training_config.json` if you need a different base model or batch shape.
