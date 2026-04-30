# H100 MoE Upgrade (Therapeutic AI Package v5.0)

This folder documents the consolidated H100 MoE training system now located in `ai/training_ready/`. All training, inference, and progress-tracking scripts have been consolidated into the canonical `training_ready` structure.

## What you get
- H100-optimized training scripts: `scripts/train_optimized.py`, `scripts/train_moe_h100.py`, `scripts/training_optimizer.py`
- MoE model + inference pieces: `models/moe_architecture.py`, `scripts/inference_optimizer.py`, `scripts/inference_service.py`
- Progress tracking: `models/therapeutic_progress_tracker.py`, `scripts/progress_tracking_api.py`
- Specialized loaders + orchestrator: `pipelines/integrated_training_pipeline.py` and edge/dual-persona/voice/psychology loaders
- Config + deps: `configs/moe_training_config.json`, `configs/requirements_moe.txt`
- Docs: `docs/LIGHTNING_H100_QUICK_DEPLOY.md`, `docs/QUICK_START_GUIDE.md`

## Where the source lives
All assets are now consolidated in `ai/training_ready/`:
```
ai/training_ready/
├── scripts/…                      # training/inference/progress scripts
├── configs/moe_training_config.json
├── configs/requirements_moe.txt
├── pipelines/…                     # loaders + integrated orchestrator
├── docs/…                          # quick deploy guides
└── utils/logger.py
```

## Run the H100 MoE path with training_ready data
1) Prepare a dataset JSON the scripts expect (single `conversations` list of `{text}` entries). You can export from our unified pipeline, e.g.:
```
uv run python ai/training_ready/pipelines/integrated_training_pipeline.py \
  --output ai/training_ready/data/training_dataset.json
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
uv pip install -r ai/training_ready/configs/requirements_moe.txt
```

4) Launch training (from training_ready dir so relative imports work):
```
cd ai/training_ready
uv run python scripts/train_optimized.py
# or: uv run python scripts/train_moe_h100.py
```

5) Serve the trained model:
```
uv run python scripts/inference_service.py
```

## Notes
- The scripts assume H100 class GPUs and will exit if CUDA is unavailable.
- Keep the dataset and configs in the package directory (or pass explicit paths if you modify the scripts).
- Progress tracking and checkpointing are enabled by default (30-minute checkpoints, WandB logging).
- The package defaults to `LatitudeGames/Wayfarer-2-12B` with 8k context and 4-expert MoE; adjust `training_config.json` if you need a different base model or batch shape.
