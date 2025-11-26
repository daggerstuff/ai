# OVH AI Training - Wayfarer-2-12B

> Supervised fine-tuning of Wayfarer-2-12B for therapeutic AI using OVHcloud AI Platform.

## Overview

This directory contains the OVHcloud AI Training integration for the Pixelated Empathy project. It adapts the existing training pipeline (`ai/pipelines/wayfarer_supervised.py`) for the OVH AI Platform.

### Training Strategy: Staged Fine-Tuning

Based on `ai/data/acquired_datasets/TRAINING_DATASET_GUIDE.md`:

1. **Stage 1: Foundation** - Natural therapeutic dialogue patterns
   - Dataset: `mental_health_counseling.json` (3,512 conversations)
   - Epochs: 3, Learning Rate: 2e-4

2. **Stage 2: Reasoning** - Clinical reasoning patterns (Chain of Thought)
   - Dataset: `cot_reasoning.json` (300 conversations)
   - Epochs: 2, Learning Rate: 1e-4

3. **Stage 3: Voice** - Tim Fletcher teaching style/personality
   - Dataset: Synthetic conversations from Tim Fletcher transcripts
   - Epochs: 2, Learning Rate: 5e-5

## Directory Structure

```
ai/ovh/
├── train_ovh.py           # Main training script (from wayfarer_supervised.py)
├── training-job.yaml      # OVH job configurations for each stage
├── sync-datasets.sh       # Upload/download datasets to/from OVH Object Storage
├── run-training.sh        # Helper script to launch training jobs
├── Dockerfile.training    # Docker image for training
├── Dockerfile.inference   # Docker image for inference
├── inference_server.py    # FastAPI inference server
├── deploy-inference.sh    # Deploy inference app to OVH AI Deploy
└── README.md              # This file
```

## Prerequisites

1. **OVH Account**: Public Cloud project in US-EAST-VA region
2. **ovhai CLI**: Installed and authenticated

```bash
# Install CLI
curl -sSL https://cli.us-east-va.ai.cloud.ovh.us/install.sh | bash
export PATH="$HOME/bin:$PATH"

# Authenticate
ovhai login
```

## Quick Start

### 1. Check Dataset Inventory

```bash
./ai/ovh/sync-datasets.sh inventory
```

Shows available local datasets:
- `ai/data/acquired_datasets/` - CoT reasoning, mental health counseling
- `ai/data/lightning_h100/` - Expert-specific training data
- `ai/data/tim_fletcher_voice/` - Voice profile data
- `/mnt/gdrive/datasets/` - Google Drive (if mounted)

### 2. Upload Datasets to OVH

```bash
# Upload all local datasets
./ai/ovh/sync-datasets.sh upload

# Check what was uploaded
./ai/ovh/sync-datasets.sh list
```

### 3. Build & Push Docker Image

```bash
./ai/ovh/run-training.sh build
```

### 4. Launch Training Job

```bash
# Run all stages (full training)
./ai/ovh/run-training.sh run

# Or run specific stage
./ai/ovh/run-training.sh run --stage foundation
./ai/ovh/run-training.sh run --stage reasoning
./ai/ovh/run-training.sh run --stage voice
```

### 5. Monitor Training

```bash
# List jobs
ovhai job list

# View logs
ovhai job logs <job-id>

# Monitor GPU/memory
ovhai job describe <job-id>
```

## Dataset Sources

### Local Datasets (`ai/data/`)

| Dataset | Path | Records | Description |
|---------|------|---------|-------------|
| CoT Reasoning | `acquired_datasets/cot_reasoning.json` | 300 | Chain of thought clinical reasoning |
| Mental Health Counseling | `acquired_datasets/mental_health_counseling.json` | 3,512 | Real therapeutic conversations |
| Expert Therapeutic | `lightning_h100/expert_therapeutic.json` | Varies | Expert-specific training data |
| Tim Fletcher Voice | `tim_fletcher_voice/` | TBD | Voice profile & synthetic conversations |

### Google Drive Datasets (if mounted)

Registry: `ai/data/dataset_registry.json`

| Category | Datasets | Focus |
|----------|----------|-------|
| CoT Reasoning | Clinical Diagnosis, Heartbreak, Neurodivergent, Men's Mental Health, Cultural Nuances | Clinical reasoning patterns |
| Professional Therapeutic | SoulChat2.0, Counsel-Chat, Psych8k, Therapist-SFT | Licensed therapist responses |
| Priority | priority_1_FINAL, priority_2_FINAL, priority_3_FINAL | Top-tier curated conversations |

## OVH Object Storage Structure

```
s3://pixelated-training-data/
├── acquired/
│   ├── cot_reasoning.json
│   └── mental_health_counseling.json
├── lightning/
│   ├── expert_therapeutic.json
│   ├── expert_empathetic.json
│   ├── expert_practical.json
│   ├── expert_educational.json
│   └── train.json
├── voice/
│   ├── tim_fletcher_voice_profile.json
│   └── synthetic_conversations.json
├── gdrive/                   # Optional: Google Drive datasets
│   ├── CoT_Reasoning.../
│   └── priority/
└── config/
    ├── dataset_registry.json
    └── moe_training_config.json

s3://pixelated-checkpoints/
├── foundation/
│   └── final/
├── reasoning/
│   └── final/
├── voice/
│   └── final/
└── final_model/
```

## GPU Flavors

| Flavor | GPU | VRAM | Use Case |
|--------|-----|------|----------|
| `gpu_nvidia_l40s_1` | L40S | 48GB | **Recommended** - Good balance |
| `gpu_nvidia_h100_1` | H100 | 80GB | Best performance |
| `gpu_nvidia_a100-80gb_1` | A100 | 80GB | High performance |

## Environment Variables

### Required for Training

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | Weights & Biases API key |
| `HF_TOKEN` | HuggingFace token (for model access) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `OVH_REGION` | `US-EAST-VA` | OVH region |
| `DATA_BUCKET` | `pixelated-training-data` | Data bucket name |
| `CHECKPOINT_BUCKET` | `pixelated-checkpoints` | Checkpoint bucket name |

## Azure Pipelines Integration

The `azure-pipelines.yml` includes an `OVHAITraining` stage that triggers on:

```yaml
parameters:
  - name: TRIGGER_AI_TRAINING
    type: boolean
    default: false
```

Trigger training:
```bash
az pipelines run --name "pixelated-pipeline" --parameters TRIGGER_AI_TRAINING=true
```

Required Azure DevOps secrets:
- `OVH_AI_TOKEN` - OVH AI Platform token
- `WANDB_API_KEY` - Weights & Biases key
- `HF_TOKEN` - HuggingFace token

## Monitoring

### WandB Dashboard

Training metrics are logged to Weights & Biases:
- Loss curves
- Learning rate schedule
- GPU utilization
- Checkpoint evaluations

### OVH Job Monitoring

```bash
# Real-time logs
ovhai job logs -f <job-id>

# Job status
ovhai job get <job-id>

# List all jobs
ovhai job list --filter "project:pixelated-empathy"
```

## Troubleshooting

### Common Issues

1. **"No data found for stage"**
   - Run `./sync-datasets.sh upload` to ensure data is on OVH
   - Check bucket contents with `./sync-datasets.sh list`

2. **Out of Memory (OOM)**
   - Reduce `per_device_batch_size` in config
   - Use larger GPU flavor (H100 instead of L40S)
   - Enable gradient checkpointing (already enabled by default)

3. **Job stuck in "Pending"**
   - Check GPU availability: `ovhai resources`
   - Try different region or flavor

4. **Authentication errors**
   - Re-run `ovhai login`
   - Check token expiration

### Getting Help

```bash
# CLI help
ovhai --help
ovhai job run --help

# Documentation
open https://docs.ovh.com/us/en/ai-training/
```

## Related Files

- `ai/pipelines/wayfarer_supervised.py` - Original training pipeline
- `ai/data/dataset_registry.json` - Full dataset inventory
- `ai/data/acquired_datasets/TRAINING_DATASET_GUIDE.md` - Training strategy guide
- `ai/lightning/moe_training_config.json` - MoE architecture config
