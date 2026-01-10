# Training Ready: Consolidated Training System

## Overview

This directory contains all consolidated training assets for the Pixelated Empathy model, organized and ready for immediate training deployment. All training configs, datasets, models, pipelines, and infrastructure have been cataloged and organized from 20+ AI directories across the codebase.

## Quick Start

### 0. Install Dependencies

For local execution with CPU-only torch:

```bash
# Option A: Using the install script (recommended)
cd ai/training_ready
./install_dependencies.sh

# Option B: Using uv directly
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option C: Using pip (system-wide)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note**: If using `uv`, run scripts with `uv run`:
```bash
uv run python3 ai/training_ready/scripts/prepare_training_data.py --all
```

### 1. Review the Training Manifest

```bash
cat TRAINING_MANIFEST.json | jq '.summary'
```

The manifest contains:
- **9,608 total assets** cataloged
- **5,208 datasets** mapped to 4-stage architecture
- **2,145 model architectures**
- **707 training configurations**
- **223 pipeline components**
- **89 infrastructure configs**
- **509 experimental features**

### 2. Review the Training Plan

```bash
cat TRAINING_PLAN.md
```

The training plan outlines:
- 4-stage training architecture (40/25/20/15)
- Model selection (Harbringer-24B/Mistral Small 3.1)
- Dataset strategy per stage
- Infrastructure deployment options
- Timeline and success metrics

### 3. Start Training

#### Option A: Lightning.ai (Recommended)

```bash
cd ai/lightning_training_package
python scripts/train_enhanced.py --config config/enhanced_training_config.json
```

#### Option B: Kubernetes

```bash
kubectl apply -f infrastructure/kubernetes/
helm install pixel-training infrastructure/helm/
```

#### Option C: Local Docker

```bash
docker build -f infrastructure/docker/Dockerfile -t pixel-training .
docker run --gpus all pixel-training
```

## Directory Structure

```
ai/training_ready/
├── README.md                          # This file
├── TRAINING_MANIFEST.json            # Complete asset inventory
├── TRAINING_PLAN.md                  # Comprehensive training strategy
├── configs/                          # All training configs
│   ├── stage_configs/               # Stage-specific configs
│   ├── model_configs/                # Model architecture configs
│   ├── infrastructure/               # K8s, Helm, deployment
│   └── hyperparameters/              # Hyperparameter configs
├── datasets/                         # Consolidated datasets
│   ├── stage1_foundation/            # Stage 1 datasets (40%)
│   ├── stage2_reasoning/             # Stage 2 datasets (25%)
│   ├── stage3_edge/                  # Stage 3 edge cases (20%)
│   └── stage4_voice/                 # Stage 4 voice data (15%)
├── models/                           # Model architectures
│   ├── moe/                          # MoE architecture
│   ├── base/                         # Base models
│   └── experimental/                 # Research models
├── pipelines/                        # Training pipelines
│   ├── integrated/                  # Integrated pipeline
│   ├── edge/                         # Edge case pipeline
│   └── voice/                        # Voice training pipeline
├── infrastructure/                   # Deployment infrastructure
│   ├── kubernetes/                   # K8s manifests
│   ├── helm/                         # Helm charts
│   └── docker/                       # Docker configs
├── tools/                            # Training utilities
│   ├── data_preparation/             # Data prep scripts
│   ├── validation/                   # Validation tools
│   └── monitoring/                   # Monitoring tools
├── experimental/                     # Experimental features
│   ├── research_models/              # Research architectures
│   ├── novel_pipelines/              # Experimental pipelines
│   ├── future_features/              # Future enhancements
│   └── UPGRADE_OPPORTUNITIES.md      # Experimental features doc
└── scripts/                          # Consolidation scripts
    ├── explore_directories.py       # Directory exploration
    ├── generate_manifest.py          # Manifest generation
    ├── create_folder_structure.py   # Folder structure creation
    ├── consolidate_assets.py        # Asset consolidation
    └── output/                       # Script outputs
        ├── directory_catalogs.json  # Directory catalogs
        └── experimental_features.json # Experimental features
```

## Stage-Based Training

### Stage 1: Foundation & Rapport (40%)
- **Objective**: Baseline therapeutic tone and empathy
- **Datasets**: Foundation datasets, tier1_priority
- **Success Metrics**: Empathy ≥ 0.70, Safety ≥ 0.80

### Stage 2: Therapeutic Expertise & Reasoning (25%)
- **Objective**: Structured reasoning and clinical knowledge
- **Datasets**: CoT reasoning datasets, professional psychology
- **Success Metrics**: Reasoning ≥ 0.75, Clinical accuracy ≥ 0.80

### Stage 3: Edge Stress Test (20%)
- **Objective**: Crisis response and trauma handling
- **Datasets**: Edge cases, crisis scenarios, trauma data
- **Success Metrics**: Crisis response ≥ 0.85, Edge success ≥ 0.80

### Stage 4: Voice & Persona (15%)
- **Objective**: Authentic voice and persona consistency
- **Datasets**: Voice data, wayfarer-balanced, persona training
- **Success Metrics**: Voice authenticity ≥ 0.80, Persona consistency ≥ 0.85

## Key Files

### Training Manifest
- **Location**: `TRAINING_MANIFEST.json`
- **Purpose**: Complete inventory of all training assets
- **Format**: JSON with structured metadata
- **Use**: Reference for dataset selection, asset discovery

### Training Plan
- **Location**: `TRAINING_PLAN.md`
- **Purpose**: Comprehensive training strategy and timeline
- **Contents**: Stage breakdown, model selection, infrastructure, metrics
- **Use**: Guide for training execution

### Upgrade Opportunities
- **Location**: `experimental/UPGRADE_OPPORTUNITIES.md`
- **Purpose**: Document experimental features for future integration
- **Contents**: MoE, CNN/ResNet layers, quantum models, etc.
- **Use**: Evaluate future model enhancements

## Dataset Access

### Large Files
Due to disk space constraints, large dataset files use symlinks or references. Access original files from their source locations as documented in `TRAINING_MANIFEST.json`.

### Stage Mapping
All datasets are mapped to stages in the manifest:
- `stage1_foundation`: Foundation datasets
- `stage2_reasoning`: Reasoning and CoT datasets
- `stage3_edge`: Edge cases and crisis scenarios
- `stage4_voice`: Voice and persona datasets

## Experimental Features

See `experimental/UPGRADE_OPPORTUNITIES.md` for:
- MoE architecture for specialized experts
- CNN/ResNet emotional layers
- Quantum-inspired emotional models
- Neuroplasticity layers
- Causal emotional reasoning

## Troubleshooting

### Disk Space Issues
- Large files use symlinks - access from original locations
- Check `TRAINING_MANIFEST.json` for original file paths
- Consider using dataset references instead of copies

### Missing Files
- Check `TRAINING_MANIFEST.json` for file paths
- Some files may be in original directories (see manifest)
- Run `explore_directories.py` to refresh catalog

### Configuration Issues
- Check `configs/` directory for training configs
- Reference original configs in source directories if needed
- See `TRAINING_PLAN.md` for infrastructure setup

## Next Steps

1. **Review Training Plan**: Understand the 4-stage training strategy
2. **Check Infrastructure**: Set up Lightning.ai, K8s, or Docker
3. **Validate Datasets**: Ensure all stage datasets are accessible
4. **Begin Training**: Start with Stage 1 foundation training
5. **Monitor Progress**: Track metrics and adjust as needed

## Documentation Links

- **Training Manifest**: `TRAINING_MANIFEST.json`
- **Training Plan**: `TRAINING_PLAN.md`
- **Upgrade Opportunities**: `experimental/UPGRADE_OPPORTUNITIES.md`
- **Stage Configuration**: `../dataset_pipeline/configs/stages.py`
- **Master Plan**: `../../.notes/pixel/pixel_master_plan-V3.md`

## Support

For questions or issues:
1. Check `TRAINING_MANIFEST.json` for asset locations
2. Review `TRAINING_PLAN.md` for training strategy
3. Consult `experimental/UPGRADE_OPPORTUNITIES.md` for experimental features
4. Reference original source directories as documented in manifest

## Summary

This consolidated training system provides:
- ✅ **Complete asset inventory** (9,608 files cataloged)
- ✅ **4-stage training architecture** (40/25/20/15 distribution)
- ✅ **Comprehensive training plan** with timelines and metrics
- ✅ **Experimental feature documentation** for future upgrades
- ✅ **Ready-to-use structure** for immediate training deployment

**Status**: Ready for training deployment

