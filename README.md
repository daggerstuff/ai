 # Pixelated Empathy AI - Production Structure

## Overview

Production-ready AI system for empathy-aware conversational AI with proper
organization and deployment structure.

## Directory Structure

```
ai/
├── training/                    # Model training and fine-tuning
│   ├── configs/                # Training configurations
│   ├── checkpoints/            # Model checkpoints
│   ├── scripts/                # Training scripts
│   └── train_pixelated_empathy.py  # Main training entry point
├── data/                       # Datasets and data management
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed datasets
│   └── synthetic/              # Synthetic data generation
├── models/                     # Model artifacts and exports
│   ├── checkpoints/            # Saved model checkpoints
│   ├── artifacts/              # Model artifacts
│   └── exports/                # Exported models for deployment
├── inference/                  # Deployment and inference
│   ├── api/                    # API endpoints
│   ├── services/               # Inference services
│   ├── deployment/             # Deployment configurations
│   └── pixelated_empathy_inference.py  # Main inference entry point
├── config/                     # Configuration management
│   ├── production/             # Production configurations
│   ├── development/            # Development configurations
│   └── testing/                # Testing configurations
├── pipelines/                  # Data and model pipelines
│   ├── data_processing/        # Data processing pipelines
│   ├── model_training/         # Training pipelines
│   ├── evaluation/             # Model evaluation
│   └── process_datasets.py     # Main data processing entry point
├── research/                   # Research and experimentation
│   ├── notebooks/              # Jupyter notebooks
│   ├── experiments/            # Research experiments
│   └── analysis/               # Analysis and reports
├── tools/                      # Utilities and tools
│   ├── utilities/              # Utility scripts
│   ├── scripts/                # Shell scripts
│   └── generators/             # Data generators
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── guides/                 # User guides
│   └── architecture/           # Architecture documentation
├── qa/                         # Quality assurance
│   ├── reports/                # QA reports
│   ├── validation/             # Validation scripts
│   └── testing/                # Test suites
└── archive/                    # Legacy and archived files
    └── legacy_files/           # Old implementation files
```

## Quick Start

### Training

```bash
cd training
python train_pixelated_empathy.py
```

### Inference

```bash
cd inference
python pixelated_empathy_inference.py
```

### Data Processing

```bash
cd pipelines
python process_datasets.py
```

## Test Suite & Training Pipeline

### Python Test Suite

Run all training tests (excluding legacy book PDF converter tests):

```bash
uv run pytest training/tests/ --ignore=training/tests/test_book_pdf_converter.py -q
```

### Coverage

Coverage is enforced at 100 % across shared_config, the safety checkers, and reward_score:

```bash
uv run pytest training/tests/ --ignore=training/tests/test_book_pdf_converter.py \
  --cov=training.shared_config,training.multilingual_safety_checker,training.clinical_safety_checker,training.reward_score \
  --cov-branch --cov-fail-under=100 -q
```

### DPO / GRPO Smoke Tests

Lightweight smoke tests for direct preference optimization and group-relative policy optimization:

```bash
uv run pytest training/tests/test_dpo_trainer.py training/tests/test_grpo_trainer.py -q
```

### Training Pipeline Environment

The training pipeline runs on a CPU-only PyTorch stack. The `[tool.uv.sources]` section in
`pyproject.toml` pins `torch`, `torchvision`, and `torchaudio` to the `pytorch-cpu` index to
avoid pulling CUDA binaries that would fail on CPU-only runners.

### Safety Checker Environment Variables

Set the following environment variable in test/CI environments to disable ML model loading for
safety checkers (enabled by default in `conftest.py`):

```bash
export AI_DISABLE_SAFETY_ML_MODELS=1
```

## Production Deployment

### Configuration

- Production configs: `config/production/`
- Development configs: `config/development/`
- Testing configs: `config/testing/`

### Model Management

- Training checkpoints: `training/checkpoints/`
- Model artifacts: `models/artifacts/`
- Exported models: `models/exports/`

### Data Management

- Raw datasets: `data/raw/`
- Processed datasets: `data/processed/`
- Synthetic data: `data/synthetic/`

### API Deployment

- API endpoints: `inference/api/`
- Deployment configs: `inference/deployment/`
- Service implementations: `inference/services/`

## Development Workflow

1. **Data Preparation**: Use `pipelines/process_datasets.py`
2. **Model Training**: Use `training/train_pixelated_empathy.py`
3. **Model Evaluation**: Use scripts in `pipelines/evaluation/`
4. **Deployment**: Use configurations in `inference/deployment/`
5. **Monitoring**: Use tools in `qa/validation/`

## Legacy Files

All previous task files and reports have been moved to `archive/legacy_files/`
for reference.

## Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/development/.env.example config/development/.env
```

## Support

- Documentation: `docs/`
- API Reference: `docs/api/`
- Architecture Guide: `docs/architecture/`
