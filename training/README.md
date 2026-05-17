<!-- markdownlint-disable -->\n\n# Pixelated Empathy Training Pipeline

This directory contains the core training scripts for the Pixel model (Phase 3).

## 🚀 Quick Start

To run the training pipeline (supports Unsloth optimization if installed):

```bash
# From project root
uv run python training/train_pixel.py --config training/ready_packages/configs/hyperparameters/enhanced_training_config.json
```

## 🏗️ Architecture

The pipeline supports two modes:

1.  **⚡ Unsloth (Preferred)**: Uses `unsloth.FastLanguageModel` for 2x faster
    training and 60% less VRAM.
2.  **🐢 HuggingFace (Fallback)**: Standard `transformers` + `peft`
    implementation for compatibility.

## 📄 Configuration

The training is driven by `enhanced_training_config.json` which defines:

- **Base Model**: LatitudeGames/Wayfarer-2-12B (or similar)
- **LoRA Parameters**: Rank 16, Alpha 32 (default)
- **Component Weights**: Specific weights for the 6 therapeutic components
- **H100 Optimizations**: BF16, Flash Attention 2

## 🧪 Verification

To verify the pipeline works (even without the full dataset):

1.  Ensure you have a GPU available (or modify script for CPU/MPS if debugging
    logic only).
2.  Run the command above.
3.  If `ULTIMATE_FINAL_DATASET.jsonl` is missing, the script will automatically
    generate dummy data to test the flow.
