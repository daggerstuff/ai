# Annotation & Labeling (Phase 1.3)

This directory contains the framework, guidelines, and tools for the "Annotation & Labeling" phase of the NGC Therapeutic Enhancement plan.

## Objectives

- Create a high-quality labeled dataset for Pixel model training.
- Achieve an inter-annotator agreement (Kappa) of > 0.85.
- Validate crisis detection and emotional intelligence labels.

## Directory Structure

- `guidelines.md`: The "Bible" for annotators, defining labels and criteria.
- `scripts/`: Python scripts for data management and calculating agreement metrics.
- `batches/`: Input JSONL files assigned to annotators.
- `results/`: Completed annotations from annotators.

## Workflow

1.  **Preparation**: Raw data is sampled from `ai/training_ready` and split into batches.
2.  **Assignment**: Batches are assigned to specific annotators (or simulated agents for testing).
3.  **Annotation**: Annotators follow `guidelines.md` to produce JSON output.
4.  **Validation**: `calculate_kappa.py` runs to check agreement.
5.  **Merger**: Verified annotations are merged back into the training dataset.

## Setup

Ensure you have the Python environment set up:

```bash
uv sync
```
