# AI Repo Reorganization Log

This file tracks the major structural changes made to the `ai/` repository to
eliminate duplication and nesting.

## 1. Promotions from `ai/training/ready_packages/`

- **Documentation**: `ready_packages/docs/` -> `ai/training_ready/docs/`
- **Ready Data**: `ready_packages/training_ready/` -> `ai/training_ready/data/`
- **Datasets**: `ready_packages/datasets/` -> `ai/training_ready/data/datasets/`
- **Manifest**: `ready_packages/TRAINING_MANIFEST.json` -> `ai/training_ready/TRAINING_MANIFEST.json`
- **Training Scripts**:
  - `ready_packages/train_unsloth_lora.py` -> `ai/training/train_unsloth_lora.py`
  - `ready_packages/packages/apex/scripts/train_enhanced.py` ->
    `ai/training/train_enhanced.py`
  - `ready_packages/packages/velocity/training_scripts/train_optimized.py` ->
    `ai/training/train_optimized.py`
- **Configs/Utils/Models**: Merged into `ai/training/configs/`,
  `ai/training/utils/`, and `ai/training/models/`.

## 2. Promotions from `ai/pipelines/`

- **Pipelines**: All subdirectories (alignment, analytics, api, therapies,
  etc.) promoted from `ai/pipelines/` to `ai/pipelines/`.
- **Nesting Removed**: The redundant `orchestrator/` level in `ai/pipelines/`
  has been eliminated.

## 3. Deployment & Readiness

- **Lightning**: `ai/orchestrator/targets/` persists as the canonical home for
  deployment readiness reports and environment-specific logic.

## 4. Refactoring & Cleanup

- **Import Updates**: All Python imports updated from
  `ai.pipelines.orchestrator.*` to `ai.pipelines.*` and
  `ai.training.ready_packages.*` to `ai.training.*`.
- **String Paths**: All references to old directory paths in Python and
  Markdown files fixed.
- **Deduplication**: Removed redundant `ready_packages/` and `orchestrator/`
  directories.
- **Dead-weight**: Eliminated duplicate `train_moe_h100_consolidated.py` in
  favor of the more robust `train_moe_h100.py`.

## 5. Verification

- Main entry point `ai/main.py` verified for correct pathing and functionality.
- Directory structure conforms to `AGENTS.md` topology.
