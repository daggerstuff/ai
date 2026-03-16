# AI Architecture & Reorganization Plan

This document outlines the consolidated architecture for the Pixelated Empathy AI codebase. The goal is to eliminate redundancy, clarify component responsibilities, and ensure all subsystems are utilized effectively.

## 🎯 Target Architecture

The codebase handles 5 main domains: Sourcing, Processing, Modeling, Training, and Infrastructure.

### 1. `ai/sourcing/` (Data Ingestion)
**Goal:** Unified entry point for all raw data acquisition.
*   `academic/` (Formerly `ai/sourcing/academic`) - Discovery of academic papers.
*   `journal/` (Formerly `ai/sourcing/journal`) - Journal crawling and analysis.
*   `research_system/` (Formerly `ai/sourcing/research_system`) - Legacy research orchestration (Evaluate for merge).

### 2. `ai/pipelines/` (Data Transformation)
**Goal:** Specialized pipelines for generating/transforming specific data types.
*   `voice/` (Merged `ai/models/pixel_core_voice` and `ai/pipelines/voice`) - Audio processing pipeline.
*   `edge_case/` (Derived from `ai/pipelines/edge_case` and `ai/pipelines/design`) - Generating difficult scenarios.
*   `dual_persona/` (From `ai/pipelines/dual_persona`) - Persona switching generation.
*   `orchestrator/` (Formerly `ai/pipelines/orchestrator`) - **The Master Pipeline**. Integrates all other pipelines into the final dataset.
    *   *Note: This folder previously contained everything. It is now focused on integration.*

### 3. `ai/models/components/` (Core Intelligence)
**Goal:** Model definitions and weights.
*   `foundation/` (Formerly `ai/models/foundation`) - Base model configurations (e.g., NeMo, Llama).
*   `components/` (Formerly `ai/models/components`) - Reusable model components (classifiers, evaluators).
*   `pixel/` (Formerly `ai/models/pixel_core`) - The specific "Pixel" agent logic and runtime.

### 4. `ai/training/` (Execution)
**Goal:** Scripts and configs for the actual training runs.
*   `ready_packages/` (Formerly `ai/training/ready_packages`) - Final dataset tarballs and export scripts.
*   `configs/` - Axolotl/Unsloth configurations.

### 5. `ai/infrastructure/` (Support Systems)
**Goal:** Operational support.
*   `distributed/` (Formerly `ai/infrastructure/distributed`) - Celery/Ray workers.
*   `database/` (Formerly `ai/database`) - SQL/Vector schemas.
*   `production/` (Formerly `ai/infrastructure/production`) - Deployment manifests.
*   `qa/` (Formerly `ai/infrastructure/qa`) - Validation suites and quality gates.

### 6. `ai/data/` (Assets)
**Goal:** clean storage for local data (ignored by git).
*   `voice_logs/` (Formerly `ai/data/voice_logs`)
*   `interim/`

---

## 🔄 Migration Map

| Old Location | New Location | Status |
| :--- | :--- | :--- |
| `ai/sourcing/academic` | `ai/sourcing/academic` | ➡️ Move |
| `ai/sourcing/journal` | `ai/sourcing/journal` | ➡️ Move |
| `ai/sourcing/research_system` | `ai/sourcing/research_system` | ➡️ Move |
| `ai/pipelines/orchestrator` | `ai/pipelines/orchestrator` | ➡️ Rename |
| `ai/pipelines/design` | `ai/pipelines/edge_case/designer` | ➡️ Move |
| `ai/infrastructure/distributed` | `ai/infrastructure/distributed` | ➡️ Move |
| `ai/models/foundation` | `ai/models/components/foundation` | ➡️ Move |
| `ai/infrastructure/integration` | `ai/infrastructure/integration` | ➡️ Move |
| `ai/pipelines/orchestrator/integration_legacy` | `ai/pipelines/orchestrator/integration` | ➡️ Merge |
| `ai/analysis/notebooks` | `ai/analysis` | ➡️ Rename |
| `ai/models/pixel_core` | `ai/models/components/pixel_core` | ➡️ Move |
| `ai/models/pixel_core_voice` | `ai/pipelines/voice` | ➡️ Merge |
| `ai/pipelines` | `ai/pipelines/*` | ➡️ Distribute |
| `ai/infrastructure/production` | `ai/infrastructure/production` | ➡️ Move |
| `ai/infrastructure/qa` | `ai/infrastructure/qa` | ➡️ Move |
| `ai/training/ready_packages` | `ai/training/ready_packages` | ➡️ Move |
| `ai/data/voice_logs` | `ai/data/voice_logs` | ➡️ Move |

## 🛠 Usage Guidelines

### How to Run the Pipeline
Use the orchestrator to run the full flow:
```bash
cd ai/pipelines/orchestrator
python main_orchestrator.py
```

### How to Add a New Source
1. Add sourcing script to `ai/sourcing/`.
2. Register it in `ai/pipelines/orchestrator/sourcing/`.
