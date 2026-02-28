# AI Repository Architecture

> This document reflects the final, consolidated structure after the
> February 2026 reorganization.

## Top-Level Domains

```bash
ai/
├── core/           # The Engine — pipelines, APIs, servicing
├── data/           # The Library — datasets, manifests, registries
├── docs/           # Documentation
├── infra/          # The Foundation — infra, docker, monitoring, safety
├── lab/            # The Sandbox — evals, demos, experiments, archive
├── logs/           # Runtime output logs
├── orchestrator/   # The Brain — entry point and deployment targets
└── training/       # The Forge — models, configs, train scripts
```

---

## `ai/orchestrator` — The Brain

Entry point and high-level pipeline control.

- `main.py` — Primary entry point (`uv run python -m ai.orchestrator.main`)
- `pipeline_orchestrator.py` — Full pipeline graph
- `pipeline_runner.py` — Stage execution harness
- `training_orchestrator.py` — Training-specific orchestration
- `targets/` — Deployment environment packages
  - `lightning_production/` — Lightning.ai production config
  - `lightning_h100/` — H100-specific deployment config

---

## `ai/core` — The Engine

All live system logic. Modules here are imported by the orchestrator.

- `pipelines/` — Data transformation (alignment, synthesis, therapies, etc.)
- `sourcing/` — Data acquisition agents (ArXiv, YouTube, HuggingFace)
- `api/` — REST API endpoints and integration bridges
- `annotation/` — Labelling and human-in-the-loop tooling
- `memory/` — Conversation memory (Mem0)
- `detection/` — Behavioural filters (`psydefdetect`)
- `cli/` — Command line interface
- `tools/` — Shared agent tools
- `nemo/` — NVIDIA NeMo integration
- `multimodal/` — Multimodal processing
- `scripts/` — Operational scripts (S3, preprocessing, etc.)
- `bin/` — Binary and compiled assets

---

## `ai/training` — The Forge

Model architecture and training loops. Self-contained; does not import
from `core/`.

- `train_pixel.py` — Standard HuggingFace training
- `train_moe_h100.py` — MoE H100 training
- `train_enhanced.py`, `train_optimized.py`, `train_unsloth_lora.py`
- `models/` — Architecture definitions
- `configs/` — Hyperparameter configurations
- `scripts/` — Data preparation and upload scripts
- `defense_mechanisms/` — Adversarial training helpers
- `rlhf/` — Reward model / RLHF components
- `v1/` — Legacy v1 training artefacts (kept for reference)

---

## `ai/data` — The Library

All persistent data artefacts and registries.

- `datasets/` — Final training datasets (by stage)
- `raw/voice/` — Tim Fletcher and other voice transcripts
- `raw/youtube/` — YouTube video transcriptions
- `research/` — Journal and sourcing research notes
- `pipeline/` — Dataset pipeline definitions
- `db/` — Database schemas and migrations
- `dataset_manifest.json`, `dataset_registry.json` — Canonical registries
- `TRAINING_MANIFEST.json` — Package manifest for OVH/cloud uploads

---

## `ai/infra` — The Foundation

Deployment, scaling, observability, and security.

- `cloud/` — Distributed compute (formerly `infrastructure/`)
- `docker/` — Dockerfile and Compose configs
- `helm/` — Kubernetes Helm charts
- `monitoring/` — Prometheus, Grafana, alerting
- `deployment/` — Deploy scripts and configs
- `autoscaling/` — Auto-scaling policies
- `safety/` — Safety rails and content filters
- `security/` — Security audit tooling
- `compliance/` — HIPAA / regulatory checks
- `config/` — Shared environment configuration

---

## `ai/lab` — The Sandbox

Research, evaluation, and experimentation. Nothing here is load-bearing.

- `evals/` — Model evaluation harnesses
- `analysis/` — Data analysis notebooks
- `demos/` — Demo scripts and apps
- `examples/` — Usage examples
- `experimental/` — Speculative / in-progress experiments
- `archive/` — Historical artefacts
- `outputs/` — Generated artefacts from experiments

---

## Entry Point

```bash
# Run the full pipeline
uv run python -m ai.orchestrator.main
```
