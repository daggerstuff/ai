# Fine-Tuning, Evaluation & Deployment — Sprint 5

## Overview

Sprint 5 completes the memory system lifecycle by providing:
1. **Dataset preparation** — extract training pairs from memories, balance valence, split train/val/test
2. **Evaluation harness** — measure retrieval, response quality, safety, and performance
3. **Deployment packaging** — package models, create monitoring dashboards, define rollback plans

## Architecture

```
ai/memory/fine_tuning/
├── dataset_preparation.py   # DatasetPreparator: extract, balance, split, PII check
├── evaluation.py            # MemorySystemEvaluator: retrieval, response, safety, perf
├── deployment.py            # DeploymentPackager: package, monitor, rollback
└── __init__.py              # Public API exports

src/lib/memory/fine_tuning/
├── dataset-preparation.ts   # TypeScript mirror of DatasetPreparator
├── evaluation.ts            # TypeScript mirror of MemorySystemEvaluator
├── index.ts                 # Public exports
└── fine_tuning.test.ts      # Unit tests (6 tests, 97% coverage)
```

## Quick Start

### Python

```python
from ai.memory.fine_tuning import DatasetPreparator, MemorySystemEvaluator, DeploymentPackager

# 1. Prepare dataset
preparator = DatasetPreparator(train_ratio=0.7, val_ratio=0.15)
split, stats = preparator.prepare(memories)
print(f"{stats.total_examples} examples, PII leak: {stats.pii_leak_detected}")

# 2. Evaluate
evaluator = MemorySystemEvaluator(k=5)
report = evaluator.evaluate(memories)
print(f"Overall pass: {report.overall_pass}")

# 3. Deploy
packager = DeploymentPackager()
package = packager.create_package("rag_retriever", report)
issues = packager.validate_pre_deploy(package)
```

### TypeScript

```typescript
import { DatasetPreparator, MemorySystemEvaluator } from '@/lib/memory/fine_tuning';

const preparator = new DatasetPreparator();
const [split, stats] = preparator.prepare(memories);

const evaluator = new MemorySystemEvaluator(5);
const report = evaluator.evaluate(memories);
```

## Dataset Preparation

### Pipeline
1. **Extract** — convert MemoryBlocks to (query, response) training pairs
2. **Balance** — stratify by valence bucket (-1 to -0.6, -0.6 to -0.2, -0.2 to 0.2, 0.2 to 0.6, 0.6 to 1.0)
3. **Split** — 70/15/15 train/val/test
4. **Validate** — PII pattern detection (email, SSN, phone, credit card)

### Valence Balancing
Undersampled buckets are upsampled by repeating examples until they match the largest bucket. Over-sampled buckets are truncated.

### PII Detection
Regex patterns for email, SSN, phone numbers, and credit cards. If `pii_leak_detected` is True, the dataset must be sanitized before training.

## Evaluation Harness

### Metrics

| Category | Metric | Threshold |
|----------|--------|-----------|
| Retrieval | Precision@K | ≥ 0.75 |
| Retrieval | Recall@K | — |
| Retrieval | MRR | — |
| Response | Appropriateness | ≥ 0.80 |
| Response | Personalization | ≥ 0.70 |
| Response | Continuity | ≥ 0.75 |
| Safety | Crisis Sensitivity | ≥ 0.98 |
| Safety | PII Leak Rate | = 0.00 |
| Performance | P95 Latency | < 500ms |

### Overall Pass
All gates must pass simultaneously. The `overall_pass` boolean is the conjunction of all thresholds.

## Deployment

### Package Structure
```
deployment_package/
├── model_config.json      # Model parameters and version
├── evaluation_report.json # Full evaluation results
├── monitoring_config.json # Dashboard and alert definitions
└── rollback_plan.json     # Step-by-step rollback procedure
```

### Pre-Deploy Validation
- Evaluation must pass all gates
- No PII leaks in training data
- Package ID must be unique
- All required files present

### Monitoring Dashboard
Tracks: retrieval precision, response appropriateness, crisis sensitivity, raw memory ratio. Alert thresholds configurable.

### Rollback Plan
6-step procedure: detect degradation → halt traffic → restore previous version → verify health → notify stakeholders → post-mortem.

## Runbook

### Deploying a New Model Version

1. **Prepare dataset**: Run `DatasetPreparator.prepare()` on consolidated memories
2. **Train**: Fine-tune model using the prepared dataset (external training pipeline)
3. **Evaluate**: Run `MemorySystemEvaluator.evaluate()` — must pass all gates
4. **Package**: Run `DeploymentPackager.create_package()` — generates deployment package
5. **Validate**: Run `DeploymentPackager.validate_pre_deploy()` — must return empty issues list
6. **Deploy**: Upload package to model registry, update routing config
7. **Monitor**: Load dashboard via `DeploymentPackager.create_monitoring_dashboard()`

### Rollback Procedure

1. Detect degradation via monitoring alerts
2. Halt traffic to new version
3. Restore previous version from package registry
4. Verify health checks pass
5. Notify stakeholders
6. Conduct post-mortem analysis

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `overall_pass=False` with low P@K | Poor retrieval quality | Increase memory diversity, adjust embedding model |
| `pii_leak_detected=True` | PII in memory content | Sanitize memories before dataset preparation |
| High P95 latency | Slow retrieval | Reduce K, optimize embedding index |
| Low crisis sensitivity | Missing crisis flags | Ensure crisis detection in memory ingestion |
