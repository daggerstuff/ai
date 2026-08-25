# Curator Configs

This directory is a **placeholder**. NeMo Curator does not carry its own
product configs in this workspace.

## Where do the product configs live?

The product-level dataset configs are shared with Data Designer and live at
the parent path (relative to the repo root):

```
scripts/data/designer/configs/
├── _bootstrap.py
├── therapeutic_sft.py
├── long_running_therapy.py
├── cptsd_dialogues.py
├── edge_cases.py
├── crisis_safety.py
├── dpo_preferences.py
└── knowledge_tasks.py
```

These configs define the dataset shape (columns, validators, sampler
params) used by both stages:

- **Stage 1 — Curator** uses them to know which seed corpora to curate and
  what the expected output schema looks like.
- **Stage 2 — Data Designer** uses them to declare the synthetic generation
  pipeline that extends the curated seeds.

## Why no symlink?

The configs directory is a sibling of this workspace under a different
parent tree (`scripts/data/designer/` at the repo root vs. `ai/tools/`).
A symlink would break on machines with different repo layouts. Reference
the parent path directly.

## Adding Curator-specific configs

If Curator-specific curation parameters (filter thresholds, dedup
config, PII redaction rules) are needed in the future, add them here as
YAML or Python modules. Until then, this placeholder keeps the directory
tracked in git.
