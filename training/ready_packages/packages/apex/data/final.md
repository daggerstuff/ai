---
name: Final training dataset + curriculum audit
overview: "Audit S3-hosted training sources against required dataset sets, then produce a canonical final dataset artifact (manifest + compiled export) and a late-2025 multi-stage training curriculum: continued pretraining → SFT curriculum → preference alignment."
status: "~70% Complete - See AUDIT_REAL_STATUS.md for verified findings"
todos:
  - id: inventory-map
    content: Build a dataset-family inventory from S3 manifest + dataset registry; generate coverage report for all required sets.
    status: completed
    evidence: "data/dataset_coverage_report.json, data/dataset_routing_config.json"
  - id: define-contract
    content: Define the final dataset schema, provenance tracking, and split/holdout rules (manifest + compiled export).
    status: completed
    evidence: "docs/FINAL_DATASET_CONTRACT.md"
  - id: dedup-leakage
    content: Implement post-encoding-fix dedup (exact + near-dup) and prevent cross-split leakage.
    status: completed
    evidence: "scripts/enhanced_deduplication.py, data/FULL_DEDUPLICATION_SUMMARY.md"
  - id: generators-missing
    content: Ensure edge-case resulting chats, sarcasm, roleplay/simulator, and preference-pair datasets exist and are routable.
    status: mostly_complete
    evidence: "Most generators exist; long_running_therapy needs extraction, preference pairs may need explicit generator"
  - id: compile-export
    content: Compile final ChatML JSONL export + S3 manifest/shards and upload to canonical S3 paths.
    status: completed
    evidence: "scripts/compile_final_dataset.py (now auto-uploads to S3 and removes local copies)"
  - id: curriculum
    content: Design late-2025 training phases (continued pretrain → SFT curriculum → preference alignment) with dataset-to-phase routing and weights.
    status: completed
    evidence: "configs/training_curriculum_2025.json, docs/TRAINING_CURRICULUM_2025.md"
  - id: verify
    content: Run coverage + leakage + distribution gates; produce final stats artifacts for signoff.
    status: completed
    evidence: "scripts/verify_final_dataset.py, coverage/leakage gates implemented"
---

## Goal

Produce a *real* final training dataset (not the misleading “ultimate/final” labels) by auditing what exists in S3 and what the pipelines can generate, then emitting a canonical dataset manifest + export and a phased training plan.

## Current facts found in repo

- `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md` defines canonical S3 organization and stage mapping.
- `ai/training_ready/data/s3_manifest.json` is a large object inventory for the OVH S3 endpoint and contains evidence of CPTSD/addiction + voice transcript corpora (e.g., Tim Fletcher transcripts) and crisis JSONL.
- `ai/data/dataset_registry.json` is an internal registry that maps datasets to stages (foundation, CoT, edge crisis, voice persona) and explicitly references an edge case generator output.
- “ULTIMATE_FINAL…” JSON summaries exist but are not authoritative per your correction.

## Deliverables

- **Dataset coverage report**: checklist-style pass/fail for each required set you listed (edge-case generator outputs + resulting chats + synthetic; long-running therapy sessions; mental health datasets; video transcripts/persona; sarcasm; niche CPTSD/addiction; experimental; roleplay/simulator designer + DPO data).
- **Final dataset artifact (both)**:
- **Manifest** in S3: pointers to shards/splits + hashes + per-source provenance.
- **Compiled export** in S3: a single ChatML JSONL for portability.
- **Training curriculum (late 2025)**: continued pretraining → SFT curriculum by stage → preference alignment phase, with dataset-to-phase routing and mixing weights.

## Approach

### 1) Source-of-truth inventory + mapping

- Treat `ai/training_ready/data/s3_manifest.json` as the *ground truth inventory* of what’s available in S3.
- Build a mapping layer from:
- S3 object prefixes (e.g., `datasets/gdrive/...`, `tier*_...`, `voice/...`, `edge...`) → canonical “dataset families”.
- Registry entries in `ai/data/dataset_registry.json` → required families + stage routing.
- Output a **coverage matrix**: required family → S3 evidence (paths/keys) → status (present/partial/missing) → next action.

### 2) Define the “final dataset” contract

- **Schema**: ChatML JSONL with strict metadata: `{source_family, source_key, content_hash, pii_status, license_tag, split, phase}`.
- **Provenance**: maintain an immutable mapping file from every output row → originating S3 object key(s).
- **Splits**:
- Train/val/test with at least one *hard holdout* for: long-session therapy, edge-case/crisis, sarcasm, and “voice persona transcripts”.

### 3) Data quality + dedup strategy (post-encoding-fix)

- Run encoding normalization first (you’re already doing this work).
- Dedup in two layers:
- **Exact**: stable normalized-text hash (current approach referenced by `ai/training_ready/data/FULL_DEDUPLICATION_SUMMARY.md`).
- **Near-dup**: add semantic/approx similarity pass for high-risk leakage across splits (especially long sessions and edge cases).
- Decide canonical vs redundant consolidated files (e.g., “unified vs priority_1/2/3”) before compilation.

### 4) Build/extend dataset generators needed for missing sets

- Ensure pipelines exist and can emit S3-ready artifacts for:
- **Edge case generator** outputs + “resulting chats” format.
- **Synthetic/roleplay/simulator designer** datasets suitable for SFT and for preference training pairs.
- **Sarcasm** and “niche criteria” bundles (CPTSD/addiction) as first-class families with holdouts.

### 5) Training phases (you selected continued-pretrain-plus + unrestricted)

- **Phase A: Continued pretraining**
- Domain-adaptive text (video transcripts, general psych/mental health text) with careful filtering for privacy/PII.
- **Phase B: Multi-stage SFT curriculum**
- Foundation therapeutic conversations → long-session therapy emphasis → edge-case/crisis robustness → voice/persona style → roleplay/simulator tasks.
- Use mixture weights + length-aware packing to preserve long sessions.
- **Phase C: Preference alignment**
- Choose algorithm (DPO/ORPO/SimPO/KTO) based on available preference data shape; generate/curate preference pairs from roleplay/simulator + adversarial prompts.
- Since you chose “unrestricted”, still keep a **hard privacy constraint**: aggressive PII stripping and exclusion.

### 6) Verification gates

- Coverage gate: all required families present (or explicitly waived) before final compile.
- Leakage gate: no near-duplicates across splits for holdout families.
- Reporting gate: counts/tokens by family, by phase, and by split.

## Key files this plan anchors to

- `ai/training_ready/data/s3_manifest.json` ✅ Exists (19,323 objects)
- `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md` ✅ Exists
- `ai/data/dataset_registry.json` ⚠️ Check location (may be in `ai/data/`)
- `ai/training_ready/packages/velocity/data_pipeline/integrated_training_pipeline.py` ✅ Exists (different location)
- `ai/training_ready/scripts/enhanced_deduplication.py` ✅ Exists (replaces deleted `full_deduplication_scan.py`)
- `ai/training_ready/data/FULL_DEDUPLICATION_SUMMARY.md` ✅ Exists

## Implementation todos

- **inventory-map**: Parse `s3_manifest.json` into dataset families + evidence paths; emit coverage matrix.
- **define-contract**: Specify final dataset schema + provenance + split rules; add holdout definitions.
- **dedup-leakage**: Implement exact + near-dup checks and split-leakage prevention.
- **generators-missing**: Wire/extend generators for edge-case resulting chats, sarcasm, roleplay/simulator, and preference-pair creation.
- **compile-export**: Produce manifest + compiled ChatML JSONL export and upload to canonical S3 location.
- **curriculum**: Produce phase routing + mixing weights and training config(s) for phases A/B/C.
- **verify**: Run coverage/leakage/stats gates and record outputs.