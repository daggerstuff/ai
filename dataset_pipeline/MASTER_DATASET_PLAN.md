# Master Dataset Consolidation Plan - Pixelated Empathy

**Version:** 1.0  
**Last Updated:** December 2025  
**Status:** Comprehensive Blueprint for Multi-Stage Training Dataset

---

## Executive Summary

This document provides the complete blueprint for consolidating all Pixelated Empathy training datasets into a cohesive, multi-stage training pipeline. The plan leverages the operational `ai/dataset_pipeline/` system, preserves critical edge case training data (without safety filtering), and defines a staged curriculum from foundation to voice/personality injection.

**Critical Principle:** Edge case data is intentionally difficult/scary/horrible for training purposes. Guardrails that filter these out would destroy training value. Edge cases must pass through WITHOUT safety filtering, tagged with `is_training_edge_case: true`.

---

## Table of Contents

1. [Dataset Harmonization & Dedup Strategy](#dataset-harmonization--dedup-strategy)
2. [Edge Case Preservation Rules](#edge-case-preservation-rules)
3. [Quality Gates & Validation](#quality-gates--validation)
4. [Multi-Stage Training Layout](#multi-stage-training-layout)
5. [Data Source Integration](#data-source-integration)
6. [Execution Strategy](#execution-strategy)
7. [Agent Instructions](#agent-instructions)

---

## Dataset Harmonization & Dedup Strategy

### Operational Pipeline Foundation

The `ai/dataset_pipeline/` is now operational per completed epic with:
- **Export capabilities**: JSONL and Parquet formats
- **Manifest generation**: SHA256 checksums, dataset counts, config commit SHA, random seed
- **QA reports**: semantic coherence, safety flags, PII detection
- **Storage integration**: S3/GCS connectors for raw/processed/export/checkpoints
- **Training orchestrator**: `ai/dataset_pipeline/training_orchestrator.py` for H100 execution
- **6-stage pipeline**: Orchestration via `ai/dataset_pipeline/orchestration/pipeline_orchestrator.py`

### Harmonization Process

**Stage 1: Normalization**
- Use `ai/dataset_pipeline/schemas/conversation_schema.py` for unified format
- Convert all inputs (JSONL/JSON/CSV/PDF transcription) to standard `Conversation` schema
- Standardize role labels: `user/client` → `user`, `assistant/therapist` → `assistant`

**Stage 2: Tier Processing**
- Leverage `ai/dataset_pipeline/orchestration/tier_processor.py` for cluster handling
- Process datasets by tier (1-6) with appropriate quality thresholds
- Apply tier-weighted sampling via `ai/dataset_pipeline/composition/tier_balancer.py`

**Stage 3: Deduplication**
- Use Bloom-filter deduplication from `ai/dataset_pipeline/ingestion_deduplication.py`
- Hash conversation content using SHA-256 for exact duplicate detection
- Canonical role label normalization before hashing
- Metadata tagging for source attribution and lineage tracking

**Stage 4: Crossover Management**
- Tag datasets with source categories to prevent cross-contamination
- Filter rules in `ai/data/training_policy_manifest.json`:
  - Priority datasets → Stage 1 only
  - Reddit safety data → Stage 5 (Safety/DPO), not Stage 1
  - Edge cases → Stage 3 only (bypass all safety filters)

### Export Format

**Primary Formats:**
- **JSONL**: Line-delimited JSON for streaming processing
- **Parquet**: Columnar format for efficient querying and analysis

**Manifest Requirements:**
- SHA256 checksums for all files
- Dataset counts by source and stage
- Config commit SHA for reproducibility
- Random seed for deterministic processing
- Quality metrics (empathy scores, coherence scores, edge case tags)

---

## Edge Case Preservation Rules

### CRITICAL REQUIREMENT

Edge case data is intentionally difficult/scary/horrible for training purposes. This is a **mental health TRAINING LLM** that must learn to handle the worst-case scenarios. Guardrails that filter these out would make training worthless.

### Edge Case Categories (25 total)

**Very High Difficulty:**
- Suicidality
- Homicidal ideation
- Psychotic episodes
- Child abuse reporting
- Severe dissociation

**High Difficulty:**
- Substance abuse crisis
- Trauma flashbacks
- Borderline crisis
- Domestic violence
- Eating disorders

**Moderate Difficulty:**
- Paranoid accusations
- Medication refusal
- Family conflicts
- Adolescent defiance
- Couples betrayal

### Preservation Rules

1. **Tag-based Bypass**: All edge cases must be tagged with `is_training_edge_case: true` metadata
2. **No Safety Filtering**: Edge cases bypass ALL safety filters (crisis detection, toxicity, PII blocking)
3. **Source Tracking**: Edge cases come from:
   - `ai/lightning/ghost/edge_case_pipeline_standalone/` (edge case generator)
   - Google Drive scenario prompts
   - Reddit edge case dialogues
   - Synthetic edge case expansion datasets

4. **Quality Validation Only**: Edge cases still undergo:
   - Format validation (schema compliance)
   - Coherence scoring (conversation flow)
   - Therapeutic appropriateness (is it useful for training?)
   - **NO** safety/threat filtering

5. **Pipeline Modification**: Update quality validation pipeline to check `is_training_edge_case` tag and skip safety filters when present

### Implementation

Modify `ai/dataset_pipeline/quality/` validators to:
```python
def validate_conversation(conversation: Conversation) -> ValidationResult:
    # Check for edge case tag
    is_edge_case = conversation.metadata.get("is_training_edge_case", False)
    
    if is_edge_case:
        # Skip safety filters, only validate format/coherence
        return validate_format_and_coherence(conversation)
    else:
        # Full validation including safety filters
        return full_validation(conversation)
```

---

## Quality Gates & Validation

### Quality Metrics (Non-Edge-Case Data)

**Stage 1: Foundation**
- Empathy score: ≥0.7 (via `ai/dataset_pipeline/quality/empathy_mental_health_validator.py`)
- Therapeutic appropriateness: Pass clinical accuracy checks
- Semantic coherence: ≥0.8 average
- Safety compliance: Pass all safety gates (PII, toxicity, crisis handling)

**Stage 2: Reasoning**
- Reasoning completeness: ≥0.8 (CoT structure validation)
- Clinical accuracy: Evidence-based reasoning patterns
- Empathy score: ≥0.6 (lower threshold for reasoning focus)

**Stage 3: Edge Cases**
- Format validation: Schema compliance only
- Coherence scoring: Conversation flow validation
- Therapeutic value: Is this scenario useful for training?
- **NO safety filtering** (by design)

**Stage 4: Voice/Personality**
- Voice consistency: Personality marker alignment
- Naturalness score: Authentic conversational flow
- Empathy score: ≥0.6

**Stage 5: Safety/DPO**
- Safety alignment: DPO preference pair validation
- Production readiness: Appropriate for deployment
- Preference quality: Clear chosen/rejected distinctions

### Validation Tools

- **Empathy Scoring**: `ai/dataset_pipeline/quality/empathy_mental_health_validator.py`
  - EMNLP 2020 empathy dimensions (ER, IP, EX, VAL)
  - Overall empathy score: 0-1 scale
  
- **Safety Validation**: `ai/dataset_pipeline/quality/safety_alignment_validator.py`
  - Crisis content detection
  - Toxic language detection
  - PII exposure prevention
  - **BYPASSED for edge cases**

- **Coherence Validation**: `ai/dataset_pipeline/quality/coherence_validator.py`
  - Semantic alignment scoring
  - Question-answer relevance
  - Therapeutic appropriateness

---

## Multi-Stage Training Layout

### Stage 1: Foundation (40% of training data)

**Purpose:** Natural therapeutic dialogue patterns, empathy, rapport building

**Datasets:**
- Curated Wendy datasets (Tier 1): prefer descriptive names (no `FINAL`/`MASTER`, no numeric “priority” labels), e.g.
  - `wendy_set_alpha_therapeutic_core.jsonl`
  - `wendy_set_beta_high_quality_core.jsonl`
  - `wendy_set_gamma_specialized_therapy.jsonl`
- Professional therapeutic (Tier 2):
  - SoulChat 2.0
  - Counsel Chat
  - LLAMA3 Mental Counseling
  - Therapist SFT
  - Neuro QA SFT
  - Mental Health Counseling Conversations
  - Psych8k

**Quality Thresholds:**
- Empathy: ≥0.7
- Safety: ≥0.7
- Bias: ≤0.2

**Training Config:**
- Epochs: 3
- Learning Rate: 2e-4
- Checkpoint: Foundation base model

---

### Stage 2: Reasoning (20% of training data)

**Purpose:** Clinical reasoning patterns, Chain-of-Thought thinking

**Datasets:**
- CoT Reasoning Clinical Diagnosis Mental Health (38MB, 30K+ entries)
- CoT Neurodivergent vs Neurotypical Interactions
- CoT Heartbreak and Breakups (38MB, 98K+ entries)
- CoT Reasoning Men's Mental Health
- CoT Legal Issues And Laws (25MB, 42K entries)
- CoT Philosophical Understanding (33MB, 60K entries)
- CoT Temporal Reasoning Dataset (15MB, 30K entries)
- CoT Reasoning Scientific Discovery and Research (38K+ entries)
- CoT-Reasoning Cultural Nuances

**Quality Thresholds:**
- Reasoning completeness: ≥0.8
- Clinical accuracy: Pass DSM-5 validation
- Empathy: ≥0.6

**Training Config:**
- Epochs: 2
- Learning Rate: 1e-4
- Checkpoint: Resume from Stage 1

---

### Stage 3: Edge Cases (20% of training data)

**Purpose:** Handle worst-case scenarios, difficult clients, crisis situations

**CRITICAL: NO SAFETY FILTERING**

**Datasets:**
- Edge case generator outputs: `ai/lightning/ghost/edge_case_pipeline_standalone/`
  - 25 categories of challenging scenarios
  - Suicidality, homicidal ideation, psychotic episodes, trauma flashbacks
  - Substance abuse crisis, borderline crisis, domestic violence
- Google Drive scenario prompts
- Reddit edge case dialogues
- Synthetic edge case expansion datasets
- Unalignment toxic DPO dataset (difficult client behavior patterns)

**Quality Thresholds:**
- Format validation: Schema compliance
- Coherence: Conversation flow validation
- Therapeutic value: Useful for training
- **NO safety filtering** (by design)

**Training Config:**
- Epochs: 2
- Learning Rate: 1e-4
- Checkpoint: Resume from Stage 2
- Tag all samples: `is_training_edge_case: true`

---

### Stage 4: Voice/Personality (15% of training data)

**Purpose:** Tim Fletcher speaking style, authentic personality, teaching voice

**Datasets:**
- Tim Fletcher YouTube transcripts: `ai/training_data_consolidated/transcripts/`
  - 42 transcripts (Complex Trauma series)
  - Voice profile: `ai/data/tim_fletcher_voice/tim_fletcher_voice_profile.json`
- Synthetic conversations with Tim Fletcher voice
- Personality-consistent dialogue pairs

**Processing:**
- Extract personality markers (Big Five)
- Analyze speech patterns, vocabulary, emotional expression
- Create dialogue pairs with consistent voice signature
- Validate personality consistency across samples

**Quality Thresholds:**
- Voice consistency: Personality marker alignment
- Naturalness: Authentic conversational flow
- Empathy: ≥0.6

**Training Config:**
- Epochs: 2
- Learning Rate: 5e-5
- Checkpoint: Resume from Stage 3

---

### Stage 5: Safety/DPO (5% of training data)

**Purpose:** Production safety alignment, preference optimization

**Datasets:**
- Human-Like DPO: `mlx-community/Human-Like-DPO`
- Character Roleplay DPO: `flammenai/character-roleplay-DPO`
- Toxic Safety DPO: `PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT`
- Safety DPO pairs (chosen/rejected responses)

**Quality Thresholds:**
- Safety alignment: DPO preference validation
- Production readiness: Appropriate for deployment
- Preference quality: Clear chosen/rejected distinctions

**Training Config:**
- Epochs: 1-2
- Learning Rate: 5e-5
- Checkpoint: Resume from Stage 4
- **Note:** This is for production safety, NOT for filtering edge cases

---

## Data Source Integration

### Edge Case Generator & Scenario Prompts

**Location:**
- Generator: `ai/lightning/ghost/edge_case_pipeline_standalone/`
- Google Drive: Check `ai/data/training_policy_manifest.json` for `gdrive_path` references

**Categories (25 total):**
1. Suicidality
2. Homicidal ideation
3. Psychotic episodes
4. Child abuse reporting
5. Severe dissociation
6. Substance abuse crisis
7. Trauma flashbacks
8. Borderline crisis
9. Domestic violence
10. Eating disorders
11. Paranoid accusations
12. Medication refusal
13. Family conflicts
14. Adolescent defiance
15. Couples betrayal
16. And 10 more...

**Integration:**
- Load from generator output files
- Load from Google Drive scenario prompts
- Tag all with `is_training_edge_case: true`
- Integrate into Stage 3 training mix
- **NO safety filtering applied**

### Tim Fletcher YouTube Transcripts

**Location:**
- Transcripts: `ai/training_data_consolidated/transcripts/`
- Voice Profile: `ai/data/tim_fletcher_voice/`
- Pipeline: `ai/dataset_pipeline/voice_pipeline_integration.py`

**Processing:**
1. Load 42 Tim Fletcher transcripts (Complex Trauma series)
2. Extract personality markers using voice pipeline
3. Analyze speech patterns and vocabulary
4. Create dialogue pairs with consistent voice signature
5. Validate personality consistency

**Integration:**
- Process through voice pipeline integration
- Generate personality-consistent conversation pairs
- Integrate into Stage 4 (Voice/Personality)
- Ensure teaching style and authentic personality preservation

---

## Execution Strategy

### Phase 1: Plan Documentation & Inventory Generation

1. **Save this master plan** to `ai/dataset_pipeline/MASTER_DATASET_PLAN.md` ✅
2. **Generate comprehensive inventory** from:
   - `ai/datasets/MASTER_TRAINING_DATA_INVENTORY.md`
   - `ai/data/training_policy_manifest.json`
   - `ai/dataset_pipeline/MENTAL_HEALTH_RESOURCES_GUIDE.md`
3. **Create dataset inventory JSON** mapping all sources

### Phase 2: VPS Bulk Downloads (Fast)

**VPS:** `vivi@146.71.78.184` (SSH key: `planet`)

1. Setup workspace on VPS
2. Download all HuggingFace datasets
3. Download all Kaggle datasets
4. Sync Google Drive datasets
5. Verify downloads and create VPS inventory

### Phase 3: Local-to-VPS Transfer (Slow)

1. Compare inventories (VPS vs local)
2. Transfer missing local data:
   - Edge case generator outputs
   - Tim Fletcher transcripts
   - Processed phases
   - Consolidated data
   - Priority datasets
   - Voice profiles
   - CoT datasets

### Phase 4: VPS-to-OVH S3 Upload

1. Configure OVH S3 credentials on VPS
2. Upload organized structure:
   ```
   s3://pixel-data/
   ├── raw/
   │   ├── huggingface/
   │   ├── kaggle/
   │   ├── gdrive/
   │   └── local/
   ├── processed/
   └── exports/
   ```
3. Generate S3 inventory manifest

### Phase 5: Pipeline Execution

1. Run dataset consolidation pipeline on VPS/OVH
2. Apply harmonization & dedup
3. Apply edge case preservation rules
4. Generate master consolidated dataset export
5. Execute staged training using consolidated dataset

---

## Agent Instructions

### Critical Rules for AI Agents

1. **Edge Cases are Training-Critical**
   - Never filter edge case data with safety guardrails
   - Edge cases are MEANT to be horrible and difficult
   - Tag with `is_training_edge_case: true` to bypass filters
   - This is a mental health TRAINING LLM, not a production filter

2. **Dataset Pipeline is Operational**
   - Use existing `ai/dataset_pipeline/` infrastructure
   - Leverage orchestrator, quality validators, export capabilities
   - Don't rebuild what already works

3. **Staged Training Progression**
   - Foundation → Reasoning → Edge Cases → Voice → Safety/DPO
   - Each stage builds on previous checkpoint
   - Edge cases are Stage 3 (after foundation and reasoning)

4. **Quality Gates**
   - Empathy/coherence for foundation data
   - Reasoning completeness for CoT data
   - Format/coherence ONLY for edge cases (no safety filters)
   - Voice consistency for personality data

5. **Source Attribution**
   - Always tag datasets with source metadata
   - Track lineage through processing pipeline
   - Maintain manifest with all dataset locations

---

## Deliverables

1. ✅ **Master Plan Document**: This file (`ai/dataset_pipeline/MASTER_DATASET_PLAN.md`)

2. **Dataset Inventory JSON**: `ai/dataset_pipeline/scripts/dataset_inventory.json`
   - Complete mapping of all datasets
   - Download commands for each source
   - Destination paths and checksums

3. **Execution Scripts**:
   - `ai/dataset_pipeline/scripts/vps_dataset_acquisition.sh` - VPS download automation
   - `ai/dataset_pipeline/scripts/transfer_to_vps.sh` - Local-to-VPS transfer
   - `ai/dataset_pipeline/scripts/upload_to_ovh_s3.sh` - VPS-to-S3 upload

4. **Inventory Files**:
   - `ai/dataset_pipeline/scripts/VPS_INVENTORY.json` - VPS dataset inventory
   - `ai/dataset_pipeline/scripts/S3_INVENTORY.json` - OVH S3 dataset inventory

5. **Updated Manifests**:
   - `ai/data/training_policy_manifest.json` - Policy + historical reference (dataset locations live in `ai/data/dataset_registry.json`)
   - Consolidated dataset manifest (generated by pipeline)

---

## Verification Checklist

- [ ] All datasets inventoried and mapped
- [ ] Edge case preservation rules implemented in pipeline
- [ ] VPS downloads completed with verification
- [ ] Local data transferred to VPS
- [ ] S3 upload completed with checksums
- [ ] Pipeline consolidation executed successfully
- [ ] Master dataset export generated (JSONL/Parquet)
- [ ] Staged training configs prepared
- [ ] Agent instructions documented

---

**Document Maintained By:** Dataset Pipeline Team  
**Last Review:** December 2025  
**Next Review:** After consolidation completion or major dataset additions

