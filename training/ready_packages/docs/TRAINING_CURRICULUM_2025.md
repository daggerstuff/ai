# Training Curriculum - Late 2025

**Last Updated**: 2025-12-12  
**Purpose**: Multi-phase training curriculum for final model training  
**Training Flow**: Continued Pretraining → SFT Curriculum → Preference Alignment

---

## Overview

This curriculum implements a **3-phase training approach** optimized for late 2025 best practices:

1. **Phase A: Continued Pretraining** - Domain-adaptive text pretraining
2. **Phase B: Multi-Stage SFT Curriculum** - Supervised fine-tuning with curriculum learning
3. **Phase C: Preference Alignment** - DPO/ORPO/SimPO/KTO preference optimization

---

## Phase A: Continued Pretraining

### Purpose
Domain-adaptive pretraining on large-scale mental health and therapeutic text to build foundational knowledge.

### Datasets
- **Video Transcripts**: Tim Fletcher, Doc Snipes, Heidi Priebe, Crappy Childhood Fairy transcripts
- **General Mental Health Text**: Psychology knowledge base, educational content
- **Filtering**: Aggressive PII stripping, privacy-focused

### Training Parameters
- **Base Model**: LatitudeGames/Wayfarer-2-12B (or current best base)
- **Objective**: Causal language modeling (next-token prediction)
- **Learning Rate**: 1e-4 (lower than SFT)
- **Epochs**: 1-2 (until validation loss plateaus)
- **Context Length**: 8192 tokens
- **Batch Size**: Adaptive based on GPU memory

### Output
- **Checkpoint**: `s3://pixelated-checkpoints/foundation/pretrained/`

---

## Phase B: Multi-Stage SFT Curriculum

### Stage 1: Foundation (Therapeutic Dialogue Patterns)
**Purpose**: Learn natural therapeutic dialogue patterns

**Datasets**:
- Professional therapeutic conversations (therapist_sft, psych8k, counsel_chat)
- Priority datasets (priority_1, priority_2, priority_3)
- Mental health counseling conversations

**Mixing Weights**:
- Professional therapeutic: 40%
- Priority datasets: 40%
- Mental health counseling: 20%

**Training Parameters**:
- **Epochs**: 2-3
- **Learning Rate**: 3e-4
- **Context Length**: 2048-4096 tokens
- **Focus**: Natural conversation flow, empathy, active listening

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage1_foundation/`

---

### Stage 2: Therapeutic Expertise (Clinical Reasoning)
**Purpose**: Learn Chain of Thought clinical reasoning patterns

**Datasets**:
- CoT Clinical Diagnosis Mental Health
- CoT Heartbreak and Breakups
- CoT Neurodivergent vs Neurotypical
- CoT Mens Mental Health
- CoT Cultural Nuances
- CoT Philosophical Understanding
- CoT Temporal Reasoning

**Mixing Weights**:
- Equal weighting across all CoT datasets

**Training Parameters**:
- **Epochs**: 1-2
- **Learning Rate**: 3e-4
- **Context Length**: 4096 tokens
- **Focus**: Clinical reasoning, step-by-step thinking, diagnostic patterns

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage2_therapeutic_expertise/`

---

### Stage 3: Edge Stress Test (Crisis Robustness)
**Purpose**: Robust handling of crisis scenarios and edge cases

**Datasets**:
- Edge case generator outputs
- Edge case resulting chats
- Edge case synthetic
- Reddit mental health archives (unrestricted content)
- Crisis detection conversations

**Mixing Weights**:
- Edge case generator: 30%
- Edge case resulting chats: 30%
- Edge case synthetic: 20%
- Reddit/crisis: 20%

**Training Parameters**:
- **Epochs**: 1
- **Learning Rate**: 2e-4 (slightly lower for stability)
- **Context Length**: 4096 tokens
- **Focus**: Crisis intervention, edge case handling, unrestricted content robustness
- **Safety Note**: PII stripping enforced, but content intensity retained

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage3_edge_stress_test/`

---

### Stage 4: Voice Persona (Teaching Style)
**Purpose**: Learn Tim Fletcher teaching style and personality

**Datasets**:
- Tim Fletcher video transcripts
- Tim Fletcher voice profile
- Pixel Voice pipeline exports
- Synthetic voice conversations

**Mixing Weights**:
- Tim Fletcher transcripts: 50%
- Pixel Voice exports: 30%
- Synthetic voice: 20%

**Training Parameters**:
- **Epochs**: 1-2
- **Learning Rate**: 3e-4
- **Context Length**: 4096-8192 tokens (longer for teaching style)
- **Focus**: Teaching style, complex trauma expertise, voice consistency

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage4_voice_persona/`

---

### Stage 5: Long-Running Therapy (Session Continuity)
**Purpose**: Handle long-running therapy sessions with context continuity

**Datasets**:
- Long-running therapy sessions (extracted from professional datasets)
- Multi-session conversations
- Extended therapeutic relationships

**Mixing Weights**:
- Long-running sessions: 100% (specialized stage)

**Training Parameters**:
- **Epochs**: 1-2
- **Learning Rate**: 2e-4
- **Context Length**: 8192 tokens (maximum for long sessions)
- **Focus**: Session continuity, relationship building, progress tracking
- **Length-Aware Packing**: Preserve full sessions, don't truncate

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage5_long_running/`

---

### Stage 6: Specialized Domains (CPTSD, Addiction, Sarcasm)
**Purpose**: Specialized knowledge for niche mental health criteria

**Datasets**:
- CPTSD-specific datasets (Tim Fletcher transcripts tagged)
- Addiction counseling datasets
- Sarcasm detection examples
- Niche mental health criteria

**Mixing Weights**:
- CPTSD: 40%
- Addiction: 30%
- Sarcasm: 20%
- Other niche: 10%

**Training Parameters**:
- **Epochs**: 1
- **Learning Rate**: 2e-4
- **Context Length**: 4096 tokens
- **Focus**: Specialized knowledge, nuanced understanding

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage6_specialized/`

---

### Stage 7: Roleplay & Simulator (Training Design)
**Purpose**: Roleplay scenarios and training simulator capabilities

**Datasets**:
- Roleplay datasets
- Training simulator designer outputs
- Scenario-based training

**Mixing Weights**:
- Roleplay: 60%
- Simulator: 40%

**Training Parameters**:
- **Epochs**: 1
- **Learning Rate**: 2e-4
- **Context Length**: 4096 tokens
- **Focus**: Roleplay consistency, scenario handling

**Output Checkpoint**: `s3://pixelated-checkpoints/foundation/stage7_roleplay/`

---

## Optional Stage 0: Generic Reasoning Warmup (Nemotron-Derived)

> **Status**: Optional extension – enabled only when Nemotron-derived datasets
> have been ingested into S3 under `external/nemotron/`.

**Purpose**: Light, non-clinical preconditioning on high-quality reasoning and
instruction-following data (for example, Nemotron 3 open datasets) before
entering the core therapeutic curriculum.

**Datasets (Examples)**:
- Generic instruction-following / structured-output corpora derived from
  Nemotron HF datasets (mirrored via
  `scripts/ingest_nemotron_datasets.py` into `external/nemotron/...`).
- Reasoning-focused QA / MCQA sets that do **not** contain PHI or real
  therapeutic conversations.

**Mixing Weights**:
- Nemotron-derived generic reasoning/instruction: 100% for this stage
  (kept separate from clinical data).

**Training Parameters**:
- **Epochs**: 0.5–1 (short warmup only)
- **Learning Rate**: 3e-4 (matched to early SFT stages)
- **Context Length**: 2048–4096 tokens
- **Focus**: Tool-agnostic reasoning, instruction adherence, JSON/structured
  output reliability.

**Output Checkpoint**:
- Optional: `s3://pixelated-checkpoints/foundation/stage0_generic_reasoning/`

**Safety & Separation**:
- Nemotron-derived datasets must remain clearly labeled and stored under
  `external/nemotron/` in S3.
- These datasets are **never** mixed with PHI or real therapeutic logs.
- This stage can be disabled entirely without impacting the main curriculum.

---

## Phase C: Preference Alignment

### Purpose
Optimize model outputs using preference data (DPO/ORPO/SimPO/KTO)

### Algorithm Selection (Late 2025 Best Practices)

**Recommended**: **ORPO (Odds Ratio Preference Optimization)** or **SimPO (Simple Preference Optimization)**

**Rationale**:
- ORPO: Combines SFT + preference optimization in one stage (more efficient)
- SimPO: Simpler than DPO, better performance, no reference model needed
- DPO: Still valid but requires reference model
- KTO: Good for binary preferences

### Preference Data Sources
1. **Roleplay/Simulator Outputs**: Generate preference pairs from roleplay scenarios
2. **Adversarial Prompts**: Use edge case generator to create challenging prompts
3. **Human Feedback**: If available, use human preference annotations
4. **Synthetic Preferences**: Generate preferences using reward model or LLM-as-judge

### Training Parameters
- **Algorithm**: ORPO or SimPO
- **Learning Rate**: 1e-5 to 5e-5 (lower than SFT)
- **Epochs**: 1-2
- **Beta**: 0.1-0.5 (preference strength)
- **Context Length**: 4096 tokens

### Output
- **Final Model**: `s3://pixelated-checkpoints/final_model/`

---

## Training Order & Dependencies

```
Phase A: Continued Pretraining
    ↓
Phase B.1: Foundation
    ↓
Phase B.2: Therapeutic Expertise
    ↓
Phase B.3: Edge Stress Test
    ↓
Phase B.4: Voice Persona
    ↓
Phase B.5: Long-Running Therapy
    ↓
Phase B.6: Specialized Domains
    ↓
Phase B.7: Roleplay & Simulator
    ↓
Phase C: Preference Alignment
    ↓
Final Model
```

**Note**: Each stage uses the checkpoint from the previous stage as its base.

---

## Dataset-to-Phase Routing

### Phase A (Pretraining)
- `video_transcripts` → 100%
- `mental_health_datasets` (text-only) → 100%

### Phase B.1 (Foundation)
- `mental_health_datasets` (conversations) → 40%
- `priority_datasets` → 40%
- `professional_therapeutic` → 20%

### Phase B.2 (Therapeutic Expertise)
- `cot_reasoning` → 100%

### Phase B.3 (Edge Stress Test)
- `edge_case_generator` → 30%
- `edge_case_resulting_chats` → 30%
- `edge_case_synthetic` → 20%
- `safety_guardrails_annihilator` → 20%

### Phase B.4 (Voice Persona)
- `voice_persona` → 100%

### Phase B.5 (Long-Running Therapy)
- `long_running_therapy` → 100%

### Phase B.6 (Specialized Domains)
- `cptsd` → 40%
- `addiction` → 30%
- `sarcasm` → 20%
- `experimental` → 10%

### Phase B.7 (Roleplay & Simulator)
- `roleplay_simulator` → 100%

### Phase C (Preference Alignment)
- `dpo_preference` → 50%
- `roleplay_simulator` (preference pairs) → 30%
- `edge_case_generator` (adversarial pairs) → 20%

---

## Training Configuration Files

See:
- `ai/training_ready/configs/enhanced_training_config.json` - Base configuration
- `ai/training_ready/configs/moe_training_config.json` - MoE-specific config
- `ai/training_ready/data/dataset_routing_config.json` - Dataset routing

---

## Validation & Evaluation

### Per-Stage Validation
- **Validation Set**: 5% holdout from each stage's training data
- **Metrics**: Loss, perplexity, therapeutic quality scores
- **Early Stopping**: Patience of 3 epochs

### Final Evaluation
- **Test Set**: Hard holdout families (long_running_therapy, edge_case_crisis, sarcasm, voice_persona)
- **Metrics**:
  - Therapeutic quality
  - Crisis handling
  - Voice consistency
  - Sarcasm detection
  - Long-session coherence

---

## Related Documentation

- **Final Dataset Contract**: `ai/training_ready/docs/FINAL_DATASET_CONTRACT.md`
- **S3 Structure**: `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md`
- **Coverage Report**: `ai/training_ready/data/dataset_coverage_report.json`
