# Master Training Plan – Pixelated Empathy

> Cohesive blueprint for consolidating all therapeutic datasets into a four-stage training ladder while preserving nightmare-fuel edge cases, scenario prompts, and Tim Fletcher voice alignment.

---

## 1. Context & Alignment Checklist

1. Review `ai/datasets/MASTER_TRAINING_DATA_INVENTORY.md` for sizes, formats, and pending pulls (Kaggle TF-IDF pack, MEMO access).
2. Cross-reference existing guides:
   - `ai/dataset_pipeline/MENTAL_HEALTH_RESOURCES_GUIDE.md`
   - `ai/dataset_pipeline/architecture/dataset_training_architecture.md`
   - `ai/dataset_pipeline/INTEGRATED_PIPELINE_GUIDE.md`
3. Surface blockers in status doc / stand-up notes:
   - MEMO dataset requires manual approval.
   - Kaggle TF-IDF bundle must be downloaded on the remote GPU node before Stage 3 can be balanced.
   - Voice pipeline still generating `tim_fletcher_pipeline` artifacts; mark in manifest as `in_progress`.

---

## 2. Four-Stage Training Ladder

| Stage | Target Share | Purpose | Primary Sources | Inclusion / Exclusion Rules |
|-------|--------------|---------|-----------------|-----------------------------|
| **Stage 1 – Foundation & Rapport** (`stage1_foundation`) | 40% | Establish baseline therapeutic tone, reflective listening, and low-risk support dialogs. | Processed Phase 1 & 2 datasets (`ai/lightning/pixelated-training/processed`), Wendy priority JSONL (`ai/lightning/ghost/datasets/priority_*`), pro datasets (`therapist_sft`, `SoulChat`, `counsel_chat`, `Psych8k`, `mental_health_counseling`, `neuro_qa_sft`). | Require empathy score ≥ 0.55, no structural corruption, dedup preference: consolidated `priority_*` overrides raw HF pulls. |
| **Stage 2 – Therapeutic Expertise & Reasoning** (`stage2_therapeutic_expertise`) | 25% | Teach structured reasoning, diagnosis scaffolding, knowledge grounding, MEMO summarization style. | CoT corpora in `ai/dataset_pipeline/cot_datasets/`, reasoning JSON under `ai/training_data_consolidated/psychology_knowledge`, DPO preference samples, MEMO (when licensed), professional PDF/book extracts. | Must have reasoning metadata (`chain_of_thought`, `summary`, `technique`). Empathy floor 0.5 but allow sterile clinical tone. Dedup keeps highest-quality reasoning path. |
| **Stage 3 – Edge Stress Test & Scenario Bank** (`stage3_edge_stress_test`) | 20% | Force model to survive nightmare fuel: crisis, trauma, Reddit rawness, edge generator prompts. | `ai/pipelines/edge_case_pipeline_standalone/output/edge_cases_training_format.jsonl`, scenario prompt corpus (Google Drive exports mirrored to `ai/dataset_pipeline/prompt_corpus/`), Reddit/Kaggle TF-IDF dumps, Synthetic edge expansions, Safety DPO violations for contrastive pairs. | **Do NOT** sanitize horror details. Only drop malformed JSON or PII (phone, SSN). Flag metadata `{"crisis_intensity": "very_high"}` for gating. Allow safety validator to mark `requires_human_review` without removal. |
| **Stage 4 – Voice, Persona & Delivery** (`stage4_voice_persona`) | 15% | Maintain Tim Fletcher tone, cadence, empathy micro-movements, and dual persona scripts. | YouTube transcripts (`ai/training_data_consolidated/transcripts/`), `ai/data/tim_fletcher_voice/`, Pixel Voice pipeline outputs, dual-persona datasets, speaker-style embeddings. | Require `voice_signature` tokens + `persona_id`. Dedup merges transcripts + synthesized persona expansions by `source_session_id`. Guard voice instructions from safety abridgement. |

### Metadata Schema (applies to all stages)

Use `ConversationRecord` from `ai/dataset_pipeline/schemas/conversation_schema.py`:
```json
{
  "conversation_id": "uuid",
  "stage": "stage3_edge_stress_test",
  "messages": [...],
  "metadata": {
    "source": "edge_case_generation",
    "quality_scores": {
      "empathy": 0.44,
      "safety": 0.61,
      "bias": 0.08
    },
    "crisis_intensity": "very_high",
    "voice_signature": "tim_fletcher_v2",
    "persona_id": "dual:therapist_mentor"
  }
}
```

---

## 3. Deduplication, Cleaning, and Guard-Rail Policy

### Hash Keys & Conflict Resolution

- **Primary hash**: `sha256(lowercase(concat(messages.role + messages.content)))`.
- **Secondary hash**: `sha1(conversation_id + stage + source + crisis_intensity)`.
- Use `ai/dataset_pipeline/ingestion_deduplication.py` (or `processing/deduplication_*`) to generate these hashes and resolve collisions.
- Conflict order: `stage4_voice_persona` > `stage3_edge_stress_test` > `stage2_therapeutic_expertise` > `stage1_foundation` > supplementary. Later stages win to preserve specialized tone.

### Cleaning Passes

1. **Normalization**: Unicode normalize, strip zero-width chars, convert smart quotes.
2. **Metadata enrichment**: attach `stage`, `source`, `difficulty_level`, `voice_signature`.
3. **Profanity tagging** (not removal): annotate `metadata.flags.contains_extreme_language = true`.
4. **PII/Compliance**: run PII detector; mask or drop only when real identifiers appear. Crisis gore stays.
5. **Structural QA**: ensure alternating roles, timestamps optional.

### Validators per Stage

| Stage | Validators | Threshold / Mode |
|-------|------------|------------------|
| 1 | `EmpathyMentalHealthValidator` | Fail if empathy < 0.55; reroute to Stage 3 bucket only if scenario qualifies as crisis. |
| 2 | Empathy (≥0.5) + Bias monitor + `evaluation_gates.reasoning_score >= 0.65`. |
| 3 | Safety validator runs in *lenient* mode (`safety_threshold=0.6`, `allow_crisis_override=True`), bias monitor flagged but not dropping. |
| 4 | Empathy ≥0.6, safety ≥0.75, voice-style discriminator (Pixel Voice pipeline). |

---

## 4. Multi-Stage Training Layout & Sampler

### Integrated Pipeline Extension

Modify `ai/dataset_pipeline/orchestration/integrated_training_pipeline.py`:

1. **Stage manifests**: load from `ai/data/master_dataset_manifest.json` where each entry now includes `stage`, `target_percentage`, `quality_profile`.
2. **Sampler**: introduce `StageSampler` helper that:
   - reads candidate files per stage,
   - enforces ratio using reservoir sampling,
   - logs under/overflow in `integration_report.stage_balance`.
3. **Output artifacts**:
   - `ai/training_data_consolidated/final/MASTER_STAGE_{n}.jsonl`
   - `training_data_consolidated/FINAL_TRAINING_DATA_MANIFEST.json` includes `stage_metrics`.
4. **Splits**: for each stage output `train/val/test = 80/10/10`, plus aggregated splits consumed by `ai/lightning/pixelated-training/training_dataset.json`.
5. **Monitoring hooks**: reuse `ai/dataset_pipeline/quality_control.py` & `evaluation_gates.py` to emit:
   ```json
   {
     "stage": "stage3_edge_stress_test",
     "samples": 42000,
     "empathy_avg": 0.41,
     "safety_avg": 0.58,
     "crisis_override": 812
   }
   ```

---

## 5. Edge Generator, Scenario Prompts, and DPO Violations

1. **Edge Generator Intake**
   - Source: `ai/pipelines/edge_case_pipeline_standalone/output/edge_cases_training_format.jsonl`.
   - Add to manifest `stage3_edge_stress_test.edge_case_generator`.
   - Metadata fields: `category`, `difficulty_level`, `expected_challenges`, `prompt_hash`.
   - Keep `purpose: "difficult_client"` to differentiate from synthetic polite data.

2. **Scenario Prompt Corpus**
   - Mirror Google Drive prompt sheets to repo under `ai/dataset_pipeline/prompt_corpus/{category}.md`.
   - Build ingestion script stub `prompt_corpus_loader.py` (future work) referencing manifest entry.
   - Prompts seed `SyntheticDataDistillationPipeline` strategies (`edge_case_expansion`, `cultural_augmentation`).

3. **Safety DPO Pairing**
   - Unsafe conversations from Stage 3 feed `create_safety_dpo_dataset` to create DPO pairs.
   - Keep pairs referenced in manifest under `stage3_edge_stress_test.safety_dpo_pairs`.

---

## 6. Voice / Persona Alignment

1. **Transcripts → Voice Tokens**
   - Run Pixel Voice pipeline (`ai/pipelines/pixel_voice/`) to convert YouTube transcripts into `conversation + voice_signature`.
   - Store derived files in `ai/data/tim_fletcher_voice/exports/`.
2. **Persona Stitching**
   - Dual persona datasets (mentor + peer) combined with Stage 1 records via `persona_blender.py` (todo script) to generate training segments with `persona_id`.
3. **Guardrail Override**
   - Safety validator runs in normal mode (≥0.75) but with `allowed_styles = ["tim_fletcher", "pixel_voice"]` so cadence instructions stay.
4. **Manifest updates**
   - `stage4_voice_persona.tim_fletcher_transcripts` (transcripts)
   - `stage4_voice_persona.voice_profile_embeddings`
   - `stage4_voice_persona.dual_persona_scripts`

---

## 7. Runbooks

### End-to-End Stage Build

```bash
cd /home/vivi/pixelated
uv run python ai/dataset_pipeline/scripts/pre_ingest_checks.py  # validates mounts, MEMO access
uv run python ai/dataset_pipeline/processing/unified_preprocessing_pipeline.py --stage all
uv run python ai/dataset_pipeline/orchestration/integrated_training_pipeline.py --staged --report reports/stage_report.json
```

### Edge Case Refresh

```bash
cd ai/pipelines/edge_case_pipeline_standalone
python generate_training_data.py --scenarios-per-category 50 --provider ollama --model llama3.2
```

### Voice Refresh

```bash
cd ai/pipelines/pixel_voice
uv run python pipeline.py --transcripts ../../training_data_consolidated/transcripts --output ../../data/tim_fletcher_voice/exports
```

### Kaggle TF-IDF Pull (Stage 3 dependency)

```bash
cd ai
kaggle kernels output rickyzou/nlp-for-mental-health-text-data -p datasets/tier4_reddit_archive/
```

---

## 8. Operational Checklists

- [ ] Inventory up to date (processed, consolidated, pending download).
- [ ] MEMO access recorded (or placeholder flagged).
- [ ] Edge generator output refreshed ≤7 days.
- [ ] Scenario prompts mirrored to `ai/dataset_pipeline/prompt_corpus/`.
- [ ] Tim Fletcher voice exports present with latest transcripts.
- [ ] Integrated pipeline report shows stage ratios within ±2%.
- [ ] Final manifest updated with `stage`, `target_percentage`, `quality_profile`, and ingestion timestamps.

---

## 9. References

- `ai/data/master_dataset_manifest.json`
- `ai/data/dataset_registry.json`
- `ai/dataset_pipeline/ingestion_deduplication.py`
- `ai/dataset_pipeline/quality/*`
- `ai/pipelines/edge_case_pipeline_standalone`
- `ai/pipelines/pixel_voice`

This document is the single source of truth for assembling the master dataset going forward. Update it whenever new stages, validators, or data sources are introduced. Guard-rail policies here supersede default safety filters for Stage 3 nightmare fuel scenarios.

