# Edge & Voice Ingestion Playbook

> How Stage 3 (Edge Stress Test) and Stage 4 (Voice / Persona) data flows into the master training pipeline.

---

## 1. Edge Case Pipeline (Stage 3)

1. **Generate / refresh edge nightmare data**
   ```bash
   cd ai/pipelines/edge_case_pipeline_standalone
   python generate_training_data.py --scenarios-per-category 50
   ```
2. **Verify output** – should appear at `ai/pipelines/edge_case_pipeline_standalone/output/edge_cases_training_format.jsonl`.
3. **Loader mapping**
   - `EdgeCaseJSONLLoader` now tags every record with `stage="stage3_edge_stress_test"`, `quality_profile="edge_crisis"`, and preserves `difficulty_level` as `crisis_intensity`.
   - Guard rails are **lenient**: only malformed JSON or PII (phone, email, SSN) is stripped during preprocessing.
4. **Kaggle TF-IDF bundle**
   ```bash
   cd ai
   kaggle kernels output rickyzou/nlp-for-mental-health-text-data -p datasets/tier4_reddit_archive/
   ```
   Place exports inside `ai/datasets/tier4_reddit_archive/` so manifest entry `kaggle_tf_idf_bundle` flips from `pending_download` → `ready`.
5. **Scenario prompt corpus**
   - Mirror Google Drive sheets to `ai/dataset_pipeline/prompt_corpus/{category}.md`.
   - Use prompts to seed `SyntheticDataDistillationPipeline(strategy="edge_case_expansion")`.
6. **Safety DPO pairs**
   ```python
   from ai.dataset_pipeline.quality.safety_alignment_validator import create_safety_dpo_dataset
   unsafe_pairs = create_safety_dpo_dataset(edge_conversations, validator)
   ```
   Save to `ai/dataset_pipeline/quality/output/safety_dpo_pairs.jsonl` (manifest entry `safety_dpo_pairs`).

## 2. Voice / Persona Pipeline (Stage 4)

1. **Sync transcripts + voice profile**
   - Transcripts reside under `ai/training_data_consolidated/transcripts/`.
   - Voice embeddings + prompt templates at `ai/data/tim_fletcher_voice/`.
2. **Run Pixel Voice pipeline**
   ```bash
   cd ai/pipelines/pixel_voice
   uv run python pipeline.py \
     --transcripts ../../training_data_consolidated/transcripts \
     --output ../../data/tim_fletcher_voice/exports
   ```
3. **Loader mapping**
   - `PixelVoiceLoader` writes `stage="stage4_voice_persona"` and `voice_signature` extracted from `personality_markers`.
   - Records without explicit signature fall back to `tim_fletcher_voice_profile`.
4. **Dual persona stitching**
   - `ai/pipelines/dual_persona_training` output should include `persona_id` metadata before entering Stage 4 bucket.

## 3. Stage-Balanced Export Verification

1. Run the integrated pipeline:
   ```bash
   cd ai/dataset_pipeline/orchestration
   python integrated_training_pipeline.py
   ```
2. Inspect the stage manifest:
   ```python
   import json
   with open('ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json') as f:
       print(json.load(f)['stages'])
   ```
3. Expected files:
   - `MASTER_stage1_foundation.jsonl`
   - `MASTER_stage2_therapeutic_expertise.jsonl`
   - `MASTER_stage3_edge_stress_test.jsonl`
   - `MASTER_stage4_voice_persona.jsonl`

## 4. Guard-Rail Policy Reminder

- **Stage 3**: Nightmare content is intentional. Never trim aggression or crisis language unless it contains real-world identifiers.
- **Stage 4**: Safety floor remains 0.75, but do **not** remove cadence / quirky phrasing from Tim Fletcher style prompts.
- Deduplication prefers Stage 4 > Stage 3 > Stage 2 > Stage 1 to keep persona + crisis samples intact when overlaps occur.

Keep this document updated whenever new edge categories, prompt corpora, or persona voices are introduced.

