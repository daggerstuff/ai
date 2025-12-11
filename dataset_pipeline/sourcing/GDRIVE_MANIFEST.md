# GDrive Training Data Manifest for S3 Upload

Generated: 2025-12-08

## Summary
- **Total Estimated Size:** ~12GB+
- **Priority:** High-value processed & curated datasets

---

## Tier 1: Processed Phase Data (9.5GB)
High-quality, pre-processed training data organized by phase.

| File | Size | Tier |
|------|------|------|
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_data.jsonl` | 2.8GB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl` | 1.9GB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_32_control_groups/control_group_conversations.jsonl` | 1.2GB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl` | 1.2GB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl` | 910MB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_33_tfidf_integration/ml_ready_conversations.jsonl` | 905MB | tier2_professional |
| `/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl` | 637MB | tier3_edge_crisis |
| `/mnt/gdrive/processed/phase_3_cot_reasoning/phase_3_cot_reasoning_consolidated.jsonl` | 318MB | tier3_cot_reasoning |
| `/mnt/gdrive/processed/phase_3_cot_reasoning/task_5_15_cot_reasoning/cot_reasoning_conversations_consolidated.jsonl` | 280MB | tier3_cot_reasoning |
| `/mnt/gdrive/processed/phase_3_cot_reasoning/task_5_26_pattern_recognition/therapeutic_reasoning_pattern_database.json` | 241MB | tier3_cot_reasoning |
| `/mnt/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl` | 153MB | tier1_priority |
| `/mnt/gdrive/processed/phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl` | 143MB | tier1_priority |
| `/mnt/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl` | 123MB | tier1_priority |

---

## Tier 2: Raw Datasets (3.5GB)
Original downloaded datasets on gdrive.

| File | Size | Tier |
|------|------|------|
| `/mnt/gdrive/datasets/Reasoning_Problem_Solving_Dataset/RPSD.json` | 651MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/reddit_mental_health/mental_disorders_reddit.csv` | 562MB | tier2_professional |
| `/mnt/gdrive/datasets/data-final.csv` | 397MB | tier2_professional |
| `/mnt/gdrive/datasets/therapist-sft-format/train.csv` | 388MB | tier1_priority |
| `/mnt/gdrive/datasets/ToT_Reasoning_Problem_Solving_Dataset_V2/ToT-RPSD-V2.json` | 230MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/merged_dataset.jsonl` | 162MB | tier1_priority |
| `/mnt/gdrive/datasets/reddit_mental_health/Suicide_Detection.csv` | 160MB | tier3_edge_crisis |
| `/mnt/gdrive/datasets/merged_mental_health_dataset.jsonl` | 86MB | tier1_priority |
| `/mnt/gdrive/datasets/CoT_Rare-Diseases_And_Health-Conditions/CoT_Rare Disseases_Health Conditions_9.8k.json` | 65MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json` | 53MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/CoT-Reasoning_Cultural_Nuances/CoT-Reasoning_Cultural_Nuances_Dataset.json` | 42MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/CoT_Heartbreak_and_Breakups_downloaded.json` | 38MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health_downloaded.json` | 20MB | tier3_cot_reasoning |
| `/mnt/gdrive/datasets/CoT_Reasoning_Mens_Mental_Health_downloaded.json` | 18MB | tier3_cot_reasoning |

---

## Tier 3: YouTube Transcripts (Voice Pipeline)
Prioritized by creator for voice/persona training.

| Channel | Size | Priority |
|---------|------|----------|
| Tim Fletcher | 2.3MB | **High** |
| Patrick Teahan | 407KB | High |
| The Diary Of A CEO | 401KB | Medium |
| Therapy in a Nutshell | 168KB | High |
| Crappy Childhood Fairy | 166KB | High |
| DoctorRamani | 159KB | High |
| Mel Robbins | 153KB | Medium |
| Heidi Priebe | 152KB | High |

**Path:** `/mnt/gdrive/youtube_transcriptions/transcripts/`

---

## Upload Commands

```bash
# Priority 1: Processed phase data (largest, highest quality)
uv run python ai/dataset_pipeline/sourcing/gdrive_manifest_uploader.py --tier processed

# Priority 2: Raw datasets
uv run python ai/dataset_pipeline/sourcing/gdrive_manifest_uploader.py --tier raw

# Priority 3: YouTube transcripts
uv run python ai/dataset_pipeline/sourcing/gdrive_manifest_uploader.py --tier youtube
```

---

## Notes
- Skip `.npy` files (binary numpy, not trainable text)
- Skip duplicate files between processed and raw
- Prioritize `.jsonl` format over `.csv` when both exist
