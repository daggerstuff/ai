# HetznerS3 Dataset Inventory

Pulled from `HetznerS3:pixeldata/` (136GB) + `HetznerS3:pixeldata-cleaned/` (20GB).
Local cache: `ai/data/raw/hetzner/`

## Production Training Pipeline

### final_dataset/ (21.8GB)
| File | Records | Size |
|------|---------|------|
| MASTER_TRAINING_SET_PREV.jsonl | 7,244,028 | 20GB |
| final_training_dataset.jsonl | 53,751 | 31MB |
| v5_shards/ (7 files) | — | — |
| regenerated_gestalt/ | — | — |

### compiled_dataset/ (2.4GB)
| Split | Shards | Records/shard | Total |
|-------|--------|---------------|-------|
| train | 138 | 5,000 | 690,000 |
| test | 7 | 5,000 | 35,000 |
| val | 7 | 5,000 | 35,000 |

### staged_datasets/ (22GB)
| Stage | Records | Size |
|-------|---------|------|
| stage1_foundation.jsonl | 7,215,772 | 20GB |
| stage2_therapeutic_expertise.jsonl | 35,400 | 284MB |
| stage3_edge_stress_test.jsonl | 29,939 | 75MB |
| stage4_voice_persona.jsonl | 577 | 34MB |
| supplementary.jsonl | 96,953 | 346MB |

### datasets/consolidated/ (8.8GB)
| File | Records | Size |
|------|---------|------|
| ULTIMATE_FINAL_DATASET.jsonl | 608,497 | 2.5GB |
| merged_dataset.jsonl | 608,458 | 2.5GB |
| training_dataset_enhanced.json | 1,087,541 | 58MB |
| raw/priority/priority_1.jsonl | 102,594 | 463MB |
| raw/priority/priority_2.jsonl | 84,143 | 330MB |
| raw/priority/priority_3.jsonl | 111,180 | 371MB |
| datasets/priority_1_FINAL.jsonl | 102,594 | 479MB |
| datasets/priority_2_FINAL.jsonl | 84,143 | 234MB |
| datasets/priority_3_FINAL.jsonl | 110,971 | 388MB |
| datasets/psychology_dataset.json | 49,231 | 5MB |
| psychology_knowledge/psychology_knowledge_base_optimized.json | 239,296 | 19MB |
| psychology_knowledge/enhanced_psychology_knowledge_base.json | 42,170 | 1.5MB |
| conversations/training_data.jsonl | 4,356 | 19MB |
| conversations/edge_case_dialogues.jsonl | 20 | 136KB |

### datasets/training_v2/
- stage1_foundation/ — Amod counseling (500), Heliosbrahma chatbot (172)
- stage2_specialist_addiction/ — fadodr therapy (500)
- stage3_edge_crisis/ — nightmare scenarios (4 batch files)
- stage4_voice/ — voice transcripts (1,241)
- final_instruct/ — train (136), val (8), test (8), dpo (155)

### datasets/training_v3/
- stage1_foundation/ — same as v2
- stage2_specialist_personality/ — Kanakmi disorders (500)
- stage2_specialist_addiction/ — fadodr (500)
- stage4_voice_persona/ — 4 persona datasets (500 each)

### Other Directories
| Dir | Records | Notes |
|-----|---------|-------|
| cot_reasoning/ | 40,936 + 35,585 + 3,512 | Alexander_Street shareGPT, train, combined |
| processed/ | 37,625 | tier1 dark humor curated |
| acquired/ | 66,729 + 5,401 | mental_health_counseling, cot_reasoning |
| clinical/pubmed/ | 1 file | PubMed clinical data |
| ai/training_ready/ | 608,497 | ULTIMATE_FINAL_DATASET copy |
| ai/data/compress/ | — | mental_health_clean.jsonl (249MB) |
| youtube_transcripts/ | 42 | Tim Fletcher transcripts |
| nemo_synthetic/ | 600 | synthetic dialogues |

## pixeldata-cleaned/ (20GB, 192 files)
- archive_judged/ — 180+ judged JSONL files from voice exports, therapy transcripts, compiled shards
- judged_and_cleaned/ — final cleaned datasets
- processing_reports/ — quality reports

### Voice Export Sources
- crappy_childhood_fairy
- doc_snipes
- heidi_priebe
- patrick_teahan
- therapy_in_a_nutshell
- tim_fletcher

## Total Records (HetznerS3)
- Master training set: 7.2M records
- Compiled shards: 760K records
- Staged pipeline: 7.4M records (stage1) + 162K (stages 2-4 + supp)
- Consolidated: 608K ultimate + 298K priority + 281K psych knowledge
- Acquired/processed: 109K + 37K
- CoT reasoning: 80K
- **Grand total: ~17M+ records**
