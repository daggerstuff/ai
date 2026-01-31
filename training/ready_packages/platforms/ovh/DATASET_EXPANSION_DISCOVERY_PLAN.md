# Pixelated Empathy - Comprehensive Dataset Expansion and Discovery Plan

**Version**: 2.0  
**Generated**: 2025-01-27  
**Status**: Active Planning

---

## Executive Summary

This comprehensive plan documents the complete 6-tier dataset system for Pixelated Empathy foundation model training, including all existing datasets, missing data to acquire, deep discovery strategies, and integration with the training pipeline.

**Training Target**: ~100,000 high-quality therapeutic conversations

**Current Status**: ~6.4GB consolidated, ~1.86GB missing from GDrive

---

## Training Strategy Overview

### Training Ratio Strategy

| Category | Percentage | Focus Areas |
|----------|-----------|------------|
| **Psychology** | 30% | Psychology knowledge base, DSM-5, therapeutic techniques |
| **Voice** | 25% | YouTube transcripts, personality-consistent dialogues |
| **Mental Health** | 20% | Professional therapeutic datasets, clinical conversations |
| **Reasoning** | 15% | Chain-of-thought reasoning datasets |
| **Personality** | 10% | Personality balancing datasets, Big Five profiles |

### Quality Thresholds by Tier

| Tier | Quality Threshold | Training Weight | Description |
|------|------------------|----------------|-------------|
| **Tier 1** | 99% | 40% | Curated Priority Datasets |
| **Tier 2** | 95% | 25% | Professional Therapeutic Datasets |
| **Tier 3** | 90% | 20% | Chain-of-Thought Reasoning |
| **Tier 4** | 85% | 10% | Reddit Mental Health Archive |
| **Tier 5** | 80% | 4% | Research & Specialized Datasets |
| **Tier 6** | Reference (1.0) | 1% | Knowledge Base & Reference Materials |

---

## Part 1: Complete 6-Tier Dataset System Audit

### Tier 1: Curated Priority Datasets

**Location**: `ai/datasets/datasets-wendy/` (GDrive: `gdrive:datasets/datasets-wendy/`)  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier1_priority_loader.py`  
**Status**: ❌ **MISSING FROM REMOTE SERVER** (1.16GB total - NOT DOWNLOADED)

| Dataset | Size | Status | Description |
|---------|------|--------|-------------|
| `priority_1_FINAL.jsonl` + summary.json | 462MB | ❌ Missing | Top-tier therapeutic conversations |
| `priority_2_FINAL.jsonl` + summary.json | 330MB | ❌ Missing | High-quality mental health data |
| `priority_3_FINAL.jsonl` + summary.json | 370MB | ❌ Missing | Specialized therapeutic content |
| `priority_4_FINAL.jsonl` + summary.json | TBD | ❌ Missing | Extended training data |
| `priority_5_FINAL.jsonl` + summary.json | TBD | ❌ Missing | Supplementary datasets |

**Action Required**: Download all priority datasets from GDrive to remote server

---

### Tier 2: Professional Therapeutic Datasets

**Location**: `~/datasets/consolidated/`  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier2_professional_loader.py`  
**Status**: ✅ **510MB total on remote server**

| Dataset | Size | Status | Source | Conversations |
|---------|------|--------|--------|---------------|
| `therapist-sft-format/train.csv` | 388MB | ✅ | Structured therapist training | ~99K |
| `SoulChat2.0/PsyDTCorpus_train_mulit_turn_packing.json` | 48MB | ✅ | Advanced psychological counselor | Multi-turn |
| `SoulChat2.0/PsyDTCorpus_test_single_turn_split.json` | 28MB | ✅ | SoulChat2.0 test set | Single-turn |
| `Psych8k/Alexander_Street_shareGPT_2.0.json` | 6.3MB | ✅ | Alexander Street professional therapy | 40K+ |
| `neuro_qa_SFT_Trainer/train.json` | 5.9MB | ✅ | Neurology/psychology Q&A | 35K+ |
| `mental_health_counseling_conversations/combined_dataset.json` | 4.6MB | ✅ | Licensed therapist responses | 3.5K |
| `counsel-chat/counsel_chat_250-tokens_full.json` | 4.0MB | ✅ | Professional counseling archive | - |
| `counsel-chat/20200325_counsel_chat.csv` | 3.2MB | ✅ | Historical counseling sessions | - |
| `counsel-chat/counselchat-data.csv` | 2.9MB | ✅ | Counsel chat data | - |

**Missing from GDrive** (not yet downloaded):
- `LLAMA3_Mental_Counseling_Data` - Advanced AI counseling conversations
- Additional SoulChat2.0 variants
- Extended counsel-chat archives

---

### Tier 3: Chain-of-Thought Reasoning Datasets

**Location**: `~/datasets/consolidated/`, `ai/dataset_pipeline/cot_datasets/`  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier3_cot_loader.py`  
**Status**: ✅ **1.3GB total on remote server**

| Dataset | Size | Status | Estimated Items | Focus Area |
|---------|------|--------|----------------|------------|
| `RPSD.json` | 653MB | ✅ | ~100K+ | Reasoning problem solving |
| `ToT-RPSD-V2.json` | 230MB | ✅ | 72,117 | Tree of Thought reasoning |
| `CoT_Rare-Diseases_And_Health-Conditions_9.8k.json` | 65MB | ✅ | 9,800 | Medical psychology reasoning |
| `CoT-Reasoning_Cultural_Nuances_Dataset.json` | 42MB | ✅ | ~6K | Culturally-sensitive approaches |
| `CoT-Breakups and heartbreak-9.8k.json` | 37MB | ✅ | 9,800 | Emotional intelligence & relationships |
| `General_Inquiry_Thinking-Chain-Of-Thought_6K.json` | 37MB | ✅ | 6,000 | General inquiry reasoning |
| `CoT_Reasoning_The_Ancient_Past.json` | 30MB | ✅ | ~4K | Historical reasoning |
| `CoT_Legal_Issues_And_Laws.json` | 25MB | ✅ | 42,000 | Legal/ethical reasoning in therapy |
| `CoT_Reasoning_Scientific Discovery and Research.json` | 25MB | ✅ | 38,000+ | Evidence-based practice reasoning |
| `CoT_Medical_Diagnosis_3k.json` | 21MB | ✅ | 3,000 | Medical diagnosis reasoning |
| `CoT_Reasoning_Clinical_Diagnosis_Mental_Health.json` | 20MB | ✅ | 30,000+ | Clinical diagnostic reasoning |
| `CoT_Reasoning_First_Responders_Triage_And_Emergencies.json` | 18MB | ✅ | ~2.5K | Emergency response reasoning |
| `CoT_Reasoning_Mens_Mental_Health.json` | 18MB | ✅ | ~2.5K | Gender-specific therapeutic reasoning |
| `CoT_Reasoning_Quantum_Physics_And_Computing.json` | 17MB | ✅ | ~2.5K | Quantum physics reasoning |
| `CoT_Temporal_Reasoning_Dataset.json` | 15MB | ✅ | 30,000 | Time-based therapeutic planning |

**Missing from GDrive** (not yet downloaded):
- `CoT_Neurodivergent_vs_Neurotypical_Interactions` (53MB) - Neurodiversity-aware therapeutic approaches
- `CoT_Philosophical_Understanding` (33MB, 60K entries) - Existential/philosophical therapy

---

### Tier 4: Comprehensive Reddit Mental Health Archive

**Location**: `~/datasets/consolidated/reddit/`, `ai/datasets/old-datasets/`  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier4_reddit_loader.py`  
**Status**: ✅ **2.9GB on remote server**, ❌ **700MB+ missing**

**Currently Available**:
- Various subreddit CSVs (Anxiety, Depression, ADHD, BPD, PTSD, schizophrenia, etc.)
- Condition-specific archives (50+ datasets)
- Temporal analysis data (2018/2019 longitudinal studies)
- TF-IDF feature vectors (256 dimensions) for ML applications

**Missing from GDrive** (700MB+ - NOT DOWNLOADED):

| File | Size | Description |
|------|------|-------------|
| `mental_disorders_reddit.csv` | 561MB | Comprehensive mental disorders data |
| `Suicide_Detection.csv` | 159MB | Suicide risk detection patterns |
| `merged_mental_health_dataset.jsonl` | 85MB | Merged mental health dataset |
| Additional Reddit archives | TBD | Additional condition-specific data |

**Specialized Populations**:
- `adhdwomen.csv` - Gender-specific ADHD experiences
- `EDAnonymous datasets` - Eating disorder support communities
- `COVID19_support_post_features` - Pandemic mental health support

**Control Groups**: fitness, jokes, meditation, personalfinance (non-clinical baselines)

---

### Tier 5: Research & Specialized Datasets

**Location**: `~/datasets/consolidated/`, `ai/datasets/`  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier5_research_loader.py`  
**Status**: ✅ **1.7GB on remote server**

| Dataset | Size | Source | Format | Description |
|---------|------|--------|--------|-------------|
| `data-final.csv` | 397MB | Big Five personality profiles | CSV | 1M+ personality assessment records |
| `Anthropic_hh-rlhf/train/data-00000-of-00001.arrow` | 295MB | Anthropic helpful/harmless | Arrow | Helpful/harmless training data |
| `RECCON` (all folds combined) | ~400MB | Emotion cause extraction | JSON/CSV | Emotion-cause relationship data |
| `Anthropic_hh-rlhf/test/data-00000-of-00001.arrow` | 16MB | Anthropic test set | Arrow | Helpful/harmless test data |
| `psychology-10k/Psychology-10K.json` | 5MB | Psychology knowledge base | JSON | 10K psychology concepts |
| `Empathy-Mental-Health datasets` | 4.5MB | EMNLP 2020 empathy research | JSON | Research-validated empathy patterns |
| `DepressionDetection` | 2.7MB | Reddit/Twitter depression detection | CSV | Clinical detection patterns |
| `formatted_annotated_addiction_counseling` | 1.3MB | Addiction counseling | CSV | Annotated addiction counseling |

**Missing/Additional Sources**:
- `IEMOCAP_EMOTION_Recognition` - Audio emotion recognition pipeline
- `MODMA-Dataset` - Multi-modal mental disorder analysis
- `unalignment_toxic-dpo-v0.2-ShareGPT` - Difficult client behavior patterns
- `Original Reddit Data/raw data` - Unprocessed source data

---

### Tier 6: Knowledge Base & Reference Materials

**Location**: `ai/training_data_consolidated/`, `ai/datasets/`  
**Loader**: `ai/dataset_pipeline/ingestion/tier_loaders/tier6_knowledge_loader.py`  
**Status**: ✅ **Available locally**

| Source | Location | Format | Content |
|--------|----------|--------|---------|
| **DSM-5** | `ai/training_data_consolidated/` | PDF → Structured | Complete diagnostic criteria |
| **psychology-10k** | `ai/training_data_consolidated/` | JSON | 10K psychology concepts |
| **Psych-101** | `ai/datasets/` | Educational prompts | Psychology fundamentals |
| **xmu_psych_books** | `ai/datasets/` | Textbook corpus | Academic psychology knowledge |
| **customized-mental-health-snli2** | `ai/datasets/` | NLI format | Mental health inference data |

**Total Knowledge Base**: 4,867+ psychology concepts

---

## Part 2: Additional Training Data Sources

### Edge Case Generation Pipeline

**Location**: `ai/lightning/ghost/edge_case_pipeline_standalone/`  
**Loader**: `ai/dataset_pipeline/ingestion/edge_case_loader.py`  
**Status**: ✅ **Active**

**25 Categories, Difficulty-Tagged Scenarios**:

- **Very High Difficulty**: Suicidality, homicidal ideation, psychotic episodes, child abuse reporting, severe dissociation
- **High Difficulty**: Substance abuse crisis, trauma flashbacks, borderline crisis, domestic violence, eating disorders
- **Moderate Difficulty**: Paranoid accusations, medication refusal, family conflicts, adolescent defiance, couples betrayal

**Features**:
- Multi-provider support (OpenAI, Anthropic, Ollama)
- Safety flagging and clinical review
- Difficulty level tagging
- Expected challenges metadata

---

### Pixel Voice Pipeline

**Location**: `ai/dataset_pipeline/voice_pipeline_integration.py`, `ai/dataset_pipeline/ingestion/youtube_processor.py`  
**Status**: ⏳ **In Progress** (Transcripts available; synthetic expansion in progress)

**Sources**:
- 28 YouTube playlists with authentic conversational style
- Tim Fletcher complex trauma series transcripts
- Expert psychology lectures and presentations
- Clinical conversation recordings
- Professional therapy demonstrations

**Processing**:
- Audio download and transcription (Faster Whisper)
- Personality marker extraction (Big Five)
- Dialogue pair construction
- Naturalness validation

---

### Dual Persona Training

**Location**: `ai/lightning/ghost/dual_persona_training/`  
**Status**: ✅ **Active**

**Features**:
- Multiple therapeutic personas (empathetic listener, cognitive restructurer, crisis interventionist)
- Context-aware persona switching
- Persona consistency validation
- Curriculum learning framework

---

### Journal Research Datasets

**Location**: `ai/journal_dataset_research/integration/`  
**Integration Service**: `ai/journal_dataset_research/integration/pipeline_integration_service.py`  
**Status**: ⏳ **Ongoing acquisition and integration**

**Process**:
1. Discovery via PubMed, DOAJ, repositories (Dryad, Zenodo, ClinicalTrials.gov)
2. Evaluation across 4 dimensions (therapeutic relevance, data structure, integration potential, ethical accessibility)
3. Acquisition via direct download, API, or request forms
4. Integration planning and conversion to training format
5. Integration into training pipeline

---

### HuggingFace Datasets (External Sources)

**Successfully Acquired**:
- ✅ `Amod/mental_health_counseling_conversations` (~15K conversations)
- ✅ `EmoCareAI/Psych8k` (8K psychology examples)
- ✅ `samhog/psychology-10k` (10K psychology knowledge)
- ✅ `wesley7137/formatted_annotated_addiction_counseling_csv_SFT` (~5K addiction counseling)

**Additional HuggingFace Sources** (from TRAINING_DATA.md - need verification):
- ⏳ `Locutusque/hercules-v6.9` (1M+ conversations, personality balancing)
- ⏳ `ChaoticNeutrals/Synthetic-Dark-RP` (role-playing capabilities)
- ⏳ `UnfilteredAI/dan_remixed` (removes excessive safety constraints)
- ⏳ `jondurbin/gutenberg-dpo-v0.1` (human-like writing capability)
- ⏳ `Gryphe/Sonnet3.5-SlimOrcaDedupCleaned-20k` (high-quality instruction following)

**Action Required**: Verify which are downloaded vs. need acquisition

---

## Part 3: Download Missing Priority Data

### Tier 1 Priority Datasets (1.16GB)

```bash
# On remote server - download wendy priority datasets
rclone copy gdrive:datasets/datasets-wendy ~/datasets/consolidated/priority_wendy/ --progress
```

### Tier 3 Missing CoT Datasets (86MB)

```bash
# Download missing CoT datasets
rclone copy gdrive:datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions ~/datasets/consolidated/cot/
rclone copy gdrive:datasets/CoT_Philosophical_Understanding ~/datasets/consolidated/cot/
```

### Tier 4 Missing Reddit Data (700MB+)

```bash
# Download additional reddit mental health
rclone copy gdrive:datasets/reddit_mental_health/mental_disorders_reddit.csv ~/datasets/consolidated/reddit/
rclone copy gdrive:datasets/reddit_mental_health/Suicide_Detection.csv ~/datasets/consolidated/reddit/
rclone copy gdrive:datasets/reddit_mental_health/merged_mental_health_dataset.jsonl ~/datasets/consolidated/reddit/
```

### Tier 2 Missing Professional Datasets

```bash
# Download LLAMA3 and additional professional datasets
rclone copy gdrive:datasets/LLAMA3_Mental_Counseling_Data ~/datasets/consolidated/professional/
```

---

## Part 4: Deep Dataset Discovery

### 4A: Journal Research System - Academic Sources

#### Search 1: Psychotherapy Transcripts

```bash
python -m ai.sourcing.journal.main \
    --keywords "psychotherapy transcript corpus" "counseling dialogue dataset" "therapeutic conversation" \
    --sources "zenodo" "dryad" "pubmed" "doaj" \
    --session-id "psychotherapy_search_2025"
```

#### Search 2: Clinical Reasoning

```bash
python -m ai.sourcing.journal.main \
    --keywords "clinical reasoning dataset" "diagnostic reasoning corpus" "medical decision making" \
    --sources "zenodo" "pubmed" "doaj" \
    --session-id "clinical_reasoning_2025"
```

#### Search 3: Emotion Recognition

```bash
python -m ai.sourcing.journal.main \
    --keywords "emotion recognition dialogue" "empathetic response dataset" "emotional intelligence corpus" \
    --sources "zenodo" "dryad" "pubmed" \
    --session-id "emotion_recognition_2025"
```

#### Search 4: Crisis Intervention

```bash
python -m ai.sourcing.journal.main \
    --keywords "crisis intervention training" "suicide prevention dialogue" "emergency mental health" \
    --sources "zenodo" "pubmed" "clinicaltrials" \
    --session-id "crisis_intervention_2025"
```

#### Search 5: Trauma-Informed Care

```bash
python -m ai.sourcing.journal.main \
    --keywords "trauma informed care dialogue" "PTSD therapeutic conversation" "trauma therapy transcript" \
    --sources "zenodo" "dryad" "pubmed" \
    --session-id "trauma_care_2025"
```

#### Search 6: Motivational Interviewing

```bash
python -m ai.sourcing.journal.main \
    --keywords "motivational interviewing corpus" "MI training dataset" "behavioral change dialogue" \
    --sources "zenodo" "pubmed" "doaj" \
    --session-id "motivational_interviewing_2025"
```

---

### 4B: HuggingFace Deep Dive

**Categories to Explore**:
- Mental health conversation datasets
- Chain-of-thought reasoning (non-domain-specific)
- Instruction-following datasets with empathy
- Multi-turn dialogue datasets
- Emotional support conversation
- Crisis intervention training data
- Therapeutic alliance datasets
- Motivational interviewing corpora
- Cognitive behavioral therapy datasets
- Dialectical behavior therapy examples

**Specific HuggingFace Search Queries**:

```
mental health counseling
therapeutic conversation
empathetic dialogue
chain of thought reasoning
emotional support
crisis intervention
cognitive behavioral therapy
motivational interviewing
psychotherapy dataset
counselor training
mental health chatbot
depression anxiety support
trauma informed care
DBT dialectical behavior therapy
CBT cognitive behavioral therapy
therapeutic alliance
client therapist interaction
```

**Verify Existing HuggingFace Datasets**:
- Check which of the 5 additional datasets from TRAINING_DATA.md are already downloaded
- Download missing ones if needed
- Evaluate for integration potential

---

### 4C: Experimental/Novel Sources to Investigate

**Academic/Research Platforms**:
- **OpenPsych datasets** - Academic psychology research data
- **CLPsych shared tasks** - Computational linguistics + psychology
- **BioNLP competitions** - Biomedical NLP challenges
- **EmpatheticDialogues** - Facebook's empathy dataset
- **ESConv** - Emotional support conversation dataset
- **MELD** - Multimodal emotion lines dataset

**High-Quality Dialogue Datasets**:
- **DailyDialog** - High-quality multi-turn dialogues
- **PersonaChat** - Persona-based conversations
- **MultiWOZ** - Multi-domain task-oriented dialogues (adapted for therapy)

**Synthetic Generation**:
- Use existing models to generate edge cases
- Curriculum learning with progressive difficulty
- Domain-specific synthetic therapeutic dialogues

---

## Part 5: Create Master Manifest JSON

**Location**: `~/datasets/consolidated/dataset_manifest.json`

### Manifest Structure

```json
{
  "manifest_version": "2.0",
  "generated": "2025-01-27",
  "training_strategy": {
    "ratios": {
      "psychology": 30,
      "voice": 25,
      "mental_health": 20,
      "reasoning": 15,
      "personality": 10
    },
    "target_conversations": 100000
  },
  "tiers": {
    "tier1_priority": {
      "quality_threshold": 0.99,
      "training_weight": 0.40,
      "datasets": [
        {
          "name": "priority_1_FINAL",
          "path": "~/datasets/consolidated/priority_wendy/priority_1_FINAL.jsonl",
          "size_mb": 462,
          "format": "jsonl",
          "status": "missing",
          "estimated_conversations": null
        }
      ]
    },
    "tier2_professional": {
      "quality_threshold": 0.95,
      "training_weight": 0.25,
      "datasets": []
    },
    "tier3_cot_reasoning": {
      "quality_threshold": 0.90,
      "training_weight": 0.20,
      "datasets": []
    },
    "tier4_reddit": {
      "quality_threshold": 0.85,
      "training_weight": 0.10,
      "datasets": []
    },
    "tier5_research": {
      "quality_threshold": 0.80,
      "training_weight": 0.04,
      "datasets": []
    },
    "tier6_knowledge": {
      "quality_threshold": 1.0,
      "training_weight": 0.01,
      "datasets": []
    }
  },
  "additional_sources": {
    "edge_case_generation": {
      "location": "ai/lightning/ghost/edge_case_pipeline_standalone/",
      "categories": 25,
      "status": "active"
    },
    "voice_pipeline": {
      "location": "ai/dataset_pipeline/voice_pipeline_integration.py",
      "sources": 28,
      "status": "in_progress"
    },
    "dual_persona": {
      "location": "ai/lightning/ghost/dual_persona_training/",
      "status": "active"
    },
    "journal_research": {
      "location": "ai/journal_dataset_research/integration/",
      "status": "ongoing"
    }
  },
  "totals": {
    "current_size_gb": "~6.4GB",
    "missing_to_download_gb": "~1.86GB",
    "estimated_conversations": "300k+",
    "target_conversations": 100000
  },
  "format_breakdown": {
    "json": "45%",
    "jsonl": "25%",
    "csv": "20%",
    "arrow": "10%"
  },
  "gaps_identified": [
    "More crisis intervention data",
    "Trauma-informed care conversations",
    "Motivational interviewing examples",
    "Cross-cultural therapeutic dialogues",
    "DBT-specific examples",
    "CBT-specific examples"
  ],
  "processing_pipeline": {
    "tier_processors": [
      "ai/dataset_pipeline/orchestration/tier_processor.py",
      "ai/dataset_pipeline/composition/tier_balancer.py"
    ],
    "tier_loaders": [
      "ai/dataset_pipeline/ingestion/tier_loaders/tier1_priority_loader.py",
      "ai/dataset_pipeline/ingestion/tier_loaders/tier2_professional_loader.py",
      "ai/dataset_pipeline/ingestion/tier_loaders/tier3_cot_loader.py",
      "ai/dataset_pipeline/ingestion/tier_loaders/tier4_reddit_loader.py",
      "ai/dataset_pipeline/ingestion/tier_loaders/tier5_research_loader.py",
      "ai/dataset_pipeline/ingestion/tier_loaders/tier6_knowledge_loader.py"
    ],
    "format_converter": "ai/dataset_pipeline/processing/format_converter.py",
    "chatml_converter": "ai/dataset_pipeline/processing/convert_chatml.py",
    "quality_validators": [
      "ai/dataset_pipeline/quality/coherence_validator.py",
      "ai/dataset_pipeline/quality/therapeutic_accuracy.py"
    ]
  }
}
```

---

## Part 6: Data Processing Pipeline Integration

### Processing Flow

1. **Tier Loaders**: Process datasets according to tier-specific quality thresholds
2. **Tier Processor**: Aggregates all tier datasets via `ai/dataset_pipeline/orchestration/tier_processor.py`
3. **Tier Balancer**: Applies training ratio weights via `ai/dataset_pipeline/composition/tier_balancer.py`
4. **Format Conversion**: Convert to ChatML format via `ai/dataset_pipeline/processing/convert_chatml.py`
5. **Quality Validation**:
   - Semantic coherence scoring (0-1 scale)
   - Therapeutic appropriateness validation
   - PII detection and removal
   - Bias detection
6. **Balanced Sampling**: Apply training ratio strategy (30% Psychology, 25% Voice, 20% Mental Health, 15% Reasoning, 10% Personality)
7. **Export**: Output to training format (ChatML or ConversationRecord)

### Integration Points

**Journal Research → Training Pipeline**:
- Integration Service: `ai/journal_dataset_research/integration/pipeline_integration_service.py`
- Pipeline Integrator: `ai/journal_dataset_research/integration/pipeline_integrator.py`
- Integration Planning: `ai/journal_dataset_research/integration/integration_planning_engine.py`
- Journal Research Adapter: `ai/dataset_pipeline/orchestration/journal_research_adapter.py`

**MCP Server → Dataset Acquisition**:
- MCP Tools: `ai/journal_dataset_research/mcp/tools/acquisition.py`
- MCP Pipeline Bridge: `ai/journal_dataset_research/integration/mcp_pipeline_bridge.py`

**Quality Validation → Training Data Selection**:
- Evaluation Score Filter: `ai/dataset_pipeline/quality/evaluation_score_filter.py`
- Quality Monitoring: `ai/monitoring/quality_analytics_dashboard.py`

---

## Execution Order

### Phase 1: Data Acquisition

1. **Download Missing GDrive Datasets**:
   - Tier 1: Priority datasets (1.16GB)
   - Tier 3: Missing CoT datasets (86MB)
   - Tier 4: Missing Reddit data (700MB+)
   - Tier 2: Missing professional datasets

2. **Verify HuggingFace Datasets**:
   - Check existing HuggingFace datasets
   - Download missing ones if needed

### Phase 2: Discovery & Evaluation

3. **Generate Complete Manifest JSON**:
   - Include all 6 tiers with quality thresholds and training weights
   - Document all datasets with sizes, formats, status
   - Include processing pipeline components
   - Identify gaps and missing data

4. **Run Journal Research Searches** (6 parallel searches):
   - Psychotherapy transcripts
   - Clinical reasoning
   - Emotion recognition
   - Crisis intervention
   - Trauma-informed care
   - Motivational interviewing

5. **Deep Dive HuggingFace**:
   - Search for experimental mental health datasets
   - Verify existing HuggingFace datasets from TRAINING_DATA.md
   - Evaluate and acquire promising discoveries

### Phase 3: Integration & Processing

6. **Evaluate and Acquire Promising Discoveries**:
   - Review journal research results
   - Evaluate HuggingFace discoveries
   - Plan integration for high-priority datasets

7. **Update Manifest with New Acquisitions**:
   - Add newly discovered datasets
   - Update status and locations
   - Recalculate totals

8. **Begin ChatML Conversion Pipeline**:
   - Use streaming processing script (`process_datasets_streaming.py`)
   - Process all datasets through tier loaders
   - Apply quality validation
   - Convert to ChatML format
   - Apply tier balancing and training ratio strategy

---

## Quality Metrics & Validation

### Processing Success Rates

**Target**: ≥99% processing success rate

**Metrics**:
- Successful processing rate per dataset
- Error rate by dataset type
- Recovery rate for failed processing
- Overall pipeline success rate

**Monitoring**: `ai/monitoring/quality_analytics_dashboard.py`

### Semantic Coherence Scores

**Scoring**: 0-1 scale for question-answer semantic alignment

**Validation**:
- Prevents generic/irrelevant Q&A mismatches
- Ensures contextual relevance
- Validates therapeutic appropriateness

**Implementation**: `ai/dataset_pipeline/quality/coherence_validator.py`

### Therapeutic Appropriateness Validation

**Checks**:
- Clinical accuracy validation
- DSM-5 accuracy checking
- Crisis intervention detection
- Emotional authenticity assessment
- Bias detection

**Implementation**: `ai/dataset_pipeline/quality/therapeutic_accuracy.py`

### Bias Detection

**Detection Areas**:
- Demographic bias
- Cultural bias
- Gender bias
- Socioeconomic bias

**Metrics**:
- Bias scores per dataset
- Fairness metrics
- Evidence-based practice validation

**Implementation**: `ai/dataset_pipeline/quality/`, `ai/lightning/ghost/run_ghost_bias_detection.py`

### Clinical Accuracy Metrics

**Validation**:
- DSM-5 diagnostic accuracy
- Therapeutic technique accuracy
- Clinical guideline compliance
- Evidence-based practice alignment

**Implementation**: `ai/dataset_pipeline/quality/therapeutic_accuracy.py`

---

## References

### Documentation

- **Training Data Documentation**: `.kiro/specs/foundation-model-training/TRAINING_DATA.md`
- **Dataset Registry**: `ai/data/dataset_registry.json`
- **Training Dataset Guide**: `ai/data/acquired_datasets/TRAINING_DATASET_GUIDE.md`
- **Dataset Pipeline Tasks**: `.notes/in-progress/tasks-dataset-MERGED.md`
- **Journal Research System**: `ai/journal_dataset_research/`

### Key Components

**Tier Loaders**:
- `ai/dataset_pipeline/ingestion/tier_loaders/tier1_priority_loader.py`
- `ai/dataset_pipeline/ingestion/tier_loaders/tier2_professional_loader.py`
- `ai/dataset_pipeline/ingestion/tier_loaders/tier3_cot_loader.py`
- `ai/dataset_pipeline/ingestion/tier_loaders/tier4_reddit_loader.py`
- `ai/dataset_pipeline/ingestion/tier_loaders/tier5_research_loader.py`
- `ai/dataset_pipeline/ingestion/tier_loaders/tier6_knowledge_loader.py`

**Processing Components**:
- `ai/dataset_pipeline/orchestration/tier_processor.py`
- `ai/dataset_pipeline/composition/tier_balancer.py`
- `ai/dataset_pipeline/processing/convert_chatml.py`
- `ai/dataset_pipeline/processing/format_converter.py`

**Quality Components**:
- `ai/dataset_pipeline/quality/coherence_validator.py`
- `ai/dataset_pipeline/quality/therapeutic_accuracy.py`
- `ai/dataset_pipeline/quality/evaluation_score_filter.py`

**Integration Components**:
- `ai/journal_dataset_research/integration/pipeline_integration_service.py`
- `ai/journal_dataset_research/integration/pipeline_integrator.py`
- `ai/dataset_pipeline/orchestration/journal_research_adapter.py`

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Current Consolidated Size** | ~6.4GB |
| **Missing to Download** | ~1.86GB |
| **Estimated Conversations** | 300K+ |
| **Target Conversations** | 100K |
| **Tier 1 Datasets** | 5 (all missing) |
| **Tier 2 Datasets** | 9 (all present) |
| **Tier 3 Datasets** | 17 (15 present, 2 missing) |
| **Tier 4 Datasets** | 50+ (most present, 3 large files missing) |
| **Tier 5 Datasets** | 8 (all present) |
| **Tier 6 Sources** | 5 (all present) |

---

**Document Maintained By**: Dataset Expansion Team  
**Last Updated**: 2025-01-27  
**Next Review**: Upon completion of Phase 1 (Data Acquisition)

