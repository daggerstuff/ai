# 🎯 MASTER TRAINING EPIC: Mental Health Dataset Consolidation & Training Pipeline

## Production Ready | December 29, 2025

> **Single Source of Truth** for all training dataset work, VPS execution, S3 streaming, and training curriculum.  
> This EPIC supersedes all scattered documentation and provides actionable tasks for all coding agents.

---

## 📋 EPIC SUMMARY

| Attribute | Value |
|-----------|-------|
| **Mission** | Deliver production-ready mental health training dataset with Tim Fletcher persona integration |
| **Status** | 🟢 70% Complete - Ready for Training Launch |
| **Dataset Size** | 52.20GB across 19,330 objects |
| **Location** | `s3://pixel-data/` (OVH S3 canonical) |
| **Format** | ChatML JSONL with metadata |
| **Model Target** | Wayfarer-2-12B / Harbringer-24B mental health specialization |

---

## 🏗️ ARCHITECTURE OVERVIEW

### Data Flow: Google Drive → VPS → S3 → Training

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Google Drive   │────▶│   VPS Server    │────▶│   OVH S3       │
│  (Staging)      │     │  (Processing)   │     │  (Canonical)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Dedup/Clean/   │     │  Training       │
                        │  Convert/Align  │     │  Scripts        │
                        └─────────────────┘     └─────────────────┘
```

### S3 Canonical Structure

```
s3://pixel-data/
├── gdrive/processed/
│   ├── professional_therapeutic/     # Stage 1: 3,512 conversations
│   ├── cot_reasoning/               # Stage 2: Clinical reasoning
│   ├── edge_cases/                  # Stage 3: Crisis scenarios
│   ├── priority/                    # Curated priority data
│   ├── cptsd/                       # CPTSD specialized
│   ├── addiction/                   # Addiction recovery
│   └── long_running_therapy/        # Extended sessions
├── voice/
│   └── tim_fletcher_persona/        # 913 transcripts, voice training
├── lightning/                       # Expert therapeutic data
├── final_dataset/
│   ├── manifest.json                # Authoritative manifest
│   ├── compiled/
│   │   └── final_training_dataset.jsonl
│   └── shards/
│       ├── train/*.jsonl
│       ├── val/*.jsonl
│       └── test/*.jsonl
└── exports/releases/vYYYY-MM-DD/    # Versioned releases
```

---

## 🚀 VPS STREAMING PIPELINE SETUP

### Why VPS + S3 Streaming?
- **Local connection too slow** for large uploads
- **52.20GB dataset** requires server-side processing
- **Stream directly** from S3 → process → push back
- **No local storage eaten** by datasets

### Environment Setup on VPS

```bash
# 1. Clone repository
git clone <repo> ~/pixelated
cd ~/pixelated

# 2. Install dependencies
pnpm install
cd ai && uv sync

# 3. Configure S3 credentials
export OVH_S3_BUCKET=pixel-data
export OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
export OVH_S3_ACCESS_KEY=<your-key>
export OVH_S3_SECRET_KEY=<your-secret>
export DATASET_STORAGE_BACKEND=s3

# 4. Configure rclone for Google Drive sync (if needed)
rclone config
# Name: gdrive
# Type: drive
# Scope: drive.readonly
```

### S3 Streaming Utilities

```python
# ai/training_ready/utils/s3_dataset_loader.py - Already implemented
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader(
    bucket="pixel-data",
    endpoint_url="https://s3.us-east-va.io.cloud.ovh.us"
)

# Stream JSONL without loading entire file into memory
for record in loader.stream_jsonl("s3://pixel-data/gdrive/processed/professional_therapeutic/conversations.jsonl"):
    process(record)
```

### Key VPS Commands

```bash
# Verify dataset integrity
python ai/training_ready/scripts/verify_final_dataset.py --report

# Compile final dataset (streams from S3, processes, uploads back)
python ai/training_ready/scripts/compile_final_dataset.py \
  --s3-bucket pixel-data \
  --upload-canonical

# Sync from Google Drive to S3 (background)
./ai/training_ready/platforms/ovh/sync-datasets.sh upload

# Launch training
./ai/ovh/run-training.sh launch --curriculum 2025 --dataset-verified

# Monitor training
wandb login && ./ai/ovh/monitor-training.sh
```

---

## 📊 TRAINING CURRICULUM 2025

### Phase A: Continued Pretraining (4 hours)
- Mental health text corpus
- Tim Fletcher transcripts (913 videos)
- Clinical documentation

### Phase B: 7-Stage SFT Curriculum (8-12 hours)

| Stage | Name | Weight | Dataset Size | Purpose |
|-------|------|--------|--------------|---------|
| 1 | Foundation Therapeutic Dialogue | 25% | 15.2GB | High-quality therapeutic conversations |
| 2 | Clinical Reasoning | 20% | 8.5GB | Chain-of-thought clinical reasoning |
| 3 | Crisis Stress Test | 15% | 12.8GB | Edge cases, crisis intervention |
| 4 | Tim Fletcher Persona | 15% | 7.1GB | Voice training, 913 transcripts |
| 5 | Long Running Therapy | 10% | 5.4GB | Extended sessions, continuity |
| 6 | Specialized Domains | 10% | 8.9GB | CPTSD, addiction, trauma |
| 7 | Simulator Tasks | 5% | 3.1GB | Roleplay, therapeutic simulation |

### Phase C: Preference Alignment (2 hours)
- ORPO/DPO/KTO implementation
- Human preference feedback integration

### Success Metrics

| Metric | Target |
|--------|--------|
| Crisis Response Accuracy | ≥85% |
| Voice Persona Matching | ≥90% |
| Clinical Reasoning Score | ≥80% |
| Cultural Competency | ≥75% |
| Dataset Coverage | 100% |

---

## 🔄 DATASET CONVERSION & ALIGNMENT

### Problem: Datasets in Various Formats
Many datasets don't match our ChatML training format. We need to:
1. **Detect format** automatically
2. **Convert** to ChatML (not erase)
3. **Validate** conversation structure
4. **Preserve** long-running conversations

### Supported Input Formats → ChatML Output

```python
# Input formats we handle:
# 1. conversation: [{from/role/speaker, content/text}, ...]
# 2. messages: [{role, content}, ...]  (already ChatML)
# 3. Human/Assistant turns
# 4. Client/Therapist turns
# 5. Alpaca format (instruction, input, output)
# 6. ShareGPT format

# Output: ChatML standard
{
  "messages": [
    {"role": "system", "content": "You are a therapeutic AI assistant..."},
    {"role": "user", "content": "I've been feeling anxious..."},
    {"role": "assistant", "content": "I hear that you're experiencing anxiety..."}
  ],
  "metadata": {
    "source_family": "professional_therapeutic",
    "content_hash": "sha256:...",
    "provenance": {...}
  }
}
```

### Conversion Pipeline

```bash
# Run format conversion on a dataset
python ai/dataset_pipeline/processing/convert_chatml.py \
  --input s3://pixel-data/raw/some_dataset.jsonl \
  --output s3://pixel-data/gdrive/processed/converted_dataset.jsonl \
  --format auto-detect

# Validate conversational structure
python ai/dataset_pipeline/validation/validate_conversations.py \
  --input s3://pixel-data/gdrive/processed/converted_dataset.jsonl \
  --min-turns 2 \
  --require-alternating
```

### Long-Running Conversation Extraction ✅ IMPLEMENTED

The `extract_long_running_therapy.py` script supports full S3 streaming with multiple input modes:

```bash
# Extract from default S3 sources (11 pre-configured datasets)
python ai/training_ready/scripts/extract_long_running_therapy.py

# Extract 30+ turns and upload directly to S3
python ai/training_ready/scripts/extract_long_running_therapy.py \
  --min-turns 30 \
  --upload-s3 \
  --s3-output-prefix gdrive/processed/long_running_therapy/

# Scan all JSONL files in an S3 prefix
python ai/training_ready/scripts/extract_long_running_therapy.py \
  --input-dir s3://pixel-data/gdrive/processed/ \
  --output extracted.jsonl

# Process specific S3 keys or local files
python ai/training_ready/scripts/extract_long_running_therapy.py \
  --source-key s3://pixel-data/gdrive/processed/professional_therapeutic/conversations.jsonl \
  --source-key /local/path/to/file.jsonl \
  --verbose

# Limit extraction for testing
python ai/training_ready/scripts/extract_long_running_therapy.py \
  --limit 100 \
  --verbose
```

**CLI Options:**
| Option | Description |
|--------|-------------|
| `--manifest` | Path to S3 manifest JSON |
| `--source-key` | S3 key or local path (repeatable) |
| `--input-dir` | S3 prefix or local dir to scan for JSONL files |
| `--min-turns` | Minimum conversation turns (default: 20) |
| `--limit` | Max conversations to extract |
| `--output` | Local output path (default: long_running_therapy.jsonl) |
| `--upload-s3` | Upload output to S3 after extraction |
| `--s3-output-prefix` | S3 prefix for upload (default: gdrive/processed/long_running_therapy/) |
| `--verbose` | Enable detailed progress logging |

### Role-Play Enhancement

```bash
# Generate roleplay scenarios from edge cases
python ai/training_ready/scripts/generate_edge_case_synthetic_dataset.py \
  --output s3://pixel-data/gdrive/processed/edge_cases/synthetic.jsonl \
  --categories all \
  --count 10000 \
  --roleplay-style therapeutic
```

---

## ✅ TASK CHECKLIST

### Phase 1: Foundation Completion (Weeks 1-2)

#### 1.1 Download Missing GDrive Data
- [ ] **Tier 1 Priority** (1.16GB, 40% training weight) - CRITICAL
  ```bash
  rclone copy gdrive:datasets/datasets-wendy ~/datasets/consolidated/priority_wendy/
  ```
  - [ ] `priority_1_FINAL.jsonl` (462MB)
  - [ ] `priority_2_FINAL.jsonl` (330MB)
  - [ ] `priority_3_FINAL.jsonl` (370MB)
  - [ ] `priority_4_FINAL.jsonl`
  - [ ] `priority_5_FINAL.jsonl`

- [ ] **Tier 3 CoT Datasets** (86MB)
  ```bash
  rclone copy gdrive:datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions ~/datasets/consolidated/cot/
  rclone copy gdrive:datasets/CoT_Philosophical_Understanding ~/datasets/consolidated/cot/
  ```

- [ ] **Tier 4 Reddit Data** (700MB+)
  ```bash
  rclone copy gdrive:datasets/reddit_mental_health/mental_disorders_reddit.csv ~/datasets/consolidated/reddit/
  rclone copy gdrive:datasets/reddit_mental_health/Suicide_Detection.csv ~/datasets/consolidated/reddit/
  ```

#### 1.2 Generate Missing Datasets
- [ ] **Edge Case Synthetic Dataset**
  ```bash
  python ai/training_ready/scripts/generate_edge_case_synthetic_dataset.py \
    --output ai/training_ready/data/generated/edge_case_synthetic.jsonl \
    --categories all --count 10000
  ```

- [x] **Long-Running Therapy Dataset** ✅ Script Enhanced
  ```bash
  # Extract from S3 and upload back
  python ai/training_ready/scripts/extract_long_running_therapy.py \
    --input-dir s3://pixel-data/gdrive/processed/ \
    --min-turns 20 \
    --upload-s3 \
    --verbose
  ```

- [ ] **CPTSD Dataset from Tim Fletcher Transcripts**
  ```bash
  python ai/training_ready/scripts/build_cptsd_dataset_from_transcripts.py \
    --input-dir ~/datasets/gdrive/tier4_voice_persona/Tim\ Fletcher/ \
    --output ai/training_ready/data/generated/cptsd_transcripts.jsonl
  ```

#### 1.3 Quality Optimization
- [ ] **Deduplication** (target: <1% duplicate rate)
  ```bash
  uv run python ai/training_ready/scripts/enhanced_deduplication.py --dry-run
  uv run python ai/training_ready/scripts/enhanced_deduplication.py --confirm
  ```

- [ ] **Encoding Fix** (UTF-8 normalization)
  ```bash
  python ai/training_ready/scripts/fix_encoding.py \
    --input-dir ~/datasets/consolidated/ \
    --output-dir ~/datasets/consolidated/fixed/
  ```

- [ ] **8-Gate Quality Validation**
  ```bash
  python ai/training_ready/scripts/verify_final_dataset.py --report
  ```
  - [ ] Coverage Gate: All 14 families present
  - [ ] Leakage Gate: No cross-split duplicates
  - [ ] Distribution Gate: Balanced splits (90/5/5)
  - [ ] PII Gate: No requires_review conversations
  - [ ] Provenance Gate: All conversations have provenance
  - [ ] Hash Gate: All conversations have valid content_hash
  - [ ] Split Gate: Holdout families only in test
  - [ ] Stats Gate: Distribution statistics present

#### 1.4 Final Dataset Compilation
- [ ] **Compile and Upload**
  ```bash
  python ai/training_ready/scripts/compile_final_dataset.py \
    --s3-bucket pixel-data \
    --upload-canonical
  ```

- [ ] **Verify S3 Upload**
  ```bash
  aws s3 ls s3://pixel-data/final_dataset/ --recursive
  ```

---

### Phase 2: Baseline Validation (Weeks 3-4)

#### 2.1 Stage 1 Training
- [ ] **Launch Foundation Training**
  ```bash
  python ai/training_ready/scripts/train_enhanced.py \
    --phase sft --stage 1 \
    --config ai/training_ready/configs/training_curriculum_2025.json
  ```

- [ ] **Monitor Metrics**
  - [ ] Empathy: ≥ 0.70
  - [ ] Therapeutic appropriateness: ≥ 0.75
  - [ ] Safety: ≥ 0.80

#### 2.2 Metrics Analysis
- [ ] Generate metrics dashboard
- [ ] Identify specific gaps
- [ ] Decision: Proceed to Phase 3 or optimize current data

---

### Phase 3: Conditional Strategic Expansion (Weeks 5-8)

*Only triggered if Phase 2 metrics show specific gaps*

#### 3.1 Journal Research Searches (6 parallel)
- [ ] Psychotherapy Transcripts Search
- [ ] Clinical Reasoning Search
- [ ] Emotion Recognition Search
- [ ] Crisis Intervention Search
- [ ] Trauma-Informed Care Search
- [ ] Motivational Interviewing Search

#### 3.2 HuggingFace Deep Dive
- [ ] Search mental health conversation datasets
- [ ] Search Chain-of-thought reasoning datasets
- [ ] Search emotional support datasets
- [ ] Evaluate and prioritize discoveries

#### 3.3 Integration
- [ ] Integrate top 5 discoveries
- [ ] Update manifest
- [ ] Re-run quality validation
- [ ] Re-train and validate improvement

---

## 🔒 COMPLIANCE & SAFETY

### Privacy Protection Checklist
- [ ] Zero PII leakage confirmed
- [ ] Context-preserving redaction applied
- [ ] Provenance tracking complete
- [ ] Licensed psychologist validation

### Crisis Protocol Verification
- [ ] Suicide/self-harm keyword detection
- [ ] Empathetic, safe response validation
- [ ] Crisis hotline references included
- [ ] Multi-expert review completed

### 8-Gate Verification System
All gates must pass before training launch:

| Gate | Description | Status |
|------|-------------|--------|
| Coverage | All 14 dataset families present | ⏳ |
| Leakage | No cross-split duplicates | ⏳ |
| Distribution | Balanced 90/5/5 splits | ⏳ |
| PII | No requires_review conversations | ⏳ |
| Provenance | All records have source tracking | ⏳ |
| Hash | All records have content_hash | ⏳ |
| Split | Holdout families only in test | ⏳ |
| Stats | Distribution statistics generated | ⏳ |

---

## 📁 KEY FILE REFERENCES

### Configuration Files
| File | Purpose |
|------|---------|
| `ai/training_ready/configs/training_curriculum_2025.json` | Training curriculum |
| `ai/training_ready/data/s3_manifest.json` | S3 inventory |
| `ai/training_ready/data/dataset_routing_config.json` | Family routing |
| `ai/training_ready/data/dataset_coverage_report.json` | Coverage status |

### Key Scripts
| Script | Purpose |
|--------|---------|
| `ai/training_ready/scripts/compile_final_dataset.py` | Dataset compilation |
| `ai/training_ready/scripts/verify_final_dataset.py` | 8-gate validation |
| `ai/training_ready/scripts/enhanced_deduplication.py` | Deduplication |
| `ai/training_ready/scripts/generate_edge_case_synthetic_dataset.py` | Edge case generation |
| `ai/training_ready/scripts/build_cptsd_dataset_from_transcripts.py` | CPTSD from transcripts |
| `ai/training_ready/scripts/extract_long_running_therapy.py` | Long-running conversation extraction (20+ turns) |

### Documentation
| Document | Purpose |
|----------|---------|
| `ai/training_ready/MASTER_TRAINING_EPIC.md` | **THIS FILE** - Single source of truth |
| `ai/training_ready/docs/S3_TRAINING_DATA_STRUCTURE.md` | S3 layout |
| `ai/training_ready/TRAINING_PLAN.md` | Training strategy |
| `ai/dataset_pipeline/MasterTrainingPlan.md` | 4-stage training ladder |
| `docs/prd/mental-health-datasets-expansion.md` | PRD |
| `docs/epics/mental-health-datasets-expansion.md` | Epic |

---

## 🎯 IMMEDIATE ACTIONS (Copy-Paste Ready)

### For VPS Setup
```bash
# SSH to VPS
ssh user@vps-server

# Clone and setup
git clone <repo> ~/pixelated && cd ~/pixelated
pnpm install && cd ai && uv sync

# Configure S3
cat >> ~/.bashrc << 'EOF'
export OVH_S3_BUCKET=pixel-data
export OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
export OVH_S3_ACCESS_KEY=<your-key>
export OVH_S3_SECRET_KEY=<your-secret>
export DATASET_STORAGE_BACKEND=s3
EOF
source ~/.bashrc
```

### For Dataset Verification
```bash
cd ~/pixelated
python ai/training_ready/scripts/verify_final_dataset.py --report
```

### For Training Launch
```bash
cd ~/pixelated
python ai/training_ready/scripts/compile_final_dataset.py --s3-bucket pixel-data
./ai/ovh/run-training.sh launch --curriculum 2025
```

---

## 📊 DATASET FAMILIES INVENTORY

| Family | Status | Count | Stage | Notes |
|--------|--------|-------|-------|-------|
| `mental_health_datasets` | ✅ Present | 450 | 1 | Largest family |
| `professional_therapeutic` | ✅ Present | 3,512 | 1 | High quality |
| `priority_datasets` | ✅ Present | - | 1 | Wendy curated |
| `cot_reasoning` | ✅ Present | - | 2 | Clinical CoT |
| `edge_case_generator` | ✅ Present | 33 | 3 | Crisis scenarios |
| `edge_case_resulting_chats` | ⚠️ Partial | 1 | 3 | Needs expansion |
| `edge_case_synthetic` | ⚠️ Partial | 1 | 3 | Needs generation |
| `safety_guardrails_annihilator` | ✅ Present | 257 | 3 | Reddit archives |
| `voice_persona` | ✅ Present | 154 | 4 | Tim Fletcher |
| `video_transcripts` | ✅ Present | 403 | 4 | 913 videos |
| `cptsd` | ✅ Present | 296 | 6 | Needs tagging |
| `addiction` | ✅ Present | 32 | 6 | Adequate |
| `long_running_therapy` | ✅ Script Ready | 1 | 5 | Extraction script enhanced |
| `sarcasm` | ⚠️ Partial | 1 | 6 | Needs expansion |

---

## 🔗 JIRA & CONFLUENCE LINKS

- **Jira Project**: https://gemcityxyz.atlassian.net/browse/KAN
- **Confluence Index**: https://gemcityxyz.atlassian.net/wiki/spaces/PE/pages/7307265
  - Governance & Licensing: KAN-1
  - Ingestion & Quality Scoring: KAN-7
  - Quality-Aware Curriculum: KAN-2
  - Training & Ablations: KAN-5
  - Evaluation & Safety Gates: KAN-6
  - Observability & Drift: KAN-4
  - Documentation: KAN-3

---

## 📝 CHANGE LOG

| Date | Change | Author |
|------|--------|--------|
| 2025-12-29 | Created MASTER_TRAINING_EPIC consolidating all scattered docs | AI |
| 2025-12-29 | Tim Fletcher integration complete (913 transcripts) | Team |
| 2025-12-29 | 52.20GB dataset confirmed in S3 | Team |
| 2025-12-29 | Training curriculum 2025 finalized | Team |
| 2025-12-29 | Enhanced extract_long_running_therapy.py with S3 streaming, upload, and dir scanning | AI |

---

**Status: READY FOR IMMEDIATE TRAINING LAUNCH**

*This EPIC is the single source of truth for all training dataset work. Update this document when new stages, validators, or data sources are introduced.*
