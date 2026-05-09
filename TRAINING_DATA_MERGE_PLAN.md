# Training Data Consolidation Plan

## Executive Summary

This plan merges the following directories into a unified, organized training data structure:

1. **ai/training** - Main training directory with configs, scripts, models, and some data
2. **ai/training_corpus** - Corpus building tools and assets
3. **ai/training_data** - External datasets (currently minimal)
4. **ai/training_data_consolidated** - Already consolidated data with stage splits
5. **ai/training_ready** - Empty placeholder for ready-to-use data
6. **ai/training_ready/data** - Empty subdirectory
7. **ai/training_ready/data/generated** - Empty subdirectory for generated data

## Current State Analysis

### Directory Purposes

#### 1. ai/training
- **Purpose**: Primary training infrastructure
- **Contents**:
  - `configs/` - Training configurations (hyperparameters, model configs, stage configs)
  - `scripts/` - Processing, merging, and upload scripts
  - `models/` - Model architectures (base, moe, experimental)
  - `sliced/` - Pre-sliced training data by stage
  - `youtube_transcripts/` - Raw transcript files from YouTube channels
  - `defense_mechanisms/` - Specialized defense mechanism training data
  - `rlhf/` - RLHF-related scripts (DPO preference maker)
  - `output/` - Processing outputs
  - `tests/` - Test files

#### 2. ai/training_corpus
- **Purpose**: Corpus building pipeline and tools
- **Contents**:
  - Pipeline scripts (assembly, builder, compose, curate)
  - Deduplication tools
  - Expansion and synthesis tools
  - Quality and integrity checks
  - Asset management
  - Source adapters and writers

#### 3. ai/training_data
- **Purpose**: External dataset storage
- **Contents**: 
  - `external_datasets/cactus_full_download/` - External downloads (currently empty)

#### 4. ai/training_data_consolidated
- **Purpose**: Consolidated training data with proper splits
- **Contents**:
  - `final/splits/` - Stage-based splits (stage1-4)
    - Each stage has train.jsonl, val.jsonl, test.jsonl
  - `transcripts/` - Raw transcript files (Tim Fletcher series)

#### 5. ai/training_ready
- **Purpose**: Placeholder for production-ready training packages
- **Contents**: Currently empty

#### 6. ai/training_ready/data
- **Purpose**: Subdirectory for organized data
- **Contents**: Currently empty

#### 7. ai/training_ready/data/generated
- **Purpose**: Generated training data
- **Contents**: Currently empty

### Key Observations

1. **Duplication**: YouTube transcripts exist in multiple locations:
   - `ai/training/youtube_transcripts/`
   - `ai/training_data_consolidated/transcripts/`
   - `ai/training/sliced/stage1_foundation/` (processed)

2. **Stage Structure**: The canonical stage organization is:
   - Stage 1: Foundation (Psychology-6K, specialized filtered data)
   - Stage 2: Therapeutic Expertise (Clinical diagnosis, cultural nuances)
   - Stage 3: Edge Stress Test
   - Stage 4: Voice Persona

3. **Data Formats**:
   - Raw: `.txt` files (transcripts)
   - Processed: `.jsonl` files (train/val/test splits)
   - Configs: `.json`, `.yaml` files

4. **Scripts Location**: Processing scripts are scattered:
   - `ai/training/scripts/` - Main processing scripts
   - `ai/training_corpus/` - Corpus building scripts
   - `ai/training/ready_packages/scripts/` - Additional scripts

## Target Architecture

### Proposed Unified Structure

```
ai/training_data/                    # NEW - Single source of truth
├── raw/                             # Raw, unprocessed data
│   ├── transcripts/                 # All raw transcripts
│   │   ├── youtube/                 # YouTube transcripts by channel
│   │   ├── books/                   # Book excerpts
│   │   └── other/                   # Other sources
│   └── external/                    # External dataset downloads
│
├── processed/                       # Processed and cleaned data
│   ├── stage1_foundation/           # Foundation training data
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── stage2_therapeutic_expertise/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── stage3_edge_stress_test/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── stage4_voice_persona/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── generated/                       # AI-generated synthetic data
│   ├── edge_cases/
│   ├── crisis_scenarios/
│   └── preference_pairs/
│
├── configs/                         # Training configurations (from ai/training/configs)
│   ├── hyperparameters/
│   ├── model_configs/
│   └── stage_configs/
│
├── scripts/                         # All processing scripts
│   ├── corpus_builder/              # From ai/training_corpus
│   ├── data_processing/             # From ai/training/scripts
│   └── upload_deploy/               # Upload and deployment scripts
│
├── models/                          # Model architectures (from ai/training/models)
│   ├── base/
│   ├── moe/
│   └── experimental/
│
├── tools/                           # Utility tools
│   ├── deduplication/
│   ├── quality_check/
│   └── validation/
│
└── docs/                            # Documentation
    ├── DATA_ORGANIZATION.md
    ├── PROCESSING_PIPELINE.md
    └── STAGE_DEFINITIONS.md
```

## Migration Strategy

### Phase 1: Preparation (Safety First)

1. **Create Backup**
   ```bash
   # Create timestamped backup of all training data directories
   BACKUP_DIR="/tmp/training_backup_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$BACKUP_DIR"
   
   # Copy all source directories
   cp -r ai/training "$BACKUP_DIR/training"
   cp -r ai/training_corpus "$BACKUP_DIR/training_corpus"
   cp -r ai/training_data "$BACKUP_DIR/training_data"
   cp -r ai/training_data_consolidated "$BACKUP_DIR/training_data_consolidated"
   cp -r ai/training_ready "$BACKUP_DIR/training_ready"
   ```

2. **Create New Structure**
   ```bash
   mkdir -p ai/training_data/{raw/{transcripts/{youtube,books,other},external},processed/{stage1_foundation,stage2_therapeutic_expertise,stage3_edge_stress_test,stage4_voice_persona},generated/{edge_cases,crisis_scenarios,preference_pairs},configs,scripts/{corpus_builder,data_processing,upload_deploy},models,tools,docs}
   ```

### Phase 2: Migration Steps

#### Step 2.1: Migrate Raw Data

1. **YouTube Transcripts**
   - Source: `ai/training/youtube_transcripts/` and `ai/training_data_consolidated/transcripts/`
   - Destination: `ai/training_data/raw/transcripts/youtube/`
   - Action: Merge both sources, deduplicate by filename

2. **Other Transcripts**
   - Source: Various locations
   - Destination: `ai/training_data/raw/transcripts/other/`

3. **External Datasets**
   - Source: `ai/training_data/external_datasets/`
   - Destination: `ai/training_data/raw/external/`

#### Step 2.2: Migrate Processed Data

1. **Stage Splits** (PRIORITY - Most Important)
   - Source: `ai/training_data_consolidated/final/splits/`
   - Destination: `ai/training_data/processed/`
   - Action: Copy all stage directories (stage1-4)
   - These are the canonical train/val/test splits

2. **Defense Mechanisms Data**
   - Source: `ai/training/defense_mechanisms/data/`
   - Destination: `ai/training_data/processed/stage2_therapeutic_expertise/` (integrate)

#### Step 2.3: Migrate Scripts

1. **Corpus Builder Scripts**
   - Source: `ai/training_corpus/*.py`
   - Destination: `ai/training_data/scripts/corpus_builder/`
   - Action: Copy all Python scripts

2. **Data Processing Scripts**
   - Source: `ai/training/scripts/*.py` and `ai/training/scripts/*.sh`
   - Destination: `ai/training_data/scripts/data_processing/`
   - Action: Copy and update paths in scripts

3. **Upload/Deploy Scripts**
   - Source: `ai/training/ready_packages/scripts/`
   - Destination: `ai/training_data/scripts/upload_deploy/`

#### Step 2.4: Migrate Configs

1. **Training Configs**
   - Source: `ai/training/configs/`
   - Destination: `ai/training_data/configs/`
   - Action: Copy entire configs directory

2. **Model Configs**
   - Source: `ai/training/configs/model_configs/`
   - Destination: `ai/training_data/configs/model_configs/`

#### Step 2.5: Migrate Models

1. **Model Architectures**
   - Source: `ai/training/models/`
   - Destination: `ai/training_data/models/`
   - Action: Copy base, moe, experimental subdirectories

#### Step 2.6: Migrate Generated Data

1. **Generated Datasets**
   - Source: `ai/training/ready_packages/data/generated/`
   - Destination: `ai/training_data/generated/`
   - Action: Copy all generated data and stats

### Phase 3: Cleanup and Validation

1. **Remove Duplicates**
   - Identify files that exist in multiple locations
   - Keep only canonical versions
   - Update all references

2. **Update Script Paths**
   - Search for hardcoded paths in scripts
   - Update to new structure
   - Test critical scripts

3. **Validate Data Integrity**
   - Check file counts match before/after
   - Verify JSONL files are valid
   - Run checksums on critical files

4. **Update Documentation**
   - Create DATA_ORGANIZATION.md
   - Document stage definitions
   - Update processing pipeline docs

### Phase 4: Decommissioning

After validation:

1. **Mark Old Directories as Deprecated**
   - Add DEPRECATED.md files in old directories
   - Point to new location
   - Wait for migration period (e.g., 2 weeks)

2. **Archive or Remove**
   - After validation period, archive old directories
   - Or remove if confident in migration

## Detailed File Inventory

### Files to Preserve (Critical)

From `ai/training_data_consolidated/final/splits/`:
- `stage1_foundation/{train,val,test}.jsonl`
- `stage2_therapeutic_expertise/{train,val,test}.jsonl`
- `stage3_edge_stress_test/{train,val,test}.jsonl`
- `stage4_voice_persona/{train,val,test}.jsonl`

From `ai/training/configs/`:
- All `.json` and `.yaml` configuration files
- Hyperparameter files
- Stage configuration files

From `ai/training/scripts/`:
- All `.py` processing scripts
- All `.sh` shell scripts
- Upload and deployment scripts

From `ai/training_corpus/`:
- All pipeline scripts (*.py)
- ENGINE.md documentation
- Asset directories

From `ai/training/ready_packages/data/generated/`:
- All generated dataset statistics
- Synthetic data files
- Transcript stats

### Files to Review

- Duplicate transcript files (check for differences)
- Old backup files (*.backup, *.bak)
- Test output files
- Temporary files

## Risk Mitigation

### Potential Issues

1. **Path References in Scripts**
   - Risk: Scripts reference old paths
   - Mitigation: Search and replace all paths, test scripts

2. **Duplicate Data**
   - Risk: Wasting storage with duplicates
   - Mitigation: Deduplication pass after migration

3. **Broken Imports**
   - Risk: Python imports break
   - Mitigation: Update sys.path references, test imports

4. **Data Loss**
   - Risk: Files lost during migration
   - Mitigation: Full backup before starting, verify counts

### Rollback Plan

If issues occur:
1. Stop all processing jobs
2. Restore from backup
3. Investigate issues
4. Update plan and retry

## Success Criteria

- [ ] All critical data files migrated
- [ ] All scripts functional with new paths
- [ ] No data loss (verified by file counts)
- [ ] Documentation updated
- [ ] Team notified of new structure
- [ ] Old directories marked as deprecated
- [ ] Validation tests pass

## Timeline Estimate

- **Phase 1 (Preparation)**: 1-2 hours
- **Phase 2 (Migration)**: 4-8 hours (depending on data size)
- **Phase 3 (Validation)**: 2-4 hours
- **Phase 4 (Decommissioning)**: 1 hour + 2-week waiting period

**Total Active Work**: 8-15 hours
**Total Elapsed Time**: 2 weeks (including validation period)

## Next Steps

1. [ ] Review and approve this plan
2. [ ] Create backup of all directories
3. [ ] Execute Phase 1 (create new structure)
4. [ ] Execute Phase 2 (migrate data)
5. [ ] Execute Phase 3 (validate)
6. [ ] Execute Phase 4 (decommission old structure)
7. [ ] Update all documentation and notify team

---

## Appendix: Directory Size Estimates

Run these commands to get actual sizes:

```bash
# Check sizes before migration
du -sh ai/training
du -sh ai/training_corpus
du -sh ai/training_data
du -sh ai/training_data_consolidated
du -sh ai/training_ready

# Count files
find ai/training -type f | wc -l
find ai/training_corpus -type f | wc -l
find ai/training_data -type f | wc -l
find ai/training_data_consolidated -type f | wc -l
find ai/training_ready -type f | wc -l
```
