#!/bin/bash
# Training Data Migration Script
# Merges: training, training_corpus, training_data, training_data_consolidated, training_ready
# Into: training_data (unified structure)

set -e  # Exit on error

echo "========================================="
echo "Training Data Migration Script"
echo "========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/tmp/training_backup_$TIMESTAMP"
NEW_DIR="$BASE_DIR/training_data_unified"

echo "Base Directory: $BASE_DIR"
echo "Backup Directory: $BACKUP_DIR"
echo "New Unified Directory: $NEW_DIR"
echo ""

# Step 0: Pre-flight checks
echo "=== Step 0: Pre-flight Checks ==="

# Check if source directories exist
for dir in training training_corpus training_data_consolidated; do
    if [ ! -d "$BASE_DIR/$dir" ]; then
        echo "ERROR: Source directory $BASE_DIR/$dir does not exist"
        exit 1
    fi
done

echo "✓ All source directories exist"

# Check if new directory already exists
if [ -d "$NEW_DIR" ]; then
    echo "ERROR: New directory $NEW_DIR already exists. Please remove it first."
    exit 1
fi

echo "✓ New directory does not exist (good)"
echo ""

# Step 1: Create backup
echo "=== Step 1: Creating Backup ==="
echo "Creating backup at $BACKUP_DIR..."

mkdir -p "$BACKUP_DIR"
cp -r "$BASE_DIR/training" "$BACKUP_DIR/training"
cp -r "$BASE_DIR/training_corpus" "$BACKUP_DIR/training_corpus"
cp -r "$BASE_DIR/training_data" "$BACKUP_DIR/training_data"
cp -r "$BASE_DIR/training_data_consolidated" "$BACKUP_DIR/training_data_consolidated"
cp -r "$BASE_DIR/training_ready" "$BACKUP_DIR/training_ready"

echo "✓ Backup created successfully"
echo "Backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo ""

# Step 2: Create new directory structure
echo "=== Step 2: Creating New Directory Structure ==="

mkdir -p "$NEW_DIR"/{raw/{transcripts/{youtube,books,other},external},processed/{stage1_foundation,stage2_therapeutic_expertise,stage3_edge_stress_test,stage4_voice_persona},generated/{edge_cases,crisis_scenarios,preference_pairs},configs/{hyperparameters,model_configs,stage_configs},scripts/{corpus_builder,data_processing,upload_deploy},models/{base,moe,experimental},tools/{deduplication,quality_check,validation},docs}

echo "✓ Directory structure created"
echo ""

# Step 3: Migrate processed data (MOST IMPORTANT)
echo "=== Step 3: Migrating Processed Data (Stage Splits) ==="

# Copy stage splits from training_data_consolidated
for stage in stage1_foundation stage2_therapeutic_expertise stage3_edge_stress_test stage4_voice_persona; do
    if [ -d "$BASE_DIR/training_data_consolidated/final/splits/$stage" ]; then
        cp "$BASE_DIR/training_data_consolidated/final/splits/$stage"/*.jsonl "$NEW_DIR/processed/$stage/" 2>/dev/null || true
        echo "  ✓ Migrated $stage splits"
    fi
done

# Also check training/sliced for any additional data
for stage in stage1_foundation stage2_therapeutic_expertise; do
    if [ -d "$BASE_DIR/training/sliced/$stage" ]; then
        # Only copy if not already copied
        for file in train.jsonl val.jsonl test.jsonl; do
            if [ -f "$BASE_DIR/training/sliced/$stage/$file" ] && [ ! -f "$NEW_DIR/processed/$stage/$file" ]; then
                cp "$BASE_DIR/training/sliced/$stage/$file" "$NEW_DIR/processed/$stage/"
                echo "  ✓ Copied additional $stage/$file from training/sliced"
            fi
        done
    fi
done

echo "✓ Processed data migrated"
echo ""

# Step 4: Migrate raw transcripts
echo "=== Step 4: Migrating Raw Transcripts ==="

# YouTube transcripts from training/youtube_transcripts
if [ -d "$BASE_DIR/training/youtube_transcripts" ]; then
    cp -r "$BASE_DIR/training/youtube_transcripts"/* "$NEW_DIR/raw/transcripts/youtube/" 2>/dev/null || true
    echo "  ✓ Migrated YouTube transcripts from training/"
fi

# Transcripts from training_data_consolidated/transcripts
if [ -d "$BASE_DIR/training_data_consolidated/transcripts" ]; then
    cp "$BASE_DIR/training_data_consolidated/transcripts"/*.txt "$NEW_DIR/raw/transcripts/other/" 2>/dev/null || true
    echo "  ✓ Migrated transcripts from training_data_consolidated/"
fi

echo "✓ Raw transcripts migrated"
echo ""

# Step 5: Migrate configs
echo "=== Step 5: Migrating Configuration Files ==="

# Copy all configs
if [ -d "$BASE_DIR/training/configs" ]; then
    cp -r "$BASE_DIR/training/configs"/* "$NEW_DIR/configs/" 2>/dev/null || true
    echo "  ✓ Migrated training configs"
fi

echo "✓ Configuration files migrated"
echo ""

# Step 6: Migrate scripts
echo "=== Step 6: Migrating Scripts ==="

# Corpus builder scripts
if [ -d "$BASE_DIR/training_corpus" ]; then
    cp "$BASE_DIR/training_corpus"/*.py "$NEW_DIR/scripts/corpus_builder/" 2>/dev/null || true
    cp "$BASE_DIR/training_corpus"/*.md "$NEW_DIR/docs/" 2>/dev/null || true
    echo "  ✓ Migrated corpus builder scripts"
fi

# Data processing scripts from training/scripts
if [ -d "$BASE_DIR/training/scripts" ]; then
    cp "$BASE_DIR/training/scripts"/*.py "$NEW_DIR/scripts/data_processing/" 2>/dev/null || true
    cp "$BASE_DIR/training/scripts"/*.sh "$NEW_DIR/scripts/data_processing/" 2>/dev/null || true
    echo "  ✓ Migrated data processing scripts"
fi

# Upload/deploy scripts from ready_packages
if [ -d "$BASE_DIR/training/ready_packages/scripts" ]; then
    cp "$BASE_DIR/training/ready_packages/scripts"/*.py "$NEW_DIR/scripts/upload_deploy/" 2>/dev/null || true
    cp "$BASE_DIR/training/ready_packages/scripts"/*.sh "$NEW_DIR/scripts/upload_deploy/" 2>/dev/null || true
    echo "  ✓ Migrated upload/deploy scripts"
fi

echo "✓ Scripts migrated"
echo ""

# Step 7: Migrate models
echo "=== Step 7: Migrating Model Architectures ==="

if [ -d "$BASE_DIR/training/models" ]; then
    cp -r "$BASE_DIR/training/models"/* "$NEW_DIR/models/" 2>/dev/null || true
    echo "  ✓ Migrated model architectures"
fi

echo "✓ Model architectures migrated"
echo ""

# Step 8: Migrate generated data
echo "=== Step 8: Migrating Generated Data ==="

if [ -d "$BASE_DIR/training/ready_packages/data/generated" ]; then
    cp -r "$BASE_DIR/training/ready_packages/data/generated"/* "$NEW_DIR/generated/" 2>/dev/null || true
    echo "  ✓ Migrated generated data"
fi

echo "✓ Generated data migrated"
echo ""

# Step 9: Migrate tools
echo "=== Step 9: Migrating Utility Tools ==="

# Copy deduplication tools
if [ -f "$BASE_DIR/training_corpus/dedup_engine.py" ]; then
    cp "$BASE_DIR/training_corpus/dedup_engine.py" "$NEW_DIR/tools/deduplication/"
    echo "  ✓ Migrated deduplication tools"
fi

# Copy quality check tools
if [ -f "$BASE_DIR/training_corpus/quality.py" ]; then
    cp "$BASE_DIR/training_corpus/quality.py" "$NEW_DIR/tools/quality_check/"
    echo "  ✓ Migrated quality check tools"
fi

# Copy integrity tools
if [ -f "$BASE_DIR/training_corpus/integrity_sweep.py" ]; then
    cp "$BASE_DIR/training_corpus/integrity_sweep.py" "$NEW_DIR/tools/validation/"
    echo "  ✓ Migrated validation tools"
fi

echo "✓ Utility tools migrated"
echo ""

# Step 10: Create documentation
echo "=== Step 10: Creating Documentation ==="

cat > "$NEW_DIR/docs/MIGRATION_INFO.md" << EOF
# Training Data Migration Information

## Migration Date
$TIMESTAMP

## Source Directories
- $BASE_DIR/training
- $BASE_DIR/training_corpus
- $BASE_DIR/training_data
- $BASE_DIR/training_data_consolidated
- $BASE_DIR/training_ready

## Backup Location
$BACKUP_DIR

## New Structure
- raw/ - Raw, unprocessed data
- processed/ - Processed stage-based data (train/val/test splits)
- generated/ - AI-generated synthetic data
- configs/ - Training configurations
- scripts/ - Processing and deployment scripts
- models/ - Model architectures
- tools/ - Utility tools
- docs/ - Documentation

## Migration Script
Location: $SCRIPT_DIR/migrate_training_data.sh

## Next Steps
1. Validate all data files are present
2. Update script paths if needed
3. Test critical workflows
4. Mark old directories as deprecated
5. Update team documentation
EOF

echo "✓ Documentation created"
echo ""

# Step 11: Validation
echo "=== Step 11: Validation ==="

echo "Checking file counts..."
OLD_COUNT=$(find "$BASE_DIR/training" "$BASE_DIR/training_corpus" "$BASE_DIR/training_data_consolidated" -type f 2>/dev/null | wc -l)
NEW_COUNT=$(find "$NEW_DIR" -type f 2>/dev/null | wc -l)

echo "  Original files: $OLD_COUNT"
echo "  New files: $NEW_COUNT"

echo ""
echo "Checking critical files..."
CRITICAL_FILES=(
    "processed/stage1_foundation/train.jsonl"
    "processed/stage1_foundation/val.jsonl"
    "processed/stage1_foundation/test.jsonl"
    "processed/stage2_therapeutic_expertise/train.jsonl"
    "configs/training_config.json"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$NEW_DIR/$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ⚠ $file missing (may be expected)"
    fi
done

echo ""
echo "✓ Validation complete"
echo ""

# Summary
echo "========================================="
echo "Migration Complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  - Backup created: $BACKUP_DIR"
echo "  - New structure: $NEW_DIR"
echo "  - Original file count: $OLD_COUNT"
echo "  - New file count: $NEW_COUNT"
echo ""
echo "Next Steps:"
echo "  1. Review the new structure: ls -la $NEW_DIR"
echo "  2. Validate critical data files"
echo "  3. Test key scripts with new paths"
echo "  4. Update documentation"
echo "  5. When ready, mark old directories as deprecated"
echo ""
echo "To rollback (if needed):"
echo "  rm -rf $NEW_DIR"
echo "  mv $BACKUP_DIR/training $BASE_DIR/"
echo "  mv $BACKUP_DIR/training_corpus $BASE_DIR/"
echo "  mv $BACKUP_DIR/training_data $BASE_DIR/"
echo "  mv $BACKUP_DIR/training_data_consolidated $BASE_DIR/"
echo "  mv $BACKUP_DIR/training_ready $BASE_DIR/"
echo ""
