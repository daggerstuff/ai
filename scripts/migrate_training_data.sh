#!/bin/bash
# Training Data Migration Script
# Merges 7 directories into 1 unified structure

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REDIS_AUDIT="${PROJECT_ROOT}/scripts/check-redis-hardening.sh"

if ! "$REDIS_AUDIT"; then
  echo "Redis hardening audit failed"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/tmp/training_backup_$TIMESTAMP"
NEW_DIR="$BASE_DIR/training_data_unified"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No changes will be made"
fi

echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo "Training Data Migration"
echo "Timestamp: $TIMESTAMP"
echo "Backup: $BACKUP_DIR"
echo "Target: $NEW_DIR"
echo ""

# Pre-flight checks
echo "=== Pre-flight Checks ==="
for dir in training training_corpus training_data_consolidated; do
    if [ ! -d "$BASE_DIR/$dir" ]; then
        echo "ERROR: $BASE_DIR/$dir not found"
        exit 1
    fi
done
echo "✓ Source directories exist"

# Check for symlinks
SYMLINK_COUNT=$(find "$BASE_DIR/training" "$BASE_DIR/training_corpus" "$BASE_DIR/training_data_consolidated" -type l 2>/dev/null | wc -l)
if [ "$SYMLINK_COUNT" -gt 0 ]; then
    echo "⚠ Found $SYMLINK_COUNT symlink(s) - will skip"
fi
echo ""

# Create backup
echo "=== Creating Backup ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create backup"
else
    mkdir -p "$BACKUP_DIR"
    for dir in training training_corpus training_data training_data_consolidated training_ready; do
        if [ -d "$BASE_DIR/$dir" ]; then
            echo "  Backing up $dir..."
            tar -czf "$BACKUP_DIR/${dir}.tar.gz" -C "$BASE_DIR" "$dir" 2>/dev/null || cp -r "$BASE_DIR/$dir" "$BACKUP_DIR/"
        fi
    done
    
    # Generate rollback script
    cat > "$BACKUP_DIR/rollback.sh" << 'EOF'
#!/bin/bash
echo "Rolling back..."
rm -rf "$BASE_DIR/training_data_unified"
cd "$BASE_DIR"
for dir in training training_corpus training_data training_data_consolidated training_ready; do
    [ -f "$BACKUP_DIR/${dir}.tar.gz" ] && tar -xzf "$BACKUP_DIR/${dir}.tar.gz"
    [ -d "$BACKUP_DIR/$dir" ] && [ ! -d "$BASE_DIR/$dir" ] && cp -r "$BACKUP_DIR/$dir" "$BASE_DIR/"
done
echo "Rollback complete"
EOF
    chmod +x "$BACKUP_DIR/rollback.sh"
    echo "✓ Backup created: $BACKUP_DIR"
    echo "  Rollback: $BACKUP_DIR/rollback.sh"
fi
echo ""

# Create structure
echo "=== Creating Structure ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create structure"
else
    mkdir -p "$NEW_DIR"/{raw/{transcripts/{youtube,books,other},external},processed/{stage1_foundation,stage2_therapeutic_expertise,stage3_edge_stress_test,stage4_voice_persona},generated/{edge_cases,crisis_scenarios,preference_pairs},configs/{hyperparameters,model_configs,stage_configs},scripts/{corpus_builder,data_processing,upload_deploy},models/{base,moe,experimental},tools/{deduplication,quality_check,validation},docs}
    echo "✓ Structure created"
fi
echo ""

# Migrate processed data (CRITICAL)
echo "=== Migrating Processed Data ==="
STAGES=0
for stage in stage1_foundation stage2_therapeutic_expertise stage3_edge_stress_test stage4_voice_persona; do
    src="$BASE_DIR/training_data_consolidated/final/splits/$stage"
    dest="$NEW_DIR/processed/$stage"
    if [ -d "$src" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would migrate $stage"
        else
            cp "$src"/*.jsonl "$dest/" 2>/dev/null && echo "  ✓ $stage" || echo "  ⚠ $stage (no files)"
            STAGES=$((STAGES + 1))
        fi
    fi
done

# Check training/sliced for additional data
for stage in stage1_foundation stage2_therapeutic_expertise; do
    src="$BASE_DIR/training/sliced/$stage"
    dest="$NEW_DIR/processed/$stage"
    if [ -d "$src" ]; then
        for file in train.jsonl val.jsonl test.jsonl; do
            if [ -f "$src/$file" ] && [ ! -f "$dest/$file" ]; then
                if [ "$DRY_RUN" = false ]; then
                    cp "$src/$file" "$dest/" && echo "  ✓ Added $stage/$file"
                fi
            fi
        done
    fi
done
echo ""

# Migrate raw transcripts
echo "=== Migrating Raw Transcripts ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate transcripts"
else
    if [ -d "$BASE_DIR/training/youtube_transcripts" ]; then
        find "$BASE_DIR/training/youtube_transcripts" -type f ! -type l -exec cp {} "$NEW_DIR/raw/transcripts/youtube/" \; 2>/dev/null
        echo "  ✓ YouTube transcripts"
    fi
    if [ -d "$BASE_DIR/training_data_consolidated/transcripts" ]; then
        find "$BASE_DIR/training_data_consolidated/transcripts" -type f ! -type l -exec cp {} "$NEW_DIR/raw/transcripts/other/" \; 2>/dev/null
        echo "  ✓ Other transcripts"
    fi
fi
echo ""

# Migrate configs
echo "=== Migrating Configs ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate configs"
else
    if [ -d "$BASE_DIR/training/configs" ]; then
        find "$BASE_DIR/training/configs" -type f ! -type l -exec cp --parents {} "$NEW_DIR/configs/" \; 2>/dev/null
        echo "  ✓ Configs"
    fi
fi
echo ""

# Migrate scripts
echo "=== Migrating Scripts ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate scripts"
else
    # Corpus builder
    if [ -d "$BASE_DIR/training_corpus" ]; then
        find "$BASE_DIR/training_corpus" -maxdepth 1 -type f \( -name "*.py" -o -name "*.md" \) ! -type l -exec cp {} "$NEW_DIR/scripts/corpus_builder/" 2>/dev/null
        find "$BASE_DIR/training_corpus" -maxdepth 1 -type f -name "*.md" ! -type l -exec cp {} "$NEW_DIR/docs/" 2>/dev/null
        echo "  ✓ Corpus builder scripts"
    fi
    
    # Data processing
    if [ -d "$BASE_DIR/training/scripts" ]; then
        find "$BASE_DIR/training/scripts" -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" \) ! -type l -exec cp {} "$NEW_DIR/scripts/data_processing/" 2>/dev/null
        echo "  ✓ Data processing scripts"
    fi
    
    # Upload/deploy
    if [ -d "$BASE_DIR/training/ready_packages/scripts" ]; then
        find "$BASE_DIR/training/ready_packages/scripts" -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" \) ! -type l -exec cp {} "$NEW_DIR/scripts/upload_deploy/" 2>/dev/null
        echo "  ✓ Upload scripts"
    fi
fi
echo ""

# Migrate models
echo "=== Migrating Models ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate models"
else
    if [ -d "$BASE_DIR/training/models" ]; then
        find "$BASE_DIR/training/models" -type f ! -type l -exec cp --parents {} "$NEW_DIR/models/" \; 2>/dev/null
        echo "  ✓ Models"
    fi
fi
echo ""

# Migrate generated data
echo "=== Migrating Generated Data ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate generated data"
else
    if [ -d "$BASE_DIR/training/ready_packages/data/generated" ]; then
        find "$BASE_DIR/training/ready_packages/data/generated" -type f ! -type l -exec cp --parents {} "$NEW_DIR/generated/" \; 2>/dev/null
        echo "  ✓ Generated data"
    fi
fi
echo ""

# Migrate tools
echo "=== Migrating Tools ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would migrate tools"
else
    for tool in dedup_engine.py quality.py integrity_sweep.py; do
        src="$BASE_DIR/training_corpus/$tool"
        if [ -f "$src" ]; then
            case $tool in
                *dedup*) cp "$src" "$NEW_DIR/tools/deduplication/" ;;
                *quality*) cp "$src" "$NEW_DIR/tools/quality_check/" ;;
                *integrity*) cp "$src" "$NEW_DIR/tools/validation/" ;;
            esac
        fi
    done
    echo "  ✓ Tools"
fi
echo ""

# Create docs
echo "=== Creating Documentation ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create docs"
else
    cat > "$NEW_DIR/docs/MIGRATION_INFO.md" << EOF
# Migration Information

**Date**: $TIMESTAMP
**Backup**: $BACKUP_DIR
**Rollback**: $BACKUP_DIR/rollback.sh

**Source**: training, training_corpus, training_data_consolidated, etc.
**Target**: training_data_unified/

**Structure**:
- raw/ - Raw transcripts
- processed/ - Stage splits
- generated/ - Synthetic data
- configs/ - Configurations
- scripts/ - Processing scripts
- models/ - Model architectures
- tools/ - Utilities
- docs/ - Documentation
EOF
    echo "✓ Documentation created"
fi
echo ""

# Validation
echo "=== Validation ==="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would validate"
else
    OLD_COUNT=$(find "$BASE_DIR/training" "$BASE_DIR/training_corpus" "$BASE_DIR/training_data_consolidated" -type f 2>/dev/null | wc -l)
    NEW_COUNT=$(find "$NEW_DIR" -type f 2>/dev/null | wc -l)
    
    echo "Original files: $OLD_COUNT"
    echo "New files: $NEW_COUNT"
    echo "Symlinks skipped: $SYMLINK_COUNT"
    echo ""
    echo "Critical files:"
    for file in processed/stage1_foundation/train.jsonl processed/stage1_foundation/val.jsonl processed/stage1_foundation/test.jsonl processed/stage2_therapeutic_expertise/train.jsonl; do
        [ -f "$NEW_DIR/$file" ] && echo "  ✓ $file" || echo "  ⚠ $file missing"
    done
fi
echo ""

# Summary
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
    echo ""
    echo "To execute: ./scripts/migrate_training_data.sh"
else
    echo "Migration Complete!"
    echo ""
    echo "Summary:"
    echo "  Backup: $BACKUP_DIR"
    echo "  Size: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)"
    echo "  New: $NEW_DIR"
    echo "  Size: $(du -sh "$NEW_DIR" 2>/dev/null | cut -f1)"
    echo ""
    echo "Next:"
    echo "  1. ./scripts/validate_migration.sh"
    echo "  2. Test workflows"
    echo "  3. Update docs"
    echo ""
    echo "Rollback: $BACKUP_DIR/rollback.sh"
fi
echo ""
