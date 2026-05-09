#!/bin/bash
# Migration Validation Script

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_DIR="$BASE_DIR/training_data_unified"

echo "========================================="
echo "Migration Validation"
echo "========================================="
echo ""

# Check if migration completed
if [ ! -d "$NEW_DIR" ]; then
    echo "❌ Migration not completed"
    echo "   Run: ./scripts/migrate_training_data.sh"
    exit 1
fi

echo "✓ Migration directory exists"
echo ""

# File counts
echo "=== File Counts ==="
for dir in raw processed generated configs scripts models tools docs; do
    if [ -d "$NEW_DIR/$dir" ]; then
        count=$(find "$NEW_DIR/$dir" -type f 2>/dev/null | wc -l)
        echo "  $dir/: $count files"
    fi
done
echo ""

# Critical files
echo "=== Critical Files ==="
CRITICAL=(
    "processed/stage1_foundation/train.jsonl"
    "processed/stage1_foundation/val.jsonl"
    "processed/stage1_foundation/test.jsonl"
    "processed/stage2_therapeutic_expertise/train.jsonl"
    "processed/stage2_therapeutic_expertise/val.jsonl"
    "processed/stage2_therapeutic_expertise/test.jsonl"
)

MISSING=0
for file in "${CRITICAL[@]}"; do
    if [ -f "$NEW_DIR/$file" ]; then
        size=$(wc -c < "$NEW_DIR/$file")
        echo "  ✓ $file ($size bytes)"
    else
        echo "  ❌ $file MISSING"
        MISSING=$((MISSING + 1))
    fi
done
echo ""

# Structure check
echo "=== Structure Check ==="
for dir in raw processed generated configs scripts models tools docs; do
    if [ -d "$NEW_DIR/$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ❌ $dir/ MISSING"
    fi
done
echo ""

# Summary
echo "========================================="
if [ $MISSING -eq 0 ]; then
    echo "✅ Validation PASSED"
    echo "========================================="
    echo ""
    echo "All critical files present."
    echo "Migration successful!"
else
    echo "⚠️  Validation WARNING"
    echo "========================================="
    echo ""
    echo "Missing $MISSING critical file(s)"
    echo "Review migration log for details."
fi
echo ""
