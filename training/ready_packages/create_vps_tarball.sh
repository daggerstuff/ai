#!/bin/bash
# Create tarball for VPS migration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT"
TARBALL_NAME="training_ready_vps_$(date +%Y%m%d_%H%M%S).tar.gz"

cd "$PROJECT_ROOT"

echo "📦 Creating VPS migration tarball..."
echo ""

# Create temporary directory for packaging
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Copy training_ready directory
echo "📋 Copying training_ready directory..."
cp -r ai/training_ready "$TMP_DIR/"

# Copy essential dataset_pipeline files (for imports)
echo "📋 Copying dataset_pipeline dependencies..."
mkdir -p "$TMP_DIR/ai/dataset_pipeline"
cp -r ai/dataset_pipeline/configs "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true
cp -r ai/dataset_pipeline/schemas "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true
cp -r ai/dataset_pipeline/types "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true
cp ai/dataset_pipeline/unified_preprocessing_pipeline.py "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true
cp -r ai/dataset_pipeline/orchestration "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true
cp -r ai/dataset_pipeline/ingestion "$TMP_DIR/ai/dataset_pipeline/" 2>/dev/null || true

# Create manifest of what's included
echo "📋 Creating package manifest..."
cat > "$TMP_DIR/PACKAGE_MANIFEST.txt" <<EOF
Training Ready VPS Migration Package
Generated: $(date)

Contents:
- ai/training_ready/          Complete training consolidation system
- ai/dataset_pipeline/        Essential dependencies (configs, schemas, types)

Key Files:
- GET_UP_TO_SPEED.md         Quick context for new session
- VPS_MIGRATION_GUIDE.md     Migration instructions
- ENV_QUICKSTART.md          Environment setup guide
- TRAINING_MANIFEST.json     Complete asset catalog
- scripts/                    All processing scripts
- install_dependencies.sh    Dependency installer

Next Steps:
1. Extract tarball on VPS
2. Read GET_UP_TO_SPEED.md
3. Follow VPS_MIGRATION_GUIDE.md
4. Set up environment per ENV_QUICKSTART.md
EOF

# Create tarball
echo "📦 Compressing..."
cd "$TMP_DIR"
tar -czf "$OUTPUT_DIR/$TARBALL_NAME" .

echo ""
echo "✅ Tarball created: $OUTPUT_DIR/$TARBALL_NAME"
echo ""
echo "📊 Package size: $(du -h "$OUTPUT_DIR/$TARBALL_NAME" | cut -f1)"
echo ""
echo "📋 Contents:"
tar -tzf "$OUTPUT_DIR/$TARBALL_NAME" | head -20
echo "..."
echo ""
echo "🚀 Ready to upload to VPS!"


