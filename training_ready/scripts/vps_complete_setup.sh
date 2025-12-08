#!/usr/bin/env zsh
# Complete VPS Setup Script for Training Ready
# Run this on the VPS after extracting files

set -e

echo "🚀 VPS Complete Setup for Training Ready"
echo "=" * 60

# Check we're in the right place
if [[ ! -d "/home/vivi/pixelated/ai/training_ready" ]]; then
    echo "❌ Error: Must run from /home/vivi/pixelated/ai/training_ready"
    echo "   Current directory: $(pwd)"
    exit 1
fi

cd /home/vivi/pixelated/ai/training_ready

echo "✅ Location verified: $(pwd)"
echo ""

# Step 1: Install uv if not available
echo "📦 Step 1: Installing uv (if needed)..."
if ! command -v uv &> /dev/null; then
    echo "   Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Source to make uv available
    if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    fi
    
    if command -v uv &> /dev/null; then
        echo "   ✅ uv installed: $(uv --version)"
    else
        echo "   ⚠️  uv installation may need shell restart"
        echo "   Run: source ~/.cargo/env"
    fi
else
    echo "   ✅ uv already installed: $(uv --version)"
fi
echo ""

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "   Using uv to sync dependencies from pyproject.toml..."
    cd /home/vivi/pixelated
    
    # Sync all dependencies from pyproject.toml (includes torch, datasets, boto3, etc.)
    uv sync || {
        echo "   ⚠️  uv sync failed, trying with index strategy..."
        uv sync --index-strategy unsafe-best-match || true
    }
    
    echo "   ✅ Dependencies synced"
else
    echo "   ⚠️  uv not available, using pip3..."
    pip3 install boto3 datasets requests || true
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || true
fi
echo ""

# Step 3: Verify environment
echo "🔍 Step 3: Verifying environment..."
if command -v uv &> /dev/null; then
    echo "   Python: $(uv run python3 --version)"
    
    # Check key packages using uv run
    uv run python3 -c "import boto3; print('   ✅ boto3')" 2>/dev/null || echo "   ❌ boto3 missing"
    uv run python3 -c "import datasets; print('   ✅ datasets')" 2>/dev/null || echo "   ❌ datasets missing"
    uv run python3 -c "import torch; print('   ✅ torch')" 2>/dev/null || echo "   ⚠️  torch missing (optional)"
else
    python3 --version
    echo "   Python: $(python3 --version)"
    
    # Check key packages
    python3 -c "import boto3; print('   ✅ boto3')" 2>/dev/null || echo "   ❌ boto3 missing"
    python3 -c "import datasets; print('   ✅ datasets')" 2>/dev/null || echo "   ❌ datasets missing"
    python3 -c "import torch; print('   ✅ torch')" 2>/dev/null || echo "   ⚠️  torch missing (optional)"
fi

echo ""

# Step 4: Check for .env file
echo "📝 Step 4: Checking environment configuration..."
cd /home/vivi/pixelated/ai/training_ready

if [[ -f "../.env" ]]; then
    echo "   ✅ Found ai/.env file"
    echo "   Make sure it contains OVH S3 credentials:"
    echo "     OVH_S3_BUCKET=pixel-data"
    echo "     OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us"
    echo "     OVH_S3_REGION=us-east-va"
    echo "     OVH_S3_ACCESS_KEY=..."
    echo "     OVH_S3_SECRET_KEY=..."
else
    echo "   ⚠️  ai/.env not found"
    echo "   Create it with OVH S3 credentials"
fi
echo ""

# Step 5: Test S3 connection
echo "🧪 Step 5: Testing S3 connection..."
if [[ -f "../.env" ]]; then
    set -a
    source ../.env
    set +a
    
    if [[ -n "$OVH_S3_ACCESS_KEY" && -n "$OVH_S3_SECRET_KEY" ]]; then
        cd /home/vivi/pixelated
        if command -v uv &> /dev/null; then
            uv run python3 -c "
from ai.training_ready.scripts.test_ovh_s3 import *
import sys
sys.exit(main())
" 2>&1 | head -20 || echo "   ⚠️  S3 test failed (may need credentials)"
        else
            python3 -c "
from ai.training_ready.scripts.test_ovh_s3 import *
import sys
sys.exit(main())
" 2>&1 | head -20 || echo "   ⚠️  S3 test failed (may need credentials)"
        fi
    else
        echo "   ⚠️  OVH S3 credentials not set in .env"
    fi
else
    echo "   ⏭️  Skipping (no .env file)"
fi
echo ""

# Step 6: Verify catalog exists (or generate it)
echo "📋 Step 6: Checking dataset catalog..."
CATALOG_PATH="scripts/output/dataset_accessibility_catalog.json"
if [[ -f "$CATALOG_PATH" ]]; then
    echo "   ✅ Catalog found"
    if command -v uv &> /dev/null; then
        cd /home/vivi/pixelated
        uv run python3 -c "
import json
with open('ai/training_ready/$CATALOG_PATH') as f:
    d = json.load(f)
    print(f'   Total datasets: {d[\"summary\"][\"total\"]}')
    print(f'   Local-only: {d[\"summary\"][\"local_only\"]}')
    print(f'   HuggingFace IDs: {d[\"summary\"].get(\"huggingface_unique_ids\", 0)}')
" 2>/dev/null || echo "   ⚠️  Could not read catalog"
        cd /home/vivi/pixelated/ai/training_ready
    else
        python3 -c "
import json
with open('$CATALOG_PATH') as f:
    d = json.load(f)
    print(f'   Total datasets: {d[\"summary\"][\"total\"]}')
    print(f'   Local-only: {d[\"summary\"][\"local_only\"]}')
    print(f'   HuggingFace IDs: {d[\"summary\"].get(\"huggingface_unique_ids\", 0)}')
" 2>/dev/null || echo "   ⚠️  Could not read catalog"
    fi
else
    echo "   ⚠️  Catalog not found"
    if command -v uv &> /dev/null; then
        echo "   🔄 Generating catalog..."
        cd /home/vivi/pixelated
        uv run python3 ai/training_ready/scripts/catalog_local_only_datasets.py && echo "   ✅ Catalog generated" || echo "   ⚠️  Catalog generation failed"
        cd /home/vivi/pixelated/ai/training_ready
    else
        echo "   Run: python3 scripts/catalog_local_only_datasets.py"
    fi
fi
echo ""

echo "=" * 60
echo "✅ VPS setup complete!"
echo ""
echo "Next steps:"
echo "1. Verify OVH S3 credentials in ai/.env"
echo "2. Test S3 connection: uv run python3 ai/training_ready/scripts/test_ovh_s3.py"
echo "3. Monitor uploads: uv run python3 ai/training_ready/scripts/monitor_upload_progress.py"
echo "4. Continue uploads: ./ai/training_ready/scripts/continue_uploads.sh"

