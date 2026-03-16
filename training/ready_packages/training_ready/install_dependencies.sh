#!/bin/bash
# Install dependencies for training_ready data processing pipeline
# CPU-only torch version for local execution

set -e

echo "📦 Installing dependencies for training_ready pipeline..."
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✅ Using uv for package management"
    echo ""
    echo "Installing torch (CPU-only)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    echo ""
    echo "Installing optional dependencies..."
    echo "  - datasets (for HuggingFace dataset sourcing)"
    uv pip install datasets || echo "⚠️  datasets installation failed (optional)"
    
    echo ""
    echo "✅ Dependencies installed in uv environment!"
    echo ""
    echo "💡 To run scripts, use:"
    echo "   uv run python3 ai/training_ready/scripts/prepare_training_data.py --all"
else
    echo "⚠️  uv not found. Using pip instead..."
    echo ""
    echo "Installing torch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    echo ""
    echo "Installing optional dependencies..."
    pip install datasets || echo "⚠️  datasets installation failed (optional)"
    
    echo ""
    echo "✅ Dependencies installed!"
fi

echo ""
echo "🎉 Ready to run data processing pipeline!"

