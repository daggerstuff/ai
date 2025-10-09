#!/bin/bash
"""
One-Command Setup Script for Pixelated Training
"""

set -e

echo "🚀 Pixelated Training Setup"
echo "=========================="

# Check system requirements
echo "📋 Checking system requirements..."

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ CUDA not detected - training will be slow"
fi

# Check disk space (need 500GB)
AVAILABLE=$(df . | tail -1 | awk '{print $4}')
REQUIRED=500000000  # 500GB in KB
if [ "$AVAILABLE" -lt "$REQUIRED" ]; then
    echo "❌ Insufficient disk space. Need 500GB+, have $(($AVAILABLE/1000000))GB"
    exit 1
else
    echo "✅ Sufficient disk space: $(($AVAILABLE/1000000))GB available"
fi

# Check memory (Lightning AI has 170GB)
MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
if [ "$MEMORY_GB" -lt 60 ]; then
    echo "⚠️ Low system memory: ${MEMORY_GB}GB (recommended: 64GB+)"
elif [ "$MEMORY_GB" -gt 160 ]; then
    echo "✅ Lightning AI detected: ${MEMORY_GB}GB RAM (optimal for training)"
else
    echo "✅ Sufficient memory: ${MEMORY_GB}GB"
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv add -r requirements.txt

# Install optional monitoring dependencies
echo "📊 Installing monitoring dependencies..."
uv add gputil psutil || echo "⚠️ Some monitoring packages failed to install"

# Setup authentication
echo "🔐 Setting up authentication..."
if [ ! -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc; then
    echo "Setting up WandB..."
    uv run python setup_wandb.py
else
    echo "✅ WandB already configured"
fi

# Verify HuggingFace
echo "🤗 Verifying HuggingFace authentication..."
if uv run python -c "from huggingface_hub import HfApi; print('✅ HF authenticated as:', HfApi().whoami()['name'])" 2>/dev/null; then
    echo "✅ HuggingFace authentication verified"
else
    echo "⚠️ HuggingFace not authenticated. Run: huggingface-cli login"
fi

# Run comprehensive tests
echo "🧪 Running validation tests..."
if uv run python comprehensive_test.py; then
    echo "✅ All validation tests passed"
else
    echo "❌ Some validation tests failed"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo ""
echo "✅ Setup complete! Ready to train."
echo ""
echo "🚀 To start training:"
echo "   uv run python train.py"
echo ""
echo "📊 To monitor training:"
echo "   uv run python system_monitor.py &"
echo "   tail -f logs/training.log"
echo ""
echo "🌐 Monitor at: https://wandb.ai/your-username/wayfarer-2-12b-finetuning"
