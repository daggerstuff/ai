#!/bin/bash
# Install dependencies for dataset processing

echo "Installing Python dependencies for dataset merging and processing..."

# Check if Python is installed
if ! command -v python &>/dev/null; then
  echo "Python is not installed. Please install Python 3.8 or higher."
  exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(echo "$PYTHON_VERSION < 3.8" | bc)" -eq 1 ]]; then
  echo "Python version $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
  exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python -m venv venv
fi

# Activate the virtual environment
if [[ $OSTYPE == "msys" || $OSTYPE == "win32" ]]; then
  # Windows
  source venv/Scripts/activate
else
  # Unix-like
  source venv/bin/activate
fi

# Install required packages
echo "Installing required packages..."
if command -v uv >/dev/null 2>&1; then
  echo "Using uv to install packages (uv found)..."
  uv pip install --upgrade pip
  uv pip install datasets pandas requests numpy huggingface_hub python-dotenv
else
  echo "uv not found — falling back to pip installs"
  pip install --upgrade pip
  pip install datasets pandas requests numpy huggingface_hub python-dotenv
fi

# Additional packages for model training and data processing
if command -v uv >/dev/null 2>&1; then
  uv pip install transformers tokenizers scikit-learn nltk spacy
else
  pip install transformers tokenizers scikit-learn nltk spacy
fi

# Download spaCy language model
# Use uv-wrapped pip if available for reproducible install of spaCy model
if command -v uv >/dev/null 2>&1; then
  uv python -m spacy download en_core_web_sm
else
  python -m spacy download en_core_web_sm
fi

# Install PyTorch (CPU version to keep it lightweight)
if command -v uv >/dev/null 2>&1; then
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installation complete!"
echo "To use the dataset tools, run: pnpm merge-datasets"
echo "To prepare datasets for fine-tuning: pnpm prepare-all-formats"
