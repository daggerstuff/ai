#!/bin/bash
set -e

echo "==========================================================="
echo " Pixelated Empathy: Empathy Gym Training Entrypoint        "
echo "==========================================================="

echo "Python version environment verification:"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required for this launcher."
  exit 1
fi
uv run python --version

echo "1. Extracting codebase securely and bypassing Volume cache lag..."
mkdir -p /workspace/code/pixelated
wget -qO /tmp/repo.tar.gz "$TARBALL_URL"
tar -xzf /tmp/repo.tar.gz -C /workspace/code/pixelated

echo "2. Installing required dependencies natively in container..."
wget -qO /tmp/reqs.txt "$REQS_URL"
uv pip install --no-cache-dir -r /tmp/reqs.txt

echo "3. Setting up artifact symlinks to persistent S3 storage..."
cd /workspace/code/pixelated
mkdir -p /workspace/s3_cache/lightning_logs
# Remove if it exists locally to prevent ln errors on job restart
rm -rf ./lightning_logs
ln -s /workspace/s3_cache/lightning_logs ./lightning_logs

echo "4. Launching Distributed PyTorch Lightning Training Loop..."
export PYTHONPATH=/workspace/code/pixelated
uv run python ai/orchestrator/targets/lightning_production/train_therapeutic_ai.py \
  --stage 1 \
  --compute-backend gpu \
  --max-steps 100000

echo "==========================================================="
echo " Training Job Exited                                       "
echo "==========================================================="
