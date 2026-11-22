#!/usr/bin/env bash
# Entrypoint for the OVH v23 training job. The bootstrap has already pulled the
# export into $ROOT/data and the trainer into $ROOT/code, so this only installs
# the training stack and runs it. Versions are pinned to the v22 run so the two
# curves differ by dataset alone.
set -euo pipefail

# The image's /workspace is root-owned and read-only in a job with no volume, so
# everything lands under /tmp. HOME is pinned there too: pip's
# --break-system-packages installs land in the user site directory.
export HOME=/tmp
export PYTHONDONTWRITEBYTECODE=1
ROOT="${ROOT:-/tmp/work}"

echo "== ovh v23 launch $(date -u +%FT%TZ) =="
nvidia-smi || true

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

# The image's system python is PEP 668 managed with no ensurepip, so the install
# has to opt out of the marker.
pip install --no-cache-dir --quiet --break-system-packages \
  "transformers==5.22.2" \
  "peft==0.21.3" \
  "datasets==5.2.0" \
  "accelerate==1.15.0"

python -c 'import accelerate, datasets, peft, torch, transformers; print("stack ok", transformers.__version__, peft.__version__, datasets.__version__, accelerate.__version__, torch.__version__, torch.cuda.is_available())'

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export DATA_DIR="$ROOT/data"
export OUT_DIR="$ROOT/out"

cd "$ROOT/code"
python run_ovh.py
echo "== ovh v23 done $(date -u +%FT%TZ) =="
