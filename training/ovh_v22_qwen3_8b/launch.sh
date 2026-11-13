#!/usr/bin/env bash
# Entrypoint for the OVH AI Training job. The container image ships torch but not
# the training stack, and its system python is PEP 668 managed with no ensurepip,
# so the install has to opt out of the marker. Versions are pinned to what the
# Kaggle v6 run used, so precision is the only difference between the two curves.
set -euo pipefail

echo "== ovh v22 launch $(date -u +%FT%TZ) =="
df -h / /workspace || true
nvidia-smi || true

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

pip install --no-cache-dir --quiet --break-system-packages \
  "transformers==5.22.2" \
  "peft==0.21.3" \
  "datasets==5.2.0" \
  "accelerate==1.15.0"

python -c 'import accelerate, datasets, peft, torch, transformers; print("stack ok", transformers.__version__, peft.__version__, datasets.__version__, accelerate.__version__, torch.__version__, torch.cuda.is_available())'

# The container rootfs has terabytes free, so the ~16 GB base model stays on local
# disk instead of the network mount; re-downloading it is cheaper than reading it
# back over NFS on a retry.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /workspace/code
python run_ovh.py
echo "== ovh v22 done $(date -u +%FT%TZ) =="
