#!/usr/bin/env bash
# Score the v22 Qwen3-8B adapters on the crisis suite inside an OVH L40S job.
#
# The adapters have no serving path, so the paired runner is driven by the local
# transport instead of a chat-completions endpoint: one copy of the base weights
# serves both arms and the LoRA is toggled between them.
#
# HOME is pinned away from /workspace because pip's --break-system-packages
# installs land in the user site directory, and /workspace is the NFS bucket.
set -euo pipefail

export HOME=/tmp
export PYTHONPATH=/workspace/code
# Keep bytecode out of the NFS mount: a stale .pyc from another interpreter
# version must never shadow the source that is actually mounted here.
export PYTHONDONTWRITEBYTECODE=1

pip install --no-cache-dir --quiet --break-system-packages \
  "transformers==5.22.2" \
  "peft==0.21.3" \
  "accelerate==1.15.0"

python - <<'PY'
import torch, transformers, peft, accelerate

print(f"torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__}", flush=True)
print(f"cuda available {torch.cuda.is_available()}", flush=True)
PY

cd /workspace/code

# Fail fast, with a readable message, if any mounted module needs a newer
# interpreter than the one in this image.
python - <<'PY'
import ast, pathlib, sys

bad = []
for path in sorted(pathlib.Path("/workspace/code").rglob("*.py")):
    try:
        ast.parse(path.read_text(), filename=str(path), feature_version=sys.version_info[:2])
    except SyntaxError as error:
        bad.append(f"{path}:{error.lineno}: {error.msg}")
if bad:
    print(f"python {sys.version.split()[0]} cannot parse the mounted code:", flush=True)
    print("\n".join(bad), flush=True)
    raise SystemExit(2)
print(f"python {sys.version.split()[0]} parses every mounted module", flush=True)
PY

run_split() {
  local adapter="$1" label="$2" split="$3" extra="${4:-}"
  # shellcheck disable=SC2086
  python -m training.local_paired_eval \
    --base-model Qwen/Qwen3-8B \
    --adapter "$adapter" \
    --cases "/workspace/data/scenario-crisis-${split}-v1.jsonl" \
    --results-dir "/workspace/out/eval-${label}-${split}" \
    --run-id "ovh-v22-${label}-${split}" \
    --device cuda \
    --dtype bfloat16 \
    --max-new-tokens 512 \
    $extra
}

for label in bf16 nf4; do
  run_split "/workspace/adapters/${label}" "$label" dev
  run_split "/workspace/adapters/${label}" "$label" sealed --allow-sealed
done

echo "eval complete" > /workspace/out/DONE
