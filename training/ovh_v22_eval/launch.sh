#!/usr/bin/env bash
# Score the v22 Qwen3-8B adapters on the crisis suite inside an OVH GPU job.
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
# Lets a second flavor run against the same bucket without overwriting the
# first job's results.
OUT_PREFIX="${OUT_PREFIX:-}"

pip install --no-cache-dir --quiet --break-system-packages \
  "transformers==5.22.2" \
  "peft==0.21.3" \
  "accelerate==1.15.0" \
  "bitsandbytes>=0.45"

python - <<'PY'
import torch, transformers, peft, accelerate, bitsandbytes

print(f"torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__}", flush=True)
print(f"bitsandbytes {bitsandbytes.__version__} cuda available {torch.cuda.is_available()}", flush=True)
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
    --results-dir "/workspace/out/${OUT_PREFIX}eval-${label}-${split}" \
    --run-id "ovh-v22-${label}-${split}" \
    --device cuda \
    --dtype bfloat16 \
    --max-new-tokens 512 \
    $extra
}

# Print a compact, machine-readable summary of a finished run to stdout so the
# scores can be recovered from `ovhai job logs` even when the volume push at
# job teardown fails.
summarize() {
  python - "$1" <<'PY'
import json, pathlib, sys

run_dir = pathlib.Path(sys.argv[1])
results = run_dir / "results.jsonl"
if not results.exists():
    print("SUMMARY " + json.dumps({"run": run_dir.name, "error": "no results.jsonl"}), flush=True)
    raise SystemExit(0)

arms = {}
for line in results.read_text().splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    failed = [c.get("type", "?") for c in (record.get("state_metrics") or []) if not c.get("passed")]
    passed = record.get("status") == "success" and not failed
    arm = arms.setdefault(
        record.get("arm_id"),
        {"run": run_dir.name, "arm": record.get("arm_id"), "total": 0, "passed": 0, "failures": []},
    )
    arm["total"] += 1
    arm["passed"] += int(passed)
    # The whole reply goes to stdout: the volume push at job teardown has
    # already lost two full result sets, and the log is the only durable copy.
    print(
        "RECORD "
        + json.dumps(
            {
                "case_id": record.get("case_id"),
                "arm_id": record.get("arm_id"),
                "status": record.get("status"),
                "passed": passed,
                "failed_checks": failed,
                "response": record.get("response") or "",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not passed:
        arm["failures"].append(
            {
                "case": record.get("case_id"),
                "status": record.get("status"),
                "failed_checks": failed,
                "response": (record.get("response") or "")[:280],
            }
        )

for arm in arms.values():
    print("SUMMARY " + json.dumps(arm, ensure_ascii=False), flush=True)
PY
}

# The nf4 adapter was fitted against a 4-bit base, so it is scored on one:
# serving it on bf16 measures transfer, not the model that was trained.
for label in bf16 nf4; do
  quant=""
  if [ "$label" = nf4 ]; then
    quant="--load-in-4bit"
  fi
  run_split "/workspace/adapters/${label}" "$label" dev "$quant"
  summarize "/workspace/out/${OUT_PREFIX}eval-${label}-dev"
  run_split "/workspace/adapters/${label}" "$label" sealed "$quant --allow-sealed"
  summarize "/workspace/out/${OUT_PREFIX}eval-${label}-sealed"
done

echo "eval complete" > /workspace/out/DONE
