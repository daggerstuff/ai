#!/usr/bin/env bash
# Score the v22 Qwen3-8B adapters on the crisis suite inside an OVH GPU job.
#
# The adapters have no serving path, so the paired runner is driven by the local
# transport instead of a chat-completions endpoint: one copy of the base weights
# serves both arms and the LoRA is toggled between them.
#
# Inputs come from either the OVH volume mount at /workspace (ROOT=/workspace) or
# from R2 through the bootstrap that sets ROOT=/tmp/work. Results always land in
# /tmp/out and are pushed to R2 with the presigned PUTs in R2_MAP_B64, because
# the volume's write-back failed on every eval job so far
# (SYNC_FAILED : DATASYNC_RETRY_ERROR), taking the whole result set with it.
#
# HOME is pinned away from /workspace because pip's --break-system-packages
# installs land in the user site directory, and /workspace is the NFS bucket.
set -euo pipefail

export HOME=/tmp
# Keep bytecode out of any mounted filesystem: a stale .pyc from another
# interpreter version must never shadow the source that is actually mounted here.
export PYTHONDONTWRITEBYTECODE=1
ROOT="${ROOT:-/workspace}"
export PYTHONPATH="$ROOT/code"
OUT_ROOT="${OUT_ROOT:-/tmp/out}"
# Lets a second flavor run against the same bucket without overwriting the
# first job's results.
OUT_PREFIX="${OUT_PREFIX:-}"
mkdir -p "$OUT_ROOT"

# r2_sync imports boto3 eagerly; pyproject.toml already declares its floor, so
# this entry is unpinned on purpose.
pip install --no-cache-dir --quiet --break-system-packages \
  "transformers==5.22.2" \
  "peft==0.21.3" \
  "accelerate==1.15.0" \
  "boto3"

python - <<'PY'
import torch, transformers, peft, accelerate

print(f"torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__}", flush=True)
print(f"cuda available {torch.cuda.is_available()}", flush=True)
PY

cd "$ROOT/code"

# Fail fast, with a readable message, if any mounted module needs a newer
# interpreter than the one in this image.
python - "$ROOT/code" <<'PY'
import ast, pathlib, sys

root = pathlib.Path(sys.argv[1])
bad = []
for path in sorted(root.rglob("*.py")):
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
    --cases "$ROOT/data/scenario-crisis-${split}-v1.jsonl" \
    --results-dir "$OUT_ROOT/${OUT_PREFIX}eval-${label}-${split}" \
    --run-id "ovh-v22-${label}-${split}" \
    --device cuda \
    --dtype bfloat16 \
    --max-new-tokens 512 \
    $extra
}

# Print a compact, machine-readable summary of a finished run to stdout so the
# scores survive even if both the volume push and the R2 upload fail.
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

# Push the run directory to R2. A failure here is not fatal: the RECORD lines
# that summarize() already printed are a complete copy of every reply.
push_results() {
  local run_dir="$1"
  if python -m training.launch.r2_sync upload-run --run-dir "$run_dir"; then
    return 0
  fi
  echo "WARN R2 upload failed for $run_dir; the stdout RECORD lines are the only copy" >&2
  return 0
}

for label in bf16 nf4; do
  run_split "$ROOT/adapters/${label}" "$label" dev
  summarize "$OUT_ROOT/${OUT_PREFIX}eval-${label}-dev"
  push_results "$OUT_ROOT/${OUT_PREFIX}eval-${label}-dev"
  run_split "$ROOT/adapters/${label}" "$label" sealed --allow-sealed
  summarize "$OUT_ROOT/${OUT_PREFIX}eval-${label}-sealed"
  push_results "$OUT_ROOT/${OUT_PREFIX}eval-${label}-sealed"
done

echo "eval complete"
