"""Upload a training-ready JSONL to Adaption RAW; persist the column mapping; request the free recipe estimate.

Mirrors the run-1 Dataset-B flow (pilot log §13.3): ``datasets.create`` with
``processing_mode="raw"`` + explicit ``column_mapping`` (prompt/completion)
-> presigned S3 PUT -> ``upload.complete_by_id`` -> poll ``get_status`` to
``succeeded``. RAW "materializes the data straight to a trainable dataset
with no augmentation" (SDK: processing_mode) — the run-2 anti-contamination
discipline; the plain ``upload.initiate`` route defaults to the adapt wizard
(chat-mapped, awaiting_input — observed and corrected 2026-10-13). Upload +
processing are free. Optionally calls ``autoscientist.recommend_hyperparams``
with NO augmentation parameters — nothing is launched; training runs are a
separate, owner-gated decision.

RAW ingest renames the source columns to ``original_prompt`` /
``original_completion`` but does not reliably record which of them plays which
role: the 11k arc-run2 dataset came back with ``prompt_column=original_prompt``
while the 217 MB v19 upload came back with ``prompt_column=None`` on the same
call (observed 2026-11-10). A dataset with no prompt column only trains if every
launch repeats ``column_mapping``, so the mapping is persisted here on the
dataset itself with an estimate-only ``datasets.adapt`` call — the SDK documents
that an estimate still saves ``column_mapping`` and costs nothing — and then read
back from ``datasets.get``. A dataset that fails that read-back fails the upload.

val/test splits are never uploaded (held-out discipline).

Run (from ai/, SDK is an ephemeral dependency):
  PYTHONPATH=.. uv run --with adaption python -m training.adaption_dataset_upload \
      --file training/output/adaption/adaption-sft-train.jsonl \
      --name pixelated-v19-selection-20261110 --model Qwen/Qwen3.6-35B-A3B

Repair the mapping on an already-uploaded dataset (no upload, no credits):
  PYTHONPATH=.. uv run --with adaption python -m training.adaption_dataset_upload \
      --dataset-id 3dc97e6c-b09a-4bef-b21d-6c694f3130ef
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

# RAW ingest canonicalises the source prompt/completion columns under these names.
CANONICAL_PROMPT = "original_prompt"
CANONICAL_COMPLETION = "original_completion"

# The account's plan rejects any launch above this many rows ("Samples to process
# (57573) exceeds your plan limit of 50000 rows per launch", observed 2026-11-10).
PLAN_ROW_LIMIT = 50_000


def read_column_mapping(client: Any, dataset_id: str) -> tuple[str | None, str | None]:
    """Return the prompt/completion roles recorded on the dataset itself."""
    configured = client.datasets.get(dataset_id).configured_column_mapping
    if configured is None:
        return (None, None)
    return (configured.prompt, configured.completion)


def configure_column_mapping(client: Any, dataset_id: str, max_rows: int | None = None) -> Any:
    """Persist the prompt/completion roles with a free estimate-only adapt call.

    The estimate is priced per row and the account rejects a launch above its plan
    row limit, so ``max_rows`` is passed when the dataset is larger than the plan
    allows — the mapping is saved either way.
    """
    job_specification = {"max_rows": max_rows} if max_rows is not None else {}
    return client.datasets.adapt(
        dataset_id,
        column_mapping={"prompt": CANONICAL_PROMPT, "completion": CANONICAL_COMPLETION},
        job_specification=job_specification,
        estimate=True,
    )


def ensure_column_mapping(client: Any, dataset_id: str, max_rows: int | None = None) -> bool:
    """Guarantee the dataset knows its prompt/completion columns; True when it does.

    Called before any launch so a paid AutoScientist run can never start against a
    dataset whose roles were only ever supplied per-request.
    """
    prompt, completion = read_column_mapping(client, dataset_id)
    if (prompt, completion) == (CANONICAL_PROMPT, CANONICAL_COMPLETION):
        print(f"column mapping already persisted: prompt={prompt} completion={completion}")
        return True

    print(f"column mapping missing (prompt={prompt} completion={completion}); saving it")
    quote = configure_column_mapping(client, dataset_id, max_rows=max_rows)
    print(
        f"adapt estimate: {quote.estimated_credits_consumed} credits / "
        f"{quote.estimated_minutes} min (estimate only, no run started)"
    )
    prompt, completion = read_column_mapping(client, dataset_id)
    if (prompt, completion) != (CANONICAL_PROMPT, CANONICAL_COMPLETION):
        print(f"FAIL: mapping did not persist (prompt={prompt} completion={completion})")
        return False
    print(f"column mapping persisted: prompt={prompt} completion={completion}")
    return True


def _evidence_path(evidence_dir: Path, stem: str, label: str) -> Path:
    """Name evidence after the dataset so a new upload never overwrites an old run's record."""
    return evidence_dir / f"{stem}_{label}.json"


def _upload(client: Any, args: argparse.Namespace, evidence_dir: Path) -> int:
    size = args.file.stat().st_size
    sha = hashlib.sha256(args.file.read_bytes()).hexdigest()
    print(f"raw ingest: {args.file.name} ({size:,} bytes, sha256 {sha[:16]}...)")

    created = client.datasets.create(
        source={
            "name": args.name,
            "file_format": "jsonl",
            "processing_mode": "raw",
            "column_mapping": {"prompt": "prompt", "completion": "completion"},
        }
    )
    dataset_id = created.dataset_id
    instructions = created.upload_instructions
    if instructions is None:
        print("FAIL: create returned no upload instructions")
        return 1
    print(f"dataset_id={dataset_id}; presigned PUT acquired (s3_key={instructions.s3_key[:40]}...)")

    with args.file.open("rb") as f, httpx.Client(timeout=1800.0) as http:
        resp = http.request(instructions.method, instructions.url, content=f)
    if resp.status_code != 200:
        print(f"FAIL: S3 PUT returned {resp.status_code}")
        return 1
    print("S3 PUT complete")

    client.datasets.upload.complete_by_id(dataset_id, file_size_bytes=size, sha256=sha)
    print("complete_by_id accepted; preprocessing triggered")
    record = {
        "dataset_id": dataset_id,
        "name": args.name,
        "file": str(args.file),
        "file_size_bytes": size,
        "sha256": sha,
        "processing_mode": "raw",
    }
    with _evidence_path(evidence_dir, "adaption_upload", args.name).open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)

    print("polling processing status (up to 30 min)...")
    status: Any = None
    for i in range(90):
        status = client.datasets.get_status(dataset_id)
        text = str(status.status).lower()
        print(f"[{i}] status={text}", flush=True)
        if text in ("succeeded", "failed", "error"):
            break
        time.sleep(20)
    if str(status.status).lower() != "succeeded":
        print(f"FAIL: dataset processing did not succeed (status={status.status})")
        return 1
    print(f"dataset READY (raw, no augmentation): {dataset_id}")

    if not ensure_column_mapping(client, dataset_id, max_rows=args.max_rows or None):
        return 1

    if args.skip_estimate or not args.model:
        print("estimate skipped")
        return 0

    # no augmentation parameters: corpus-only effective rows (anti-contamination)
    rec = client.autoscientist.recommend_hyperparams(dataset_id=dataset_id, model=args.model)
    payload = rec.model_dump()
    with _evidence_path(evidence_dir, "adaption_recommend_hyperparams", args.name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(json.dumps(payload, indent=1, default=str)[:4000])
    return 0


def main(argv: list[str] | None = None, client: Any | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=None, help="JSONL to upload (raw prompt/completion rows)")
    ap.add_argument("--name", default=None, help="dataset name shown in Adaption")
    ap.add_argument("--model", default=None, help="model id for recommend_hyperparams (omit to skip)")
    ap.add_argument("--skip-estimate", action="store_true")
    ap.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="where to dump upload/estimate JSON evidence (default: file's parent dir)",
    )
    ap.add_argument(
        "--dataset-id",
        default=None,
        help="configure+verify the column mapping on an existing dataset (skips the upload)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=PLAN_ROW_LIMIT,
        help=f"cap the rows priced when persisting the mapping (0 = no cap; plan limit {PLAN_ROW_LIMIT:,})",
    )
    args = ap.parse_args(argv)

    if not os.environ.get("ADAPTION_API_KEY"):
        print("FAIL: ADAPTION_API_KEY not set (read it from the root .env)")
        return 1
    if args.dataset_id:
        client = client or _client()
        return 0 if ensure_column_mapping(client, args.dataset_id, max_rows=args.max_rows or None) else 1
    if args.file is None or args.name is None:
        print("FAIL: --file and --name are required unless --dataset-id is given")
        return 1
    if not args.file.exists():
        print(f"FAIL: {args.file} not found")
        return 1

    evidence_dir = args.evidence_dir or args.file.parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return _upload(client or _client(), args, evidence_dir)


def _client() -> Any:
    from adaption import Adaption

    return Adaption()


if __name__ == "__main__":
    raise SystemExit(main())
