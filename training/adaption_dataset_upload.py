"""Upload a training-ready JSONL to Adaption; request the free recipe estimate.

Mirrors the run-1 Dataset-B flow (pilot log §13.3): ``upload.initiate`` ->
presigned S3 PUT -> ``upload.complete`` -> poll ``get_status`` to
``succeeded``; upload + processing are free. Optionally calls
``autoscientist.recommend_hyperparams`` with NO augmentation parameters —
at >=10,000 effective rows the estimate validates corpus-only, which is the
run-2 anti-contamination discipline (the run-1 augmentation pulled 3,043
generic health rows, §13.7). Nothing is launched; training runs are a
separate, owner-gated decision.

val/test splits are never uploaded (held-out discipline).

Run (from ai/, SDK is an ephemeral dependency):
  PYTHONPATH=.. uv run --with adaption python -m training.adaption_dataset_upload \
      --file training/output/arc_corpus/export/adaption_run2_v1/adaption_train_unrolled.jsonl \
      --name arc-run2-train-v1 --model Qwen/Qwen3.6-35B-A3B
"""

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from adaption import Adaption


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, required=True, help="JSONL to upload (raw prompt/completion rows)")
    ap.add_argument("--name", required=True, help="dataset name shown in Adaption")
    ap.add_argument("--model", default=None, help="model id for recommend_hyperparams (omit to skip)")
    ap.add_argument("--skip-estimate", action="store_true")
    ap.add_argument("--evidence-dir", type=Path, default=None,
                    help="where to dump upload/estimate JSON evidence (default: file's parent dir)")
    args = ap.parse_args()

    if not os.environ.get("ADAPTION_API_KEY"):
        print("FAIL: ADAPTION_API_KEY not set (read it from the root .env)")
        return 1
    if not args.file.exists():
        print(f"FAIL: {args.file} not found")
        return 1

    evidence_dir = args.evidence_dir or args.file.parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    client = Adaption()
    size = args.file.stat().st_size
    print(f"initiating upload: {args.file.name} ({size:,} bytes)")

    init = client.datasets.upload.initiate(name=args.name, file_format="jsonl")
    s3_key = urlsplit(init.upload_url).path.lstrip("/")
    print(f"presigned PUT acquired; s3_key={s3_key[:40]}...")

    with args.file.open("rb") as f, httpx.Client(timeout=1800.0) as http:
        resp = http.put(init.upload_url, content=f)
    if resp.status_code != 200:
        print(f"FAIL: S3 PUT returned {resp.status_code}")
        return 1
    print("S3 PUT complete")

    done = client.datasets.upload.complete(
        s3_key=s3_key, name=args.name, file_format="jsonl", file_size_bytes=size,
    )
    dataset_id = done.dataset_id
    print(f"dataset_id={dataset_id}")
    with (evidence_dir / "adaption_upload_run2.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_id": dataset_id, "name": args.name, "file": str(args.file),
                   "file_size_bytes": size, "s3_key": s3_key}, f, indent=1)

    print("polling processing status (up to 30 min)...")
    status = None
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
    print(f"dataset READY: {dataset_id}")

    if args.skip_estimate or not args.model:
        print("estimate skipped")
        return 0

    # no augmentation parameters: corpus-only effective rows (anti-contamination)
    rec = client.autoscientist.recommend_hyperparams(dataset_id=dataset_id, model=args.model)
    payload = rec.model_dump()
    with (evidence_dir / "adaption_recommend_hyperparams_run2.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(json.dumps(payload, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
