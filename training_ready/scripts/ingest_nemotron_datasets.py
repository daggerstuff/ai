#!/usr/bin/env python3
"""
Nemotron Dataset Ingestion -> S3 (external/nemotron/)

Utility script to mirror NVIDIA Nemotron datasets from Hugging Face
into the canonical S3 structure used by training_ready.

Goals:
- Keep Nemotron data clearly separated from core mental-health datasets.
- Normalize records into a simple JSONL wrapper with explicit provenance.
- Write directly to S3 under: external/nemotron/{dataset}/{subset}/{split}.jsonl

Example (from ai/training_ready/):

  python scripts/ingest_nemotron_datasets.py \\
    --hf-dataset nvidia/Nemotron-RL-knowledge-mcqa \\
    --split train \\
    --max-samples 5000

This will create:
  s3://{bucket}/external/nemotron/nvidia__Nemotron-RL-knowledge-mcqa/default/train.jsonl

You can then reference this path in your curriculum or downstream pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset  # type: ignore[import-untyped]

# Add project root to path
# Script is at: ai/training_ready/scripts/ingest_nemotron_datasets.py
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # scripts -> training_ready -> ai -> project_root
sys.path.insert(0, str(project_root))

from ai.training_ready.utils.s3_dataset_loader import (  # noqa: E402
    S3DatasetLoader,
)


def _iter_samples(
    ds: Dataset | IterableDataset, max_samples: int | None
) -> Iterable[dict[str, Any]]:
    """Yield up to max_samples records from a HF dataset."""
    count = 0
    for record in ds:
        yield record
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def ingest_nemotron_dataset(
    hf_dataset: str,
    subset: str | None,
    split: str,
    bucket: str,
    max_samples: int | None,
) -> str:
    """
    Mirror a Nemotron (or Nemotron-adjacent) HF dataset into S3.

    Returns the S3 path that was written.
    """
    print("📥 Loading Hugging Face dataset")
    print(f"   📚 Dataset: {hf_dataset}")
    if subset:
        print(f"   🔹 Subset: {subset}")
    print(f"   🔸 Split:  {split}")

    if subset:
        ds = load_dataset(hf_dataset, subset, split=split)
    else:
        ds = load_dataset(hf_dataset, split=split)

    # HF datasets can be iterable; treat generically
    total = len(ds) if isinstance(ds, Dataset) else None
    if total is not None:
        print(f"   ✅ Loaded {total} records")

    loader = S3DatasetLoader(bucket=bucket)
    dataset_slug = hf_dataset.replace("/", "__")
    subset_slug = subset.replace("/", "__") if subset else "default"
    key = f"external/nemotron/{dataset_slug}/{subset_slug}/{split}.jsonl"
    s3_path = f"s3://{loader.bucket}/{key}"

    print("\n☁️ Writing wrapped records to S3")
    print(f"   📦 Bucket: {loader.bucket}")
    print(f"   📄 Key:    {key}")

    lines: list[str] = []
    for idx, record in enumerate(_iter_samples(ds, max_samples=max_samples), start=1):
        wrapped = {
            "source": "nemotron",
            "hf_dataset": hf_dataset,
            "subset": subset,
            "split": split,
            "index": idx - 1,
            "record": record,
        }
        lines.append(json.dumps(wrapped, ensure_ascii=False))

        if idx % 1000 == 0:
            print(f"   ... processed {idx} samples", flush=True)

    if not lines:
        raise RuntimeError("No samples were ingested; check dataset name/split/max-samples.")

    body = ("\n".join(lines) + "\n").encode("utf-8")
    loader.s3_client.put_object(Bucket=loader.bucket, Key=key, Body=body)

    print(f"\n✅ Ingestion complete: {s3_path}")
    return s3_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a Nemotron HF dataset into S3 under external/nemotron/."
    )
    parser.add_argument(
        "--hf-dataset",
        required=True,
        help="Hugging Face dataset name (e.g. nvidia/Nemotron-RL-knowledge-mcqa)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional dataset subset/config name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to ingest (default: train)",
    )
    parser.add_argument(
        "--bucket",
        default="pixelated-training-data",
        help="S3 bucket name (default: pixelated-training-data)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to ingest (for experimentation)",
    )

    args = parser.parse_args()

    try:
        ingest_nemotron_dataset(
            hf_dataset=args.hf_dataset,
            subset=args.subset,
            split=args.split,
            bucket=args.bucket,
            max_samples=args.max_samples,
        )
        print(
            "\n💡 Next steps: reference this S3 path from your curriculum as a generic "
            "reasoning/instruction stage, separate from clinical/PHI datasets."
        )
        return 0
    except Exception as exc:
        print(f"\n❌ Nemotron ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
