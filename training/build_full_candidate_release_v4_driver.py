"""Rebuild the full-corpus W&B release from the final 190,244-row gold.

v4 supersedes v3 (built 2026-10-19 from the 190,230-row pre-arc-completion
master): it adds the 14 sessions from arcs 0319/0327/0342/0380/0387/0392.

Monkey-patches build_full_candidate_release module constants (they are read at
call time) and corrects the hardcoded source_candidate row_count in the
manifest. Otherwise runs the stock builder end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai" / "training"))

import build_full_candidate_release as builder

builder.CANDIDATE = ROOT / "ai/data/curated/sft_chatml/train_master_gold.jsonl"
builder.LEDGER = ROOT / "docs/training/execution-2026-10-17/current-master-source-decision-ledger-2026-10-19-v4.jsonl"
builder.CANDIDATE_SHA256 = "f56e10b52ed83645ec3eca1b468724644dcbf6f843da96d7e82b5b4ef47bc349"

_original_create_manifest = builder.create_manifest


def _create_manifest(
    stats: dict[str, Any], output: Path, release_id: str, repairs_path: Path | None
) -> dict[str, Any]:
    manifest = _original_create_manifest(stats, output, release_id, repairs_path)
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be an object")
    hardcoded = manifest["source_candidate"]["row_count"]
    if hardcoded != 190139:
        raise ValueError(f"unexpected hardcoded source_candidate row_count {hardcoded}")
    manifest["source_candidate"]["row_count"] = 190244
    manifest["owner_direction"] = (
        "User directed rebuild of the full-corpus release from the final "
        "190,244-row gold (six completed arcs) for research training on 2026-10-19."
    )
    return manifest


builder.create_manifest = _create_manifest

sys.argv = [
    "build_full_candidate_release.py",
    "--release-id", "wandb-full-corpus-current-master-20261019-v4",
    "--output", "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261019-v4",
    "--manifest", "docs/training/execution-2026-10-19/wandb-full-corpus-current-master-20261019-v4.json",
    "--repairs", "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-repairs-20261019-v3/row_repairs.jsonl",
]

builder.main()
