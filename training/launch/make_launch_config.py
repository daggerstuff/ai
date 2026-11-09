#!/usr/bin/env python3
"""Build the W&B Serverless SFT launch config for a release from its manifest.

The launcher consumes a JSON file holding the manifest path, the train payload
identity, and the training config. That file used to be assembled by hand for
every release, which made the learning-rate schedule an unreviewable artifact:
the numbers were recovered once (v14) and then copied forward without a way to
check they still matched the recipe.

This script derives the whole thing from the release manifest, so the schedule
is a function of the train row count rather than a transcription. ``--verify``
rebuilds an existing config and fails when the result differs, which is how the
recipe was pinned down in the first place.

Recipe (recovered from the v13/v14/v15 configs, which agree exactly):

* ``total_batch = ceil(train_rows / batch_size)`` steps, one LR entry per step.
* Warmup over the first 5% of steps, linear from 1e-5 to the 5e-5 peak.
  The warmup lands on the peak exactly, so the last warmup step is step
  ``int(n * 0.05) - 1`` and carries the peak value.
* Hold the peak until 60% of the schedule (``int(n * 0.6)``).
* Cosine decay from the peak to 5e-6 over the remaining steps, ending on 5e-6.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

BATCH_SIZE = 16
START_LR = 1e-5
PEAK_LR = 5e-5
END_LR = 5e-6
WARMUP_FRACTION = 0.05
HOLD_FRACTION = 0.6
RECIPE_SOURCE = "v5 trapezoid: warmup 5% 1e-5->5e-5, hold 5e-5 to 60%, cosine decay to 5e-6"
MIN_SCHEDULE_STEPS = 3

CONFIG_KEYS = (
    "manifest",
    "release_id",
    "train_payload",
    "train_rows",
    "train_sha256",
    "total_batch",
    "recipe_source",
    "config",
)


def total_batch(train_rows: int, batch_size: int = BATCH_SIZE) -> int:
    """Return the number of optimiser steps the provider will run."""
    if train_rows <= 0:
        raise ValueError("train row count must be positive")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return math.ceil(train_rows / batch_size)


def warmup_steps(steps: int) -> int:
    """Return the index of the first step that carries the peak learning rate."""
    return int(steps * WARMUP_FRACTION) - 1


def hold_end(steps: int) -> int:
    """Return the index of the last step that carries the peak learning rate."""
    return int(steps * HOLD_FRACTION)


def learning_rate_schedule(steps: int) -> list[float]:
    """Return one learning rate per optimiser step."""
    if steps < MIN_SCHEDULE_STEPS:
        raise ValueError("schedule needs at least three steps")
    warm = warmup_steps(steps)
    hold = hold_end(steps)
    if warm < 1 or hold <= warm or hold >= steps - 1:
        raise ValueError(f"degenerate schedule boundaries for {steps} steps (warm={warm}, hold={hold})")
    schedule: list[float] = []
    for step in range(steps):
        if step < warm:
            schedule.append(START_LR + (PEAK_LR - START_LR) * step / warm)
        elif step <= hold:
            schedule.append(PEAK_LR)
        else:
            progress = (step - hold) / (steps - 1 - hold)
            schedule.append(END_LR + (PEAK_LR - END_LR) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return schedule


def build_config(manifest_path: Path, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Return the launch config for the release described by ``manifest_path``."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = manifest["partitions"]["train"]
    rows = train.get("row_count", train.get("rows"))
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("train partition is missing a usable row count")
    payload_path = train.get("path", train.get("file"))
    digest = train.get("sha256", train.get("file_sha256"))
    if not isinstance(payload_path, str) or not isinstance(digest, str):
        raise ValueError("train partition is missing a payload path or SHA-256")
    steps = total_batch(rows, batch_size)
    config: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "release_id": manifest["release_id"],
        "train_payload": str((REPO_ROOT / payload_path).resolve()),
        "train_rows": rows,
        "train_sha256": digest,
        "total_batch": steps,
        "recipe_source": RECIPE_SOURCE,
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate_schedule(steps),
            "assistant_turns": "all",
            "metric_logging": {"enabled": True, "target_training_step": 50},
        },
    }
    return {key: config[key] for key in CONFIG_KEYS}


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return human-readable differences between two launch configs."""
    diffs: list[str] = []
    for key in CONFIG_KEYS:
        if key not in actual:
            diffs.append(f"{key}: missing from existing config")
        elif key not in expected:
            diffs.append(f"{key}: present only in existing config")
        elif expected[key] != actual[key]:
            diffs.append(f"{key}: rebuilt differs from existing")
    extra = sorted(set(actual) - set(CONFIG_KEYS))
    if extra:
        diffs.append(f"unexpected keys in existing config: {', '.join(extra)}")
    return diffs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="release manifest JSON")
    parser.add_argument("--out", type=Path, help="where to write the launch config (default: next to the manifest)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--verify", type=Path, help="compare against an existing launch config instead of writing")
    parser.add_argument("--force", action="store_true", help="overwrite an existing launch config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = build_config(args.manifest, args.batch_size)
    if args.verify is not None:
        existing = json.loads(args.verify.read_text(encoding="utf-8"))
        diffs = _compare(config, existing)
        if diffs:
            sys.stderr.write(json.dumps({"verified": False, "differences": diffs}, indent=2) + "\n")
            return 1
        sys.stdout.write(
            json.dumps({"verified": True, "config": str(args.verify), "total_batch": config["total_batch"]}, indent=2)
            + "\n"
        )
        return 0
    out = args.out or args.manifest.parent / f"{_release_tag(config['release_id'])}-launch-config.json"
    if out.exists() and not args.force:
        sys.stderr.write(f"refusing to overwrite {out} (pass --force)\n")
        return 2
    out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "written": str(out),
                "release_id": config["release_id"],
                "train_rows": config["train_rows"],
                "total_batch": config["total_batch"],
                "learning_rate": {
                    "first": config["config"]["learning_rate"][0],
                    "peak": max(config["config"]["learning_rate"]),
                    "last": config["config"]["learning_rate"][-1],
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _release_tag(release_id: str) -> str:
    """Return the short release tag (``v16``) used in launch config filenames."""
    tag = release_id.rsplit("-", 1)[-1]
    if not tag.startswith("v"):
        raise ValueError(f"cannot derive a release tag from {release_id!r}")
    return tag


if __name__ == "__main__":
    raise SystemExit(main())
