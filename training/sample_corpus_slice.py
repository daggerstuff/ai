#!/usr/bin/env python3
"""Deterministic corpus slice sampler (PIX-4584 representative-run support).

The full 258k-row curation regen is deferred (user decision); validation runs
use a representative slice instead. This helper draws a seeded, deterministic
slice from one or more real corpus JSONL files so the representative run can
be reproduced bit-for-bit and spans multiple source families.

Usage:
    python -m training.sample_corpus_slice \
        --output /tmp/slice.jsonl \
        --inputs path/to/corpusA.jsonl:4000 path/to/corpusB.jsonl:4000 \
        [--seed 42] [--max-messages 5]

Each ``--inputs`` entry is ``path:count`` (count defaults to 1000). Rows are
selected with a seeded shuffle per input, so the same command always yields
the same slice. ``--max-messages N`` keeps only rows whose ``messages`` array
has at most N entries (0 disables the filter) — used to keep the slice's
conversation-length strata homogeneous so the small-stratum integrity gates
remain statistically meaningful at slice scale.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def _read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line for line in f if line.strip()]

def sample_slice(
    specs: list[tuple[Path, int]], seed: int, max_messages: int = 0
) -> list[dict[str, Any]]:
    """Read each input, seeded-shuffle it, and take the requested count."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for path, count in specs:
        lines = _read_lines(path)
        rng.shuffle(lines)
        taken = 0
        for line in lines:
            if taken >= count:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if max_messages and len(record.get("messages", [])) > max_messages:
                continue
            records.append(record)
            taken += 1
        print(f"  {path}: sampled {taken}/{len(lines)} rows", file=sys.stderr)
    return records

def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic corpus slice sampler")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input specs as path:count (count defaults to 1000)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL slice path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Keep only rows with at most this many messages (0 = no filter)",
    )
    args = parser.parse_args()

    specs: list[tuple[Path, int]] = []
    for spec in args.inputs:
        path_str, _, count_str = spec.rpartition(":")
        if count_str.isdigit():
            specs.append((Path(path_str), int(count_str)))
        else:
            specs.append((Path(spec), 1000))

    records = sample_slice(specs, args.seed, args.max_messages)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows to {out_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
