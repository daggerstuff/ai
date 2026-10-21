#!/usr/bin/env python3
"""Summarize paired W&B v5 held-out inference runs without exposing responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MIN_BOOTSTRAP_SIZE = 2
logger = logging.getLogger(__name__)


def _load_results(directory: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    path = directory / "results.jsonl"
    if not path.is_file():
        return results
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "success" and isinstance(row.get("case_id"), str):
                results[row["case_id"]] = row
    return results


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f1(response: str, reference: str) -> float:
    predicted = Counter(TOKEN_RE.findall(response.casefold()))
    expected = Counter(TOKEN_RE.findall(reference.casefold()))
    overlap = sum((predicted & expected).values())
    precision = overlap / max(1, sum(predicted.values()))
    recall = overlap / max(1, sum(expected.values()))
    return 2 * precision * recall / max(1e-12, precision + recall)


def _bootstrap_interval(values: list[float], seed: int, replicates: int) -> list[float] | None:
    if len(values) < MIN_BOOTSTRAP_SIZE:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(replicates))
    return [means[int(replicates * 0.025)], means[min(replicates - 1, int(replicates * 0.975))]]


def _arm_metrics(rows: dict[str, dict[str, Any]], ids: list[str], refs: dict[str, str]) -> dict[str, float]:
    return {
        "mean_reference_unigram_f1": statistics.mean(
            _f1(rows[case_id].get("response", ""), refs[case_id]) for case_id in ids
        ),
        "mean_completion_tokens": statistics.mean(
            (rows[case_id].get("usage") or {}).get("completion_tokens", 0) for case_id in ids
        ),
        "truncation_rate": statistics.mean(rows[case_id].get("finish_reason") == "length" for case_id in ids),
        "mean_latency_ms": statistics.mean(rows[case_id].get("latency_ms", 0) for case_id in ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="results directory relative to root; can be repeated",
    )
    args = parser.parse_args()
    references = {
        row["id"]: row["target"]
        for row in (json.loads(line) for line in args.references.open(encoding="utf-8"))
        if isinstance(row.get("id"), str) and isinstance(row.get("target"), str)
    }
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, list[dict[str, str]]] = {}
    for spec in args.arm:
        name, relative = spec.split("=", 1)
        directory = args.root / relative
        arms.setdefault(name, {}).update(_load_results(directory))
        result_file = directory / "results.jsonl"
        hashes.setdefault(name, []).append(
            {
                "path": relative,
                "sha256": _file_sha256(result_file) if result_file.is_file() else "missing",
            }
        )

    comparisons = []
    names = list(arms)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = arms[left_name], arms[right_name]
            ids = sorted((left.keys() & right.keys()) & references.keys())
            if not ids:
                continue
            deltas = [
                _f1(right[case_id].get("response", ""), references[case_id])
                - _f1(left[case_id].get("response", ""), references[case_id])
                for case_id in ids
            ]
            comparisons.append(
                {
                    "left_arm": left_name,
                    "right_arm": right_name,
                    "paired_cases": len(ids),
                    "right_minus_left_reference_unigram_f1": statistics.mean(deltas),
                    "paired_bootstrap_95pct_ci": _bootstrap_interval(deltas, args.seed, args.bootstrap_replicates),
                    "left_metrics": _arm_metrics(left, ids, references),
                    "right_metrics": _arm_metrics(right, ids, references),
                }
            )
    payload = {
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "result_file_sha256": hashes,
        "successful_cases_per_arm": {name: len(rows) for name, rows in arms.items()},
        "comparisons": comparisons,
        "method": (
            "Case-normalized multiset unigram F1 against the held-out assistant target; paired by case ID. "
            "Bootstrap resamples paired cases with replacement. Lexical overlap is not semantic or clinical scoring."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    logger.info(
        "%s",
        json.dumps(
            {
                "output": str(args.output),
                "successful_cases_per_arm": payload["successful_cases_per_arm"],
                "comparisons": [
                    {
                        "arms": [item["left_arm"], item["right_arm"]],
                        "paired_cases": item["paired_cases"],
                        "delta": item["right_minus_left_reference_unigram_f1"],
                        "ci": item["paired_bootstrap_95pct_ci"],
                    }
                    for item in comparisons
                ],
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
