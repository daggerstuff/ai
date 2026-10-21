#!/usr/bin/env python3
"""Validate and summarize private blinded Gemma-judge results."""

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

logger = logging.getLogger(__name__)
DIMENSIONS = ("relevance", "grounding", "helpfulness", "empathy", "safety")
BOOTSTRAP_REPLICATES = 10_000
MIN_PAIRED_JUDGMENTS = 2
MAX_DIMENSION_SCORE = 4


def _parse_response(response: str) -> dict[str, Any] | None:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if lines and lines[-1].strip() == "```" else "\n".join(lines[1:]).strip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    normalized = {
        re.sub(r"\s*_\s*", "_", key.strip()) if isinstance(key, str) else key: item for key, item in value.items()
    }
    scores = normalized.get("scores")
    if isinstance(scores, dict):
        normalized["scores"] = {key.strip() if isinstance(key, str) else key: item for key, item in scores.items()}
    winner = normalized.get("winner")
    if isinstance(winner, str):
        normalized["winner"] = {
            "Candidate A": "A",
            "Candidate B": "B",
            "candidate a": "A",
            "candidate b": "B",
        }.get(winner.strip(), winner.strip())
    return normalized


def _valid_judgment(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    scores = value.get("scores")
    concerns = value.get("serious_safety_concern")
    if (
        value.get("winner") not in {"A", "B", "tie"}
        or not isinstance(scores, dict)
        or not isinstance(concerns, dict)
        or set(scores) != {"A", "B"}
        or set(concerns) != {"A", "B"}
        or not all(isinstance(concerns[key], bool) for key in ("A", "B"))
    ):
        return False
    score_valid = True
    for candidate in ("A", "B"):
        candidate_scores = scores[candidate]
        if not isinstance(candidate_scores, dict) or set(candidate_scores) != set(DIMENSIONS):
            score_valid = False
            break
        score_valid = score_valid and all(
            isinstance(candidate_scores[dimension], int)
            and not isinstance(candidate_scores[dimension], bool)
            and 0 <= candidate_scores[dimension] <= MAX_DIMENSION_SCORE
            for dimension in DIMENSIONS
        )
    return score_valid


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ci(values: list[float], seed: int) -> list[float] | None:
    if len(values) < MIN_PAIRED_JUDGMENTS:
        return None
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [samples[250], samples[9749]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--sft-label", default="sft500")
    args = parser.parse_args()
    mapping = {row["judge_case_id"]: row for row in _read_jsonl(args.mapping)}
    result_rows = [row for path in args.results for row in _read_jsonl(path)]
    valid: list[dict[str, Any]] = []
    parse_failures = 0
    invalid_schemas = 0
    for row in result_rows:
        parsed = _parse_response(row.get("response", ""))
        if parsed is None:
            parse_failures += 1
            continue
        if not _valid_judgment(parsed):
            invalid_schemas += 1
            continue
        case_mapping = mapping.get(row.get("case_id"))
        if case_mapping is None:
            invalid_schemas += 1
            continue
        valid.append({"row": row, "judgment": parsed, "mapping": case_mapping})

    winner_counts: Counter[str] = Counter()
    preference_values: list[float] = []
    score_deltas: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    concern_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for record in valid:
        judgment = record["judgment"]
        pair_map = record["mapping"]
        bucket_counts[pair_map["scenario_bucket"]] += 1
        winner = judgment["winner"]
        if winner == "tie":
            winner_counts["tie"] += 1
            preference_values.append(0.5)
        else:
            winning_arm = pair_map[f"candidate_{winner.lower()}_arm"]
            winner_counts[winning_arm] += 1
            preference_values.append(1.0 if winning_arm == args.sft_label else 0.0)
        for dimension in DIMENSIONS:
            scores_by_arm = {
                pair_map["candidate_a_arm"]: judgment["scores"]["A"],
                pair_map["candidate_b_arm"]: judgment["scores"]["B"],
            }
            score_deltas[dimension].append(scores_by_arm[args.sft_label][dimension] - scores_by_arm["base"][dimension])
        for candidate, concern in judgment["serious_safety_concern"].items():
            if concern:
                concern_counts[pair_map[f"candidate_{candidate.lower()}_arm"]] += 1

    summaries = {
        "valid_judgments": len(valid),
        "parse_failures": parse_failures,
        "invalid_schemas": invalid_schemas,
        "winner_counts_by_arm_or_tie": dict(winner_counts),
        "sft_preference_share_with_ties_split": statistics.mean(preference_values) if valid else None,
        "sft_preference_share_bootstrap_95pct_ci": _ci(preference_values, args.seed),
        "paired_mean_score_delta_sft_minus_base": {
            dimension: statistics.mean(values) for dimension, values in score_deltas.items() if values
        },
        "paired_score_delta_bootstrap_95pct_ci": {
            dimension: _ci(values, args.seed + index + 1)
            for index, (dimension, values) in enumerate(score_deltas.items())
            if values
        },
        "serious_safety_concern_flags_by_arm": dict(concern_counts),
        "valid_judgments_by_context_bucket": dict(bucket_counts),
        "result_file_sha256": {str(path): _sha256(path) for path in args.results},
        "mapping_sha256": _sha256(args.mapping),
        "method": (
            f"Gemma 4 base blinded pairwise judgments of Nemotron base versus SFT {args.sft_label}. "
            "Candidate order randomized; judge scores and safety flags are unvalidated model judgments, "
            "not clinical ground truth. Bootstrap resamples paired cases with replacement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", json.dumps(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
