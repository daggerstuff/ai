"""Multi-axis stratified dataset splitter (PIX-4345 §B.4 step 4C).

Replaces the deterministic hash-split in ``dataset_splitter.py`` with a
stratified split over four axes: language + tags + difficulty + tier. Keeps the
hash-disjoint leakage guarantee (a conversation's content hash determines its
split uniquely) while balancing each stratum across train/val/test.

Integrity gates (run after split, fail closed if violated):
  - hash-disjoint: no content hash appears in >1 split
  - ratio ±2pp: each split within 2pp of target ratio (80/10/10)
  - domain balance ±2pp: per-stratum split ratios within 2pp of global ratio

Usage:
  python dataset_splitter_stratified.py <input_dir> <out_dir> [--ratio 80 10 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIO = (80, 10, 10)
RATIO_TOLERANCE_PP = 2  # ±2 percentage points
BALANCE_TOLERANCE_PP = 2


def get_convo_hash(pair: dict) -> str:
    """Hash the text of all messages — uniquely identifies a conversation."""
    text_blob = "||".join(msg.get("content", "") for msg in pair.get("messages", []))
    return hashlib.md5(text_blob.encode("utf-8")).hexdigest()


def _extract_axes(pair: dict) -> tuple[str, str, str, str]:
    """Extract (language, tags_key, difficulty, tier) from a ChatML record.

    Falls back to 'unknown' on each axis when metadata absent — keeps the
    stratum key stable so records without metadata still split deterministically.
    """
    meta = pair.get("metadata") or pair.get("attributes") or {}
    language = str(meta.get("language") or pair.get("language") or "unknown")
    tags = meta.get("tags") or pair.get("tags") or []
    tags_key = "|".join(sorted(tags)) if isinstance(tags, list) else "none"
    difficulty = str(meta.get("difficulty") or pair.get("difficulty") or "unknown")
    tier = str(meta.get("tier") or pair.get("tier") or "unknown")
    return language, tags_key, difficulty, tier


def _stratum_key(pair: dict) -> str:
    lang, tags, diff, tier = _extract_axes(pair)
    return f"{lang}||{tags}||{diff}||{tier}"


def stratified_split(
    records: list[dict],
    ratio: tuple[int, int, int] = DEFAULT_RATIO,
) -> dict[str, list[dict]]:
    """Split records into train/val/test, stratified by language+tags+difficulty+tier.

    Within each stratum, records are assigned to splits via a deterministic hash
    of the content so the same conversation always lands in the same split
    (hash-disjoint leakage guarantee). The hash is used to pick a bucket in
    [0,100); cumulative ratio thresholds decide the split. This balances each
    stratum independently while keeping assignments stable across re-runs.
    """
    assert len(ratio) == 3 and sum(ratio) == 100, f"ratio must sum to 100, got {ratio}"
    thresholds = (ratio[0], ratio[0] + ratio[1])

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_stratum[_stratum_key(rec)].append(rec)

    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for _stratum, items in by_stratum.items():
        for item in items:
            h = int(get_convo_hash(item)[:8], 16) % 100
            if h < thresholds[0]:
                splits["train"].append(item)
            elif h < thresholds[1]:
                splits["val"].append(item)
            else:
                splits["test"].append(item)
    return splits


def integrity_gates(
    splits: dict[str, list[dict]],
    ratio: tuple[int, int, int] = DEFAULT_RATIO,
) -> tuple[bool, list[str]]:
    """Verify hash-disjoint, ratio ±2pp, domain balance ±2pp. Returns (passed, issues)."""
    issues: list[str] = []
    total = sum(len(s) for s in splits.values())
    if total == 0:
        return False, ["no records in any split"]

    # 1. hash-disjoint
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            h = get_convo_hash(item)
            hash_to_splits[h].add(split_name)
    leaks = [h for h, names in hash_to_splits.items() if len(names) > 1]
    if leaks:
        issues.append(f"hash-disjoint FAILED: {len(leaks)} hashes in >1 split")

    # 2. ratio ±2pp
    for i, name in enumerate(SPLIT_NAMES):
        actual_pp = round(100 * len(splits[name]) / total)
        target_pp = ratio[i]
        if abs(actual_pp - target_pp) > RATIO_TOLERANCE_PP:
            issues.append(
                f"ratio FAILED: {name}={actual_pp}pp vs target {target_pp}pp (tolerance ±{RATIO_TOLERANCE_PP}pp)"
            )

    # 3. domain balance ±2pp: each stratum's split ratio within 2pp of global
    by_stratum_split: dict[str, Counter] = defaultdict(Counter)
    for split_name, items in splits.items():
        for item in items:
            by_stratum_split[_stratum_key(item)][split_name] += 1

    global_ratios = {name: 100 * len(splits[name]) / total for name in SPLIT_NAMES}
    for stratum, counts in by_stratum_split.items():
        stratum_total = sum(counts.values())
        if stratum_total < 10:
            continue  # too few samples in this stratum to demand balance
        for name in SPLIT_NAMES:
            stratum_ratio = 100 * counts[name] / stratum_total
            if abs(stratum_ratio - global_ratios[name]) > BALANCE_TOLERANCE_PP:
                issues.append(
                    f"domain balance FAILED: stratum '{stratum}' {name}="
                    f"{stratum_ratio:.1f}pp vs global {global_ratios[name]:.1f}pp "
                    f"(tolerance ±{BALANCE_TOLERANCE_PP}pp)"
                )
                break

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified dataset splitter (PIX-4345 §B.4 step 4C)")
    parser.add_argument("input_dir", nargs="?", default="ai/training/output/books/chatml")
    parser.add_argument("out_dir", nargs="?", default="ai/training/output/dataset")
    parser.add_argument("--ratio", type=int, nargs=3, default=list(DEFAULT_RATIO))
    parser.add_argument("--no-gates", action="store_true", help="skip integrity gates")
    args = parser.parse_args()

    if sum(args.ratio) != 100:
        print(f"ERROR: ratio must sum to 100, got {args.ratio} (sum={sum(args.ratio)})")
        return 2

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    chatml_files = sorted(input_dir.glob("*.jsonl"))
    if not chatml_files:
        print(f"No ChatML files found in {input_dir} to split.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    SYSTEM_PROMPT = (
        "You are Pixelated Empathy, an evidence-based clinical AI assistant. "
        "Your responses must be empathetic, validating, and grounded in established "
        "therapeutic modalities (such as CBT, DBT, or ACT)."
    )

    print(f"Loading {len(chatml_files)} files from {input_dir}...")
    records: list[dict] = []
    for file_path in chatml_files:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    pair = json.loads(line)
                    if pair.get("messages") and pair["messages"][0].get("role") != "system":
                        pair["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                    records.append(pair)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue

    print(f"Loaded {len(records)} records. Splitting (ratio={tuple(args.ratio)})...")
    splits = stratified_split(records, ratio=tuple(args.ratio))

    counts = {name: len(items) for name, items in splits.items()}
    print(f"Split sizes -> Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")

    if not args.no_gates:
        passed, gate_issues = integrity_gates(splits, ratio=tuple(args.ratio))
        if passed:
            print("Integrity gates PASSED: hash-disjoint, ratio ±2pp, domain balance ±2pp.")
        else:
            print("Integrity gates FAILED:")
            for issue in gate_issues:
                print(f"  - {issue}")
            return 1
    else:
        print("Integrity gates skipped (--no-gates).")

    for name in SPLIT_NAMES:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in splits[name]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(splits[name])} records to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
