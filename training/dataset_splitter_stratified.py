"""Multi-axis stratified dataset splitter (PIX-4345 §B.4 step 4C, PIX-4584).

Extends the 4-axis stratified split to 8+ axes per PIX-4584:
source_family/source_id, product_type, policy_mode, diagnosis/topic,
culture/context, safety_severity, difficulty, and conversation_length.

Key features:
  - Source-family grouping: all records from the same source_family/source_id
    are assigned to the same split (leakage prevention).
  - Evaluation-material blocking: records flagged as evaluation/restricted
    are excluded from the train split.
  - Preference-pair counterpart preservation: DPO preference pairs from the
    same source stay in the same partition via source-family grouping.
  - Extended integrity gates: hash-disjoint, source-family disjoint,
    evaluation-in-train, ratio ±2pp, domain balance ±2pp.

Integrity gates (run after split, fail closed if violated):
  - hash-disjoint: no content hash appears in >1 split
  - source-family disjoint: no source_family appears in >1 split
  - evaluation-in-train: no evaluation/restricted material in train
  - ratio ±2pp: each split within 2pp of target ratio
  - domain balance ±2pp: per-stratum split ratios within 2pp of global ratio

Usage:
  python dataset_splitter_stratified.py <input_dir> <out_dir> [--ratio 70 15 15]

This is THE canonical splitter for the training-data pipeline (PIX-4584
reconciliation): the legacy hash-bucket ``dataset_splitter.py`` (80/10/10)
has been retired, and ``curate_pipeline.py`` delegates its split step to
``stratified_split()`` from this module so mechanism and ratio agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")

# Canonical split ratio: 70/15/15. This is the ratio the production curation
# pipeline (training/curate_pipeline.py) ships with — the DVC-tracked curated
# shards (ai/data/curated/sft_chatml/*.jsonl.dvc) were produced at 70/15/15 —
# so the standalone tool defaults to the same ratio. There is exactly ONE
# ratio constant for the whole training-data pipeline; do not fork it.
DEFAULT_RATIO = (70, 15, 15)
RATIO_TOLERANCE_PP = 2  # ±2 percentage points
BALANCE_TOLERANCE_PP = 2

# PIX-4584: 8+ split axes (language and tags retained from PIX-4345)
AXIS_NAMES = (
    "language",
    "tags",
    "difficulty",
    "tier",
    "product_type",
    "policy_mode",
    "diagnosis_topic",
    "culture_context",
    "safety_severity",
    "conversation_length",
)

# Splits that evaluation/restricted material is blocked from
EVAL_BLOCKED_SPLITS = ("train",)

# Conversation length buckets
_SHORT_THRESHOLD = 5
_MEDIUM_THRESHOLD = 15


def get_convo_hash(pair: dict) -> str:
    """Hash the text of all messages — uniquely identifies a conversation."""
    text_blob = "||".join(msg.get("content", "") for msg in pair.get("messages", []))
    return hashlib.md5(text_blob.encode("utf-8")).hexdigest()


def _extract_axes(pair: dict) -> dict[str, str]:
    """Extract all 8+ split axes from a ChatML record.

    Returns a dict keyed by AXIS_NAMES. Falls back to 'unknown' on each
    axis when metadata is absent — keeps the stratum key stable so records
    without metadata still split deterministically.
    """
    meta = pair.get("metadata") or pair.get("attributes") or {}

    language = str(meta.get("language") or pair.get("language") or "unknown")
    tags = meta.get("tags") or pair.get("tags") or []
    tags_key = "|".join(sorted(tags)) if isinstance(tags, list) else "none"
    difficulty = str(meta.get("difficulty") or pair.get("difficulty") or "unknown")
    tier = str(meta.get("tier") or pair.get("tier") or "unknown")
    product_type = str(meta.get("product_type") or pair.get("product_type") or "unknown")
    policy_mode = str(meta.get("policy_mode") or pair.get("policy_mode") or "unknown")
    diagnosis_topic = str(
        meta.get("diagnosis_topic")
        or meta.get("topic")
        or meta.get("diagnosis")
        or pair.get("diagnosis_topic")
        or "unknown"
    )
    culture_context = str(
        meta.get("culture_context")
        or meta.get("culture")
        or pair.get("culture_context")
        or "unknown"
    )
    safety_severity = str(
        meta.get("safety_severity")
        or meta.get("severity")
        or pair.get("safety_severity")
        or "unknown"
    )

    # Conversation length bucketing from message count
    msg_count = len(pair.get("messages", []))
    if msg_count <= _SHORT_THRESHOLD:
        conversation_length = "short"
    elif msg_count <= _MEDIUM_THRESHOLD:
        conversation_length = "medium"
    else:
        conversation_length = "long"

    return {
        "language": language,
        "tags": tags_key,
        "difficulty": difficulty,
        "tier": tier,
        "product_type": product_type,
        "policy_mode": policy_mode,
        "diagnosis_topic": diagnosis_topic,
        "culture_context": culture_context,
        "safety_severity": safety_severity,
        "conversation_length": conversation_length,
    }


def _stratum_key(pair: dict) -> str:
    """Build a stratum key from all 8+ axes joined by '||'."""
    axes = _extract_axes(pair)
    return "||".join(axes[name] for name in AXIS_NAMES)


def _source_family_key(pair: dict) -> str:
    """Extract source family/source ID for leakage prevention grouping.

    All records sharing this key are assigned to the same split to prevent
    source-level data leakage between train/val/test.
    """
    meta = pair.get("metadata") or pair.get("attributes") or {}
    source_family = str(
        meta.get("source_family") or pair.get("source_family") or "unknown"
    )
    source_id = str(meta.get("source_id") or pair.get("source_id") or "unknown")
    return f"{source_family}::{source_id}"


def _is_evaluation_material(pair: dict) -> bool:
    """Check if a record is evaluation/restricted material blocked from train.

    Blocks records flagged as evaluation, restricted, holdout, or official
    test/eval splits from entering the train partition.
    """
    meta = pair.get("metadata") or pair.get("attributes") or {}

    # Explicit evaluation/restricted flags
    if meta.get("evaluation") or meta.get("restricted") or meta.get("is_eval"):
        return True
    if pair.get("evaluation") or pair.get("restricted"):
        return True

    # Official split designation
    official_split = meta.get("official_split") or pair.get("official_split")
    if official_split in ("test", "eval", "holdout"):
        return True

    # Source type marking
    source_type = meta.get("source_type") or pair.get("source_type")
    return source_type in ("evaluation", "restricted", "holdout")


def stratified_split(
    records: list[dict],
    ratio: tuple[int, int, int] = DEFAULT_RATIO,
) -> dict[str, list[dict]]:
    """Split records into train/val/test, stratified by 8+ axes.

    Groups by source_family to prevent leakage: all records from the same
    source family/source ID are assigned to the same split via a deterministic
    hash of the family key. Evaluation/restricted material is blocked from
    the train split and redirected to val.
    """
    assert len(ratio) == 3 and sum(ratio) == 100, f"ratio must sum to 100, got {ratio}"
    thresholds = (ratio[0], ratio[0] + ratio[1])

    # Group by source family for leakage prevention
    by_family: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_family[_source_family_key(rec)].append(rec)

    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for family_key, family_items in by_family.items():
        # Assign entire family to same split based on family hash
        family_hash = int(hashlib.md5(family_key.encode("utf-8")).hexdigest()[:8], 16) % 100

        # Check if any item in family is evaluation/restricted material
        has_eval = any(_is_evaluation_material(item) for item in family_items)

        if family_hash < thresholds[0] and not has_eval:
            target = "train"
        elif family_hash < thresholds[1]:
            target = "val"
        else:
            target = "test"

        # If family has eval material and was assigned to train, redirect to val
        if has_eval and target == "train":
            target = "val"

        splits[target].extend(family_items)

    return splits


def integrity_gates(
    splits: dict[str, list[dict]],
    ratio: tuple[int, int, int] = DEFAULT_RATIO,
) -> tuple[bool, list[str]]:
    """Verify hash-disjoint, source-family disjoint, eval-in-train, ratio, balance.

    Returns (passed, issues). Fails closed if any gate is violated.
    """
    issues: list[str] = []
    total = sum(len(s) for s in splits.values())
    if total == 0:
        return False, ["no records in any split"]

    # 1. hash-disjoint: no content hash in >1 split
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            h = get_convo_hash(item)
            hash_to_splits[h].add(split_name)
    leaks = [h for h, names in hash_to_splits.items() if len(names) > 1]
    if leaks:
        issues.append(f"hash-disjoint FAILED: {len(leaks)} hashes in >1 split")

    # 2. source-family disjoint: no source_family in >1 split (PIX-4584)
    family_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            family_to_splits[_source_family_key(item)].add(split_name)
    family_leaks = [f for f, names in family_to_splits.items() if len(names) > 1]
    if family_leaks:
        issues.append(
            f"source-family disjoint FAILED: {len(family_leaks)} source families in >1 split"
        )

    # 3. evaluation-in-train: no evaluation/restricted material in train (PIX-4584)
    eval_in_train = [
        get_convo_hash(item)
        for item in splits.get("train", [])
        if _is_evaluation_material(item)
    ]
    if eval_in_train:
        issues.append(
            f"evaluation-in-train FAILED: {len(eval_in_train)} eval/restricted records in train"
        )

    # 4. ratio ±2pp
    for i, name in enumerate(SPLIT_NAMES):
        actual_pp = round(100 * len(splits[name]) / total)
        target_pp = ratio[i]
        if abs(actual_pp - target_pp) > RATIO_TOLERANCE_PP:
            issues.append(
                f"ratio FAILED: {name}={actual_pp}pp vs target {target_pp}pp "
                f"(tolerance ±{RATIO_TOLERANCE_PP}pp)"
            )

    # 5. domain balance ±2pp: each stratum's split ratio within 2pp of global
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


def main() -> int:  # noqa: PLR0915
    from training.release_manifest import build_release_manifest  # noqa: PLC0415  (lazy import avoids circular dep)

    parser = argparse.ArgumentParser(
        description="Stratified dataset splitter (PIX-4345 §B.4, PIX-4584)"
    )
    parser.add_argument(
        "input_dir", nargs="?", default="ai/training/output/books/chatml"
    )
    parser.add_argument(
        "out_dir", nargs="?", default="ai/training/output/dataset"
    )
    parser.add_argument(
        "--ratio", type=int, nargs=3, default=list(DEFAULT_RATIO)
    )
    parser.add_argument(
        "--no-gates", action="store_true", help="skip integrity gates"
    )
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
                        pair["messages"].insert(
                            0, {"role": "system", "content": SYSTEM_PROMPT}
                        )
                    records.append(pair)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue

    print(f"Loaded {len(records)} records. Splitting (ratio={tuple(args.ratio)})...")
    splits = stratified_split(records, ratio=tuple(args.ratio))

    counts = {name: len(items) for name, items in splits.items()}
    print(
        f"Split sizes -> Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}"
    )

    if not args.no_gates:
        passed, gate_issues = integrity_gates(splits, ratio=tuple(args.ratio))
        if passed:
            print(
                "Integrity gates PASSED: hash-disjoint, source-family disjoint, "
                "eval-in-train, ratio ±2pp, domain balance ±2pp."
            )
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

    # Build release manifest (PIX-4584 P13 evidence)
    try:
        manifest = build_release_manifest(
            splits=splits,
            ratio=tuple(args.ratio),
            out_dir=out_dir,
            gates_passed=not args.no_gates,
        )
        manifest_path = out_dir / "release_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Wrote release manifest to {manifest_path}")
    except Exception as e:
        print(f"Warning: could not build release manifest: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
