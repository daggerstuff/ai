"""Multi-axis stratified dataset splitter.

Replaces the hash-bucket split in ``dataset_splitter.py`` and
``curate_pipeline.py:assign_split`` with a true stratified split along four
axes:

* **domain** — therapeutic category (from ``data_audit.CATEGORY_KEYWORDS``)
* **difficulty** — easy / medium / hard (heuristic on message length + turns)
* **language** — en / es / fr / pt / de
* **tier** — T1_GOLD / T2_SILVER / T3_BRONZE / T4_SAFETY

Algorithm (blueprint B.7):

1. Group records by language.
2. Within each language group, multi-label stratify on domain tags using
   ``iterative-stratification`` (``MultilabelStratifiedShuffleSplit``).
3. Rare classes (< ``min_class_samples``) collapsed to ``__OTHER__`` for
   stratification only; original tags are preserved on the record.
4. Split rest into val/test (50/50).
5. Hash-split preserved as deterministic fallback for edge cases.

Integrity gates (B.7.6):

* Hash-disjoint — no record appears in multiple splits.
* Ratio tolerance ±2pp.
* Domain / language marginal proportion ±2pp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("stratified_split")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "pt", "de")
DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
RATIO_TOLERANCE: float = 0.02  # ±2pp
MIN_CLASS_SAMPLES: int = 50
OTHER_MARKER: str = "__OTHER__"

# Language detection keyword maps (mirrors multilingual_safety_checker patterns)
_LANG_KEYWORDS: dict[str, list[str]] = {
    "pt": ["obrigado", "você", "muito", "bom dia", "tudo bem"],
    "es": ["hola", "gracias", "qué", "quiero", "como", "muy"],
    "fr": ["bonjour", "merci", "je", "vous", "très", "avec"],
    "de": ["guten", "danke", "ich", "und", "nicht", "sehr"],
}

# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _message_text(record: dict[str, Any]) -> str:
    """Concatenate all message content into a single lowercase string."""
    parts: list[str] = []
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return " ".join(parts).lower()


def _category_keywords() -> dict[str, list[str]]:
    """Import CATEGORY_KEYWORDS lazily (avoids circular import risk)."""
    try:
        from training.data_audit import CATEGORY_KEYWORDS  # type: ignore[import-untyped]
        return CATEGORY_KEYWORDS  # type: ignore[no-any-return]
    except Exception:
        return {}


def classify_domain(record: dict[str, Any]) -> str:
    """Return the therapeutic domain of *record*.

    Checks (in order): ``record["domain"]``, ``record["category"]``,
    ``record["metadata"]["category"]``, then keyword matching against
    ``CATEGORY_KEYWORDS``.
    """
    # Explicit field
    for key in ("domain", "category"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("domain", "category"):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    # Keyword matching
    text = _message_text(record)
    keywords = _category_keywords()
    if keywords:
        for domain, kws in keywords.items():
            if any(kw in text for kw in kws):
                return domain
    return "uncategorized"


def classify_difficulty(record: dict[str, Any]) -> str:
    """Heuristic difficulty: easy / medium / hard.

    Based on total message count and character length.
    """
    messages = record.get("messages", [])
    n_turns = len(messages) if isinstance(messages, list) else 0
    text = _message_text(record)
    n_chars = len(text)

    if n_turns <= 3 and n_chars < 800:
        return "easy"
    if n_turns >= 8 or n_chars >= 3000:
        return "hard"
    return "medium"


def classify_language(record: dict[str, Any]) -> str:
    """Detect language: checks explicit field, then keyword matching, defaults en."""
    for key in ("language", "lang"):
        val = record.get(key)
        if isinstance(val, str) and val.strip().lower() in SUPPORTED_LANGUAGES:
            return val.strip().lower()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("language", "lang"):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip().lower() in SUPPORTED_LANGUAGES:
                return val.strip().lower()

    text = _message_text(record)
    for lang, keywords in _LANG_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return lang
    return "en"


def classify_tier(record: dict[str, Any]) -> str:
    """Return tier from record or classify via ``curate_pipeline.classify_tier``."""
    # Explicit field
    for key in ("tier", "quality_tier"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("tier", "quality_tier"):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    # Delegate to curate_pipeline if available
    try:
        from training.curate_pipeline import classify_tier as _classify_tier  # type: ignore[import-untyped]
        return _classify_tier(record)  # type: ignore[no-any-return]
    except Exception:
        return "T3_BRONZE"


def extract_features(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return a list of feature dicts with domain, difficulty, language, tier."""
    return [
        {
            "domain": classify_domain(r),
            "difficulty": classify_difficulty(r),
            "language": classify_language(r),
            "tier": classify_tier(r),
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# Hash-split fallback
# ---------------------------------------------------------------------------

def _content_hash(record: dict[str, Any]) -> str:
    """MD5 of concatenated message content (same as dataset_splitter.get_convo_hash)."""
    text_blob = "||".join(
        msg.get("content", "") for msg in record.get("messages", [])
    )
    return hashlib.md5(text_blob.encode("utf-8")).hexdigest()  # noqa: S324


def _hash_split(
    records: list[dict[str, Any]],
    ratios: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic hash-bucket split (fallback / edge-case)."""
    ratios = ratios or DEFAULT_RATIOS
    train_cut = int(ratios["train"] * 100)
    val_cut = train_cut + int(ratios["val"] * 100)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in records:
        h = int(_content_hash(record)[:8], 16) % 100
        if h < train_cut:
            result["train"].append(record)
        elif h < val_cut:
            result["val"].append(record)
        else:
            result["test"].append(record)
    return result


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _collapse_rare(
    values: list[str],
    min_samples: int = MIN_CLASS_SAMPLES,
) -> list[str]:
    """Replace rare values (< min_samples) with ``__OTHER__``."""
    counts = Counter(values)
    return [v if counts[v] >= min_samples else OTHER_MARKER for v in values]


def _try_iterstrat(
    n_samples: int,
    tag_matrix: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt ``MultilabelStratifiedShuffleSplit``; return None on failure."""
    if tag_matrix.shape[1] == 0 or n_samples < 4:
        return None
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, rest_idx = next(
            msss.split(np.zeros((n_samples, 1)), tag_matrix)
        )
        return train_idx, rest_idx
    except Exception as exc:
        logger.warning("iterstrat failed, falling back to hash split: %s", exc)
        return None


def _try_stratified_shuffle(
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Single-label ``StratifiedShuffleSplit``; return None on failure."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, rest_idx = next(sss.split(np.zeros(len(labels)), labels))
        return train_idx, rest_idx
    except Exception as exc:
        logger.warning("StratifiedShuffleSplit failed: %s", exc)
        return None


def _split_language_group(
    group_records: list[dict[str, Any]],
    group_features: list[dict[str, str]],
    ratios: dict[str, float],
    random_state: int,
    min_class_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    """Stratify a single language group into train/val/test."""
    n = len(group_records)
    if n < 4:
        # Too few for stratification — use hash split
        return _hash_split(group_records, ratios)

    # Build multi-label tag matrix from domain + difficulty + tier
    domains = [f["domain"] for f in group_features]
    difficulties = [f["difficulty"] for f in group_features]
    tiers = [f["tier"] for f in group_features]

    # Collapse rare classes
    domains_c = _collapse_rare(domains, min_class_samples)
    difficulties_c = _collapse_rare(difficulties, min_class_samples)
    tiers_c = _collapse_rare(tiers, min_class_samples)

    # Build combined label set for multi-label stratification
    # Each sample gets labels: domain_*, difficulty_*, tier_*
    all_labels: list[set[str]] = []
    for i in range(n):
        labels: set[str] = set()
        labels.add(f"domain:{domains_c[i]}")
        labels.add(f"diff:{difficulties_c[i]}")
        labels.add(f"tier:{tiers_c[i]}")
        all_labels.append(labels)

    # Binarize
    unique_labels = sorted({lbl for s in all_labels for lbl in s})
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    tag_matrix = np.zeros((n, len(unique_labels)), dtype=int)
    for i, s in enumerate(all_labels):
        for lbl in s:
            tag_matrix[i, label_to_idx[lbl]] = 1

    # First split: train vs rest
    test_size_first = 1.0 - ratios["train"]
    result = _try_iterstrat(n, tag_matrix, test_size_first, random_state)

    if result is None:
        # Fallback to single-label stratification on difficulty
        labels_arr = np.array(
            [f"{d}|{t}" for d, t in zip(difficulties_c, tiers_c, strict=True)]
        )
        result = _try_stratified_shuffle(labels_arr, test_size_first, random_state)

    if result is None:
        return _hash_split(group_records, ratios)

    train_idx, rest_idx = result

    # Second split: val vs test (50/50 of rest)
    rest_records = [group_records[i] for i in rest_idx]
    rest_features = [group_features[i] for i in rest_idx]
    rest_domains = [f["domain"] for f in rest_features]
    rest_difficulties = [f["difficulty"] for f in rest_features]
    rest_tiers = [f["tier"] for f in rest_features]

    rest_domains_c = _collapse_rare(rest_domains, min_class_samples)
    rest_difficulties_c = _collapse_rare(rest_difficulties, min_class_samples)
    rest_tiers_c = _collapse_rare(rest_tiers, min_class_samples)

    rest_all_labels: list[set[str]] = []
    for i in range(len(rest_records)):
        labels: set[str] = set()
        labels.add(f"domain:{rest_domains_c[i]}")
        labels.add(f"diff:{rest_difficulties_c[i]}")
        labels.add(f"tier:{rest_tiers_c[i]}")
        rest_all_labels.append(labels)

    rest_unique = sorted({lbl for s in rest_all_labels for lbl in s})
    rest_label_to_idx = {lbl: i for i, lbl in enumerate(rest_unique)}
    rest_matrix = np.zeros(
        (len(rest_records), len(rest_unique)), dtype=int
    )
    for i, s in enumerate(rest_all_labels):
        for lbl in s:
            rest_matrix[i, rest_label_to_idx[lbl]] = 1

    val_ratio_of_rest = ratios["val"] / (ratios["val"] + ratios["test"])
    test_size_second = 1.0 - val_ratio_of_rest

    rest_result = _try_iterstrat(
        len(rest_records), rest_matrix, test_size_second, random_state + 1
    )

    if rest_result is None:
        rest_labels_arr = np.array(
            [f"{d}|{t}" for d, t in zip(rest_difficulties_c, rest_tiers_c, strict=True)]
        )
        rest_result = _try_stratified_shuffle(
            rest_labels_arr, test_size_second, random_state + 1
        )

    if rest_result is None:
        # Last resort: hash split the rest
        rest_split = _hash_split(rest_records, {
            "train": 0.0,
            "val": val_ratio_of_rest,
            "test": 1.0 - val_ratio_of_rest,
        })
        return {
            "train": [group_records[i] for i in train_idx],
            "val": rest_split["val"],
            "test": rest_split["test"],
        }

    val_idx_local, test_idx_local = rest_result

    return {
        "train": [group_records[i] for i in train_idx],
        "val": [rest_records[i] for i in val_idx_local],
        "test": [rest_records[i] for i in test_idx_local],
    }


def stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    min_class_samples: int = MIN_CLASS_SAMPLES,
) -> dict[str, list[dict[str, Any]]]:
    """Multi-axis stratified split: domain, difficulty, language, tier.

    Groups by language → multi-label stratifies on (domain, difficulty, tier)
    tags → merges. Falls back to hash-split for tiny groups.

    Parameters
    ----------
    records
        List of JSONL record dicts (each with ``messages`` list).
    train_ratio, val_ratio, test_ratio
        Target split ratios (should sum to 1.0).
    random_state
        Seed for reproducibility.
    min_class_samples
        Classes with fewer samples are collapsed to ``__OTHER__``
        for stratification purposes (original tags preserved on records).

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` → list of records.
    """
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    if not records:
        return {"train": [], "val": [], "test": []}

    # Extract features for all records
    features = extract_features(records)

    # Group by language
    lang_groups: dict[str, list[int]] = defaultdict(list)
    for i, feat in enumerate(features):
        lang_groups[feat["language"]].append(i)

    # Split each language group independently
    merged: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for lang, indices in lang_groups.items():
        group_records = [records[i] for i in indices]
        group_features = [features[i] for i in indices]
        split_result = _split_language_group(
            group_records,
            group_features,
            ratios,
            random_state,
            min_class_samples,
        )
        for split_name in ("train", "val", "test"):
            merged[split_name].extend(split_result[split_name])

    logger.info(
        "Stratified split: %d train, %d val, %d test (languages: %s)",
        len(merged["train"]),
        len(merged["val"]),
        len(merged["test"]),
        ", ".join(sorted(lang_groups)),
    )

    return merged


# ---------------------------------------------------------------------------
# Integrity gates (B.7.6)
# ---------------------------------------------------------------------------

def _record_identity(record: dict[str, Any]) -> str:
    """Return a unique identity string for a record (content hash)."""
    return _content_hash(record)


def integrity_gates(
    splits: dict[str, list[dict[str, Any]]],
    target_ratios: dict[str, float] | None = None,
    tolerance: float = RATIO_TOLERANCE,
) -> dict[str, Any]:
    """Validate split integrity.

    Checks:
    1. **Hash-disjoint** — no record appears in multiple splits.
    2. **Ratio** — actual split ratios within ±tolerance of targets.
    3. **Domain balance** — marginal domain proportions within ±tolerance.
    4. **Language balance** — marginal language proportions within ±tolerance.

    Returns
    -------
    dict with ``passed`` (bool), ``checks`` (dict of check_name → bool),
    and ``details`` (dict with specifics).
    """
    target_ratios = target_ratios or DEFAULT_RATIOS
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    train_recs = splits.get("train", [])
    val_recs = splits.get("val", [])
    test_recs = splits.get("test", [])
    total = len(train_recs) + len(val_recs) + len(test_recs)

    if total == 0:
        return {
            "passed": True,
            "checks": {"hash_disjoint": True, "ratio": True, "domain_balance": True, "language_balance": True},
            "details": {"message": "empty splits"},
        }

    # 1. Hash-disjoint
    train_ids = {_record_identity(r) for r in train_recs}
    val_ids = {_record_identity(r) for r in val_recs}
    test_ids = {_record_identity(r) for r in test_recs}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    disjoint = not (overlap_tv or overlap_tt or overlap_vt)
    checks["hash_disjoint"] = disjoint
    if not disjoint:
        details["overlaps"] = {
            "train_val": len(overlap_tv),
            "train_test": len(overlap_tt),
            "val_test": len(overlap_vt),
        }

    # 2. Ratio tolerance
    actual_ratios = {
        "train": len(train_recs) / total,
        "val": len(val_recs) / total,
        "test": len(test_recs) / total,
    }
    ratio_ok = all(
        abs(actual_ratios[k] - target_ratios.get(k, 0.0)) <= tolerance
        for k in ("train", "val", "test")
    )
    checks["ratio"] = ratio_ok
    details["actual_ratios"] = actual_ratios
    details["target_ratios"] = target_ratios

    # 3. Domain balance (train vs full)
    full_domains = Counter(classify_domain(r) for r in [*train_recs, *val_recs, *test_recs])
    train_domains = Counter(classify_domain(r) for r in train_recs)
    domain_ok = True
    domain_details: dict[str, float] = {}
    for domain, full_count in full_domains.items():
        full_prop = full_count / total
        train_prop = train_domains.get(domain, 0) / max(len(train_recs), 1)
        diff = abs(full_prop - train_prop)
        domain_details[domain] = diff
        if diff > tolerance:
            domain_ok = False
    checks["domain_balance"] = domain_ok
    details["domain_deviations"] = domain_details

    # 4. Language balance (train vs full)
    full_langs = Counter(classify_language(r) for r in [*train_recs, *val_recs, *test_recs])
    train_langs = Counter(classify_language(r) for r in train_recs)
    lang_ok = True
    lang_details: dict[str, float] = {}
    for lang, full_count in full_langs.items():
        full_prop = full_count / total
        train_prop = train_langs.get(lang, 0) / max(len(train_recs), 1)
        diff = abs(full_prop - train_prop)
        lang_details[lang] = diff
        if diff > tolerance:
            lang_ok = False
    checks["language_balance"] = lang_ok
    details["language_deviations"] = lang_details

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point: stratified split of a JSONL dataset."""
    parser = argparse.ArgumentParser(
        description="Multi-axis stratified dataset split"
    )
    parser.add_argument("input_path", help="Input JSONL file or directory")
    parser.add_argument("output_dir", help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class", type=int, default=MIN_CLASS_SAMPLES)
    parser.add_argument("--skip-gates", action="store_true", help="Skip integrity gates")
    args = parser.parse_args()

    # Load records
    input_path = Path(args.input_path)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        files = [input_path]

    records: list[dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON in %s: %s", f, exc)

    if not records:
        print("No records found.")
        return 1

    print(f"Loaded {len(records)} records from {len(files)} file(s).")

    # Split
    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        min_class_samples=args.min_class,
    )

    # Integrity gates
    if not args.skip_gates:
        gates = integrity_gates(
            splits,
            target_ratios={
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
        )
        if gates["passed"]:
            print("Integrity gates: PASSED")
        else:
            print("Integrity gates: FAILED")
            for check, ok in gates["checks"].items():
                if not ok:
                    print(f"  {check}: FAILED — {gates['details']}")
            print("  Continuing anyway (use --skip-gates to suppress).")

    # Write output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for record in split_records:
                fh.write(json.dumps(record) + "\n")
        print(f"  {split_name}: {len(split_records)} records → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
